"""Light tests for experiments/nh81_joules (NH-81, issue #14)."""

from __future__ import annotations

import plistlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "nh81_joules"))

import analyze  # noqa: E402
import data  # noqa: E402
import power  # noqa: E402


def test_split_rule_is_10_percent_held_out():
    kinds = [data.split_of_block(i) for i in range(200)]
    assert kinds.count("val") == 10 and kinds.count("test") == 10


def test_strip_gutenberg():
    txt = "head\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nbody\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicence"
    assert data.strip_gutenberg(txt) == "body"


def _log(step, val, test):
    return {"step": step, "val_nats": val, "test_nats": test}


def test_select_dense_matches_on_val():
    dense = [_log(250, 6.0, 6.0), _log(500, 5.0, 5.01), _log(750, 4.5, 4.5)]
    spk = [_log(12000, 5.02, 5.0)]
    sel = analyze.select_dense(dense, spk)
    assert sel["dense_star_step"] == 500
    assert sel["iso_quality"] and sel["iso_direction"] == "iso"
    sel = analyze.select_dense([_log(250, 4.0, 4.0)], spk)
    assert sel["iso_direction"] == "spiking_worse"


def test_decision_rule():
    ok = {"pairs_valid": 12}
    assert analyze.decide({**ok, "R_hat": 0.8, "R_ci95": (0.7, 0.9)}, "iso") == "withdraw"
    assert analyze.decide({**ok, "R_hat": 0.8, "R_ci95": (0.7, 0.9)}, "spiking_worse") == "withdraw"
    assert analyze.decide({**ok, "R_hat": 0.8, "R_ci95": (0.7, 0.9)}, "spiking_better").startswith("inconclusive")
    assert analyze.decide({**ok, "R_hat": 2.5, "R_ci95": (1.5, 3.0)}, "iso") == "reopen"
    assert analyze.decide({**ok, "R_hat": 2.5, "R_ci95": (1.0, 3.0)}, "iso") == "inconclusive"
    assert analyze.decide({**ok, "R_hat": 1.5, "R_ci95": (1.2, 1.8)}, "iso") == "inconclusive"
    assert analyze.decide({"pairs_valid": 9, "R_hat": 0.5, "R_ci95": (0.4, 0.6)}, "iso").startswith("inconclusive")


def _rest(pair, util=0.0, busy=()):
    return {"kind": "rest", "workload": "W1", "path": "fused", "pair": pair, "dur_s": 20.0,
            "gpu_j": 2.0, "ane_j": 0.0, "cpu_j": 0.0, "gpu_util_mean": util, "busy_python": list(busy)}


def _run(pair, arm, gpu_j, tokens=1000):
    return {"kind": "run", "workload": "W1", "path": "fused", "pair": pair, "arm": arm,
            "dur_s": 10.0, "gpu_j": gpu_j, "ane_j": 0.0, "cpu_j": 0.0, "tokens": tokens,
            "vm_cpu_pct": 300.0}


def test_pairs_subtract_rest_and_apply_rejection():
    recs = [_rest(0), _run(0, "dense", 101.0), _run(0, "fused", 201.0),
            _rest(1, util=50.0), _run(1, "dense", 101.0), _run(1, "fused", 201.0), _rest(2)]
    pairs = analyze.build_pairs(recs, "W1", "fused")
    # rest power 0.1 W -> 1 J over 10 s; dense 100 J / 1000 tok, spiking 200 J
    assert pairs[0]["valid"] is False  # rest after pair 0 is busy
    assert pairs[1]["valid"] is False  # rest before pair 1 is busy
    assert pytest.approx(pairs[0]["dense_j_per_tok"]) == 0.1
    assert pytest.approx(pairs[0]["spiking_j_per_tok"]) == 0.2
    recs[3] = _rest(1)
    pairs = analyze.build_pairs(recs, "W1", "fused")
    assert all(p["valid"] for p in pairs)
    st = analyze.summarize(pairs)
    assert pytest.approx(st["R_hat"]) == 0.5


def test_failed_utilisation_read_rejects_pair():
    recs = [_rest(0, util=-1.0), _run(0, "dense", 101.0), _run(0, "fused", 201.0), _rest(1)]
    assert analyze.build_pairs(recs, "W1", "fused")[0]["valid"] is False


def test_powermetrics_parser_and_window():
    ts = datetime(2026, 10, 25, 20, 0, 0, tzinfo=timezone.utc)
    blob = b"\x00".join(
        plistlib.dumps({"elapsed_ns": 100_000_000, "timestamp": ts.replace(microsecond=k * 100_000),
                        "gpu": {"gpu_energy": 500.0}, "processor": {"cpu_energy": 100.0}})
        for k in range(1, 6))
    s = power.parse_powermetrics_plist(blob + b"\x00<trunc")
    assert len(s) == 5 and s[0].gpu_j == pytest.approx(0.5)
    t0 = ts.timestamp()
    # 1-s plist dates: samples end at t0+0.1..0.5 once re-anchored (+/- 0.5 s
    # anchor uncertainty here, so test relative to the first sample start)
    start = s[0].t_end - s[0].dt
    w = power.window_energy(s, start + 0.05, start + 0.25)  # 1/2 + 1 + 1/2 samples
    assert w["gpu_j"] == pytest.approx(1.0)
    assert s[-1].t_end - s[0].t_end == pytest.approx(0.4)
    assert t0 <= s[0].t_end <= t0 + 1.0


def test_rail_energy_reads_one_channel_per_rail():
    r = power.rail_energy({"GPU Energy": 2.0, "GPU0": 2.0, "CPU Energy": 1.0, "ANE0": 0.0})
    assert r["gpu_j"] == 2.0 and r["cpu_j"] == 1.0


def test_foreign_gpu_process_rejects_pair_and_replay_indexing():
    # attempt 0 sees a foreign compute pid in its spiking run -> rejected;
    # attempt 1 is the replay; attempt 2 was paused (rest only, no runs).
    r0 = _run(0, "fused", 201.0)
    r0["foreign_gpu_pids"] = [4242]
    recs = [_rest(0), _run(0, "dense", 101.0), r0, _rest(1),
            _run(1, "dense", 101.0), _run(1, "fused", 201.0), _rest(2), _rest(3)]
    pairs = analyze.build_pairs(recs, "W1", "fused")
    assert [p["pair"] for p in pairs] == [0, 1]
    assert pairs[0]["valid"] is False and pairs[1]["valid"] is True
    rest_bad = _rest(2)
    rest_bad["foreign_gpu_pids"] = [4242]
    assert analyze.rest_ok(_rest(2)) and not analyze.rest_ok(rest_bad)
    assert pytest.approx(pairs[1]["ratio_gross"]) == 101.0 / 201.0


def test_busy_between_uses_window_ticks():
    import power_nvml

    hz = __import__("os").sysconf("SC_CLK_TCK")
    a = {10: ("python a", 0), 11: ("python b", 0), 12: ("python c", 0)}
    b = {10: ("python a", int(0.5 * hz * 10)), 11: ("python b", int(0.1 * hz * 10)),
         13: ("python new", 999)}
    rows = power_nvml.busy_between(a, b, 10.0, exclude_pid=99)
    assert [r[1] for r in rows] == [10]


def test_lif_matches_numpy_reference():
    mx = pytest.importorskip("mlx.core")
    import model_mlx

    x = np.random.default_rng(0).uniform(0, 1.5, size=(4, 7)).astype(np.float32)
    got = np.stack([np.array(e) for e in model_mlx.lif_emissions(mx.array(x), 6, train=False)])
    v, ref = np.zeros_like(x), []
    for _ in range(6):
        v = v + x
        a = (v > 0.5).astype(np.float32)
        ref.append(a)
        v = v - a * 0.5
    np.testing.assert_array_equal(got, np.stack(ref))


def test_fused_and_event_paths_agree():
    mx = pytest.importorskip("mlx.core")
    import model_mlx

    mx.random.seed(0)
    m = model_mlx.LM(vocab=64, d=64, n_layer=2, seq=16, arm="spkt", ticks=6)
    x = mx.array(np.random.default_rng(1).integers(0, 64, size=(2, 16)))
    a, b = m(x, mode="fused"), m(x, mode="event")
    np.testing.assert_allclose(np.array(a), np.array(b), rtol=1e-4, atol=1e-4)
