"""Light tests for experiments/demo_35b_gb300 (no GPU, no MLX)."""

from __future__ import annotations

import ast
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "demo_35b_gb300"))
sys.path.insert(0, str(ROOT / "experiments" / "nh81_gb300"))

import demo_rules as D  # noqa: E402
import gb300_rules as R  # noqa: E402

PROMPTS = ROOT / "docs" / "prereg" / "demo-35b-lora-gb300" / "prompts.json"


def test_frozen_prompts_load_and_match_domains():
    by = D.load_prompts(str(PROMPTS))
    assert list(by) == D.DOMAINS
    assert all(len(v) == 3 for v in by.values())


def test_calibrate_k():
    assert D.calibrate_k(4.0) == 4
    assert D.calibrate_k(20.0) == 1
    with pytest.raises(ValueError):
        D.calibrate_k(0.0)


def test_check_adapter_exact_missing_extra_shape_dtype():
    want = {"a.lora_a": ((2048, 16), "float32"), "a.lora_b": ((16, 512), "float32")}
    assert D.check_adapter(dict(want), want) == []
    got = {"a.lora_a": ((2048, 8), "float32"), "x.lora_b": ((16, 512), "bfloat16")}
    probs = D.check_adapter(got, want)
    assert "missing a.lora_b" in probs and "extra x.lora_b" in probs
    assert any(p.startswith("shape a.lora_a") for p in probs)


def test_jpt_and_zero_tokens():
    rest = {"gpu_j": 200.0, "dur_s": 20.0}
    assert D.jpt({"gpu_j": 400.0, "dur_s": 20.0, "generated_tokens": 100}, rest) == pytest.approx(2.0)
    assert D.jpt({"gpu_j": 400.0, "dur_s": 20.0, "generated_tokens": 0}, rest) is None


def _run(d, arm, j, n=100, dur=20.0, valid=True):
    return {"domain": d, "arm": arm, "valid": valid, "j_per_token": j, "gpu_j": 400.0,
            "generated_tokens": n, "dur_s": dur}


def test_summarize_medians_ratio_and_exclusions():
    runs = [_run("python", "base", x) for x in (1.0, 2.0, 3.0)]
    runs += [_run("python", "adapter", x) for x in (2.0, 4.0, -1.0)]
    runs += [_run("python", "adapter", 9.0, valid=False), _run("cpp", "base", 1.0, dur=10.0)]
    s = D.summarize(runs)
    c = s["cells"]
    assert c["python|base"]["j_per_token_median"] == 2.0
    assert c["python|adapter"]["excluded_nonpositive"] == 1
    assert c["python|adapter"]["j_per_token_median"] == 3.0
    assert s["ratio_of_medians_adapter_over_base"]["python"] == pytest.approx(1.5)
    assert c["cpp|base"]["short_runs"] == 1
    assert s["status"] == "demo, not a scientific claim"


def _demo_function(name, **env):
    """Load the actual function body without importing MLX or the frozen code."""
    path = ROOT / "experiments" / "demo_35b_gb300" / "run_demo.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"D": D, "R": R, "json": json, **env}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns[name]


class _Thread:
    def __init__(self, target, daemon):
        self.target = target

    def start(self):
        try:
            self.target()
        except Exception:
            pass  # Worker exceptions do not propagate through Thread.start/join.

    def join(self):
        pass


def _watch_error(*_):
    raise RuntimeError("watch failed")


def test_p0_rejects_unclean_sampler():
    checked = []
    s = SimpleNamespace(start=lambda: None, stop=threading.Event(),
                        clean=lambda a, b: checked.append((a, b)) or False,
                        between=lambda *_: ([], []))
    rp = SimpleNamespace(exclusivity=lambda _: True,
                         qualify_counter=lambda _: {"instrument": "power_integrated"},
                         Sampler=lambda *_: s)
    probe = _demo_function("probe_before_model", RP=rp,
                           time=SimpleNamespace(time=iter([100.0, 220.0]).__next__, sleep=lambda _: None))
    with pytest.raises(SystemExit, match="instrument fault during P0.*not run"):
        probe(object(), "unused")
    assert checked == [(100.0, 220.0)]


@pytest.mark.parametrize("guard", ["clean_reason", "cpu_window"])
def test_p1_rejects_watcher_exception(guard, capsys):
    rp = SimpleNamespace(P1_REST_S=0, clean_reason=lambda *_: "", cpu_window=lambda *_: "")
    setattr(rp, guard, _watch_error)
    p1 = _demo_function("p1_once", RP=rp, R=SimpleNamespace(STAB_S=0), p=object(),
                        sampler=SimpleNamespace(clean=lambda *_: True),
                        threading=SimpleNamespace(Event=threading.Event, Thread=_Thread),
                        time=SimpleNamespace(time=lambda: 100.0, sleep=lambda _: None))
    assert p1() is None
    assert "watch failed" in capsys.readouterr().out


def _clean_rest():
    return {"rest_rule": "gb300", "gpu_j": 200.0, "dur_s": 20.0, "rest_max_w": 125.0,
            "busy_python": [], "load1": 0.0, "load_max": 4.0, "host_foreign_cores": 0.0,
            "host_cores_max": 1.0, "foreign_gpu_pids": [], "instrument_invalid": False,
            "cap_changed": False}


def test_window_rejects_watcher_exception():
    extra = _clean_rest()
    p = SimpleNamespace(foreign_gpu_pids=_watch_error, python_ticks=lambda: {},
                        busy_between=lambda *_: [], window_extra=lambda: extra)
    snapshots = iter([(100.0, {"GPU Energy": 0.0}), (120.0, {"GPU Energy": 200.0})])
    window = _demo_function("window", p=p, meter=SimpleNamespace(snapshot=snapshots.__next__),
                            mx=SimpleNamespace(synchronize=lambda: None),
                            threading=SimpleNamespace(Event=threading.Event, Thread=_Thread))
    rec, payload = window(lambda: "generated")
    assert payload == "generated"
    assert any("watch failed" in e for e in rec["watch_errors"])
    assert not R.window_ok_gb300(rec)
    assert not R.rest_ok_gb300(rec)


@pytest.mark.parametrize("dur, valid, attempts", [(14.9, False, 3), (15.0, True, 1), (15.1, True, 1)])
def test_block_short_runs_do_not_fill_quota(monkeypatch, dur, valid, attempts):
    monkeypatch.setattr(D, "VALID_RUNS", 1)
    monkeypatch.setattr(D, "MAX_ATTEMPTS", 3)
    logged = []
    block = _demo_function("block", gen=lambda *_: None, ids_of=lambda _: [],
                           prompts={"python": [{"text": "test"}]}, wait_clean=lambda: None,
                           rest=_clean_rest, window=lambda _: ({"dur_s": dur, "gpu_j": 400.0}, (100, [])),
                           out=SimpleNamespace(run=lambda **r: logged.append(r)))
    block("python", "base", 1)
    assert len(logged) == attempts
    assert all(r["valid"] is valid and r["short"] is (not valid) for r in logged)


@pytest.mark.parametrize("short_valid", [False, True])
def test_summarize_counts_short_runs_separately(short_valid):
    runs = [_run("python", "base", 9.0, dur=14.9, valid=short_valid) for _ in range(3)]
    runs += [_run("python", "base", 2.0, dur=D.MIN_RUN_S),
             _run("python", "base", 99.0, valid=False)]
    cell = D.summarize(runs)["cells"]["python|base"]
    assert cell["short_runs"] == 3
    assert cell["valid_runs"] == 1
    assert cell["short_j_per_token"] == [9.0] * 3
    assert cell["j_per_token_median"] == 2.0


def test_download_retries_share_remaining_deadline():
    script = (ROOT / "experiments" / "demo_35b_gb300" / "demo_launcher.sh").read_text()
    fetch = script.split("fetch() {", 1)[1].split("grep -v", 1)[0]
    assert 'local left=$(( DEADLINE - $(date +%s) ))' in fetch
    assert 'timeout --signal=TERM --kill-after=5 "$left" curl -fL --retry 3' in fetch
