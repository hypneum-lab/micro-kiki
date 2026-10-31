"""Light tests for experiments/nh81_gb300 (no GPU, no MLX, no NVML)."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "nh81_gb300"))

import gb300_rules as R  # noqa: E402

TAG = "prereg-nh81-m3pro-20261030"


def test_threshold_is_k_times_rest():
    assert R.threshold(100.0) == pytest.approx(125.0)
    with pytest.raises(ValueError):
        R.threshold(0.0)


def test_max_sliding_mean():
    t = [float(i) for i in range(61)]
    e, acc = [], 0.0
    for i in range(61):
        e.append(acc)
        acc += 100.0 if 30 <= i < 50 else 50.0  # 20 s at 100 W inside 50 W
    assert R.max_sliding_mean(t, e, 20.0) == pytest.approx(100.0)
    with pytest.raises(ValueError):
        R.max_sliding_mean(t[:5], e[:5], 20.0)


def test_bus_id_and_uuid():
    assert R.parse_bus_id("00000009:01:00.0") == R.parse_bus_id("0009:01:00.0")
    assert R.uuid_from_bytes(bytes(range(16))) == "GPU-00010203-0405-0607-0809-0a0b0c0d0e0f"


def test_cpu_busy_counts_new_processes_from_zero():
    a = {1: ("x", 100)}
    b = {1: ("x", 101), 2: ("new", 50)}  # pid 2 appeared during the window
    rows = R.cpu_busy(a, b, dt=1.0, hz=100, exclude=set())
    assert [r[1] for r in rows] == [2]


def test_host_foreign_cores():
    assert R.host_foreign_cores(0, 300, 0, 100, hz=100, dt=1.0) == pytest.approx(2.0)


def _rest(w, **kw):
    r = {"rest_rule": "gb300", "gpu_j": w * 20.0, "dur_s": 20.0, "rest_max_w": 125.0,
         "busy_python": [], "load1": 1.0, "load_max": 4.0, "host_foreign_cores": 0.1,
         "host_cores_max": 1.0, "foreign_gpu_pids": [], "instrument_invalid": False,
         "cap_changed": False}
    r.update(kw)
    return r


def test_rest_ok_boundaries():
    assert R.rest_ok_gb300(_rest(125.0))
    assert not R.rest_ok_gb300(_rest(125.1))
    assert not R.rest_ok_gb300(_rest(10.0, load1=4.5))
    assert not R.rest_ok_gb300(_rest(10.0, host_foreign_cores=1.5))
    assert not R.rest_ok_gb300(_rest(10.0, busy_python=["x"]))
    assert not R.rest_ok_gb300(_rest(10.0, foreign_gpu_pids=[-1]))
    assert not R.rest_ok_gb300(_rest(10.0, instrument_invalid=True))
    assert not R.rest_ok_gb300(_rest(10.0, cap_changed=True))


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    """Frozen experiments/nh81_joules at the tag, exported read-only."""
    if subprocess.run(["git", "-C", str(ROOT), "rev-parse", "-q", "--verify", TAG],
                      capture_output=True).returncode != 0:
        pytest.skip(f"tag {TAG} not fetched")
    d = tmp_path_factory.mktemp("frozen")
    tar = subprocess.run(["git", "-C", str(ROOT), "archive", TAG, "experiments/nh81_joules"],
                         capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(d)], input=tar, check=True)
    return d / "experiments" / "nh81_joules"


def _records(nonpositive_pair: int | None = None):
    recs = []
    for i in range(12):
        recs.append({"kind": "rest", "workload": "W1", "path": "fused", "pair": i,
                     **_rest(10.0), "ane_j": 0.0, "cpu_j": 0.0})
        for arm, j in (("dense", 400.0), ("fused", 600.0)):
            if i == nonpositive_pair and arm == "dense":
                j = 100.0  # below rest x duration: non-positive net energy
            recs.append({"kind": "run", "workload": "W1", "path": "fused", "pair": i, "arm": arm,
                         "gpu_j": j, "ane_j": 0.0, "cpu_j": 0.0, "dur_s": 20.0, "tokens": 1000,
                         "vm_cpu_pct": 0.0, "foreign_gpu_pids": [], "instrument_invalid": False,
                         "cap_changed": False})
    recs.append({"kind": "rest", "workload": "W1", "path": "fused", "pair": 12,
                 **_rest(10.0), "ane_j": 0.0, "cpu_j": 0.0})
    return recs


def test_describe_excludes_and_counts_nonpositive(frozen, monkeypatch):
    monkeypatch.setenv("NH81_CODE_DIR", str(frozen))
    monkeypatch.syspath_prepend(str(frozen))
    sys.modules.pop("analyze", None)
    sys.modules.pop("describe", None)
    describe = importlib.import_module("describe")
    st = describe.describe_block(describe.analyze.build_pairs(_records(3), "W1", "fused"))
    assert st["pairs_valid"] == 12
    assert st["pairs_nonpositive_net_excluded"] == 1
    assert st["R_hat"] == pytest.approx((400 - 200) / (600 - 200))
    assert "decision" not in st and "catalogue" not in st


def test_power_module_imports_without_nvml(frozen, monkeypatch):
    monkeypatch.syspath_prepend(str(frozen))
    sys.modules.pop("power_gb300", None)
    p = importlib.import_module("power_gb300")
    assert p.set_baseline() == []
    a = {10: ("x", 0)}
    assert p.busy_between(a, {10: ("x", 100), 11: ("y", 1)}, 1.0, exclude_pid=10) == []
