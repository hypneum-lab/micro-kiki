"""NVML energy instrument and Linux host guards for NH-81 (KXKM RTX 4090).

Drop-in replacement for ``power`` on Linux/CUDA (replication prereg
``docs/prereg/2026-10-28-nh81-kxkm-replication.md``), selected in
``measure.py`` with ``NH81_HOST=kxkm``:

* ``EnergyMeter`` reads ``nvmlDeviceGetTotalEnergyConsumption`` (cumulative
  board energy, mJ) through ``ctypes`` on the driver's ``libnvidia-ml.so.1``
  -- no extra Python dependency. Its snapshots use the same shape as the
  IOReport reader (``(t, {"GPU Energy": J})``), so ``power.rail_energy``
  and ``analyze`` are reused unchanged.
* ``gpu_utilization`` and ``foreign_gpu_pids`` come from NVML.
* ``busy_python`` measures %CPU over a short interval from ``/proc``
  (Linux ``ps %cpu`` is a lifetime average, useless as a guard).
* ``n10c_active`` and ``pause_reason`` implement the pause rule.
"""

from __future__ import annotations

import ctypes
import glob
import os
import subprocess
import time

from power import rail_energy  # noqa: F401  (re-exported for measure.py)

_NVML = None
_HANDLE = None
N10C_DIR = os.environ.get("NH81_N10C_DIR", "/ai/saisail/n10c-run/campaign")


class _ProcInfo(ctypes.Structure):
    # nvmlProcessInfo_t as used by GetComputeRunningProcesses_v3 (24 bytes):
    # pid, usedGpuMemory, gpuInstanceId, computeInstanceId.
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]


class _Util(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _Mem(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


def _nv():
    global _NVML, _HANDLE
    if _NVML is None:
        lib = ctypes.CDLL("libnvidia-ml.so.1")
        if lib.nvmlInit_v2() != 0:
            raise RuntimeError("nvmlInit_v2 failed")
        h = ctypes.c_void_p()
        idx = int(os.environ.get("NH81_GPU_INDEX", "0"))
        if lib.nvmlDeviceGetHandleByIndex_v2(idx, ctypes.byref(h)) != 0:
            raise RuntimeError("nvmlDeviceGetHandleByIndex_v2 failed")
        _NVML, _HANDLE = lib, h
    return _NVML, _HANDLE


def total_energy_j() -> float:
    lib, h = _nv()
    e = ctypes.c_ulonglong()
    rc = lib.nvmlDeviceGetTotalEnergyConsumption(h, ctypes.byref(e))
    if rc != 0:
        raise RuntimeError(f"nvmlDeviceGetTotalEnergyConsumption rc={rc}")
    return e.value * 1e-3


class EnergyMeter:
    """Cumulative NVML board energy, IOReport-compatible snapshots."""

    def __init__(self) -> None:
        total_energy_j()

    def snapshot(self) -> tuple[float, dict[str, float]]:
        e = total_energy_j()
        return time.time(), {"GPU Energy": e}

    @staticmethod
    def delta_joules(a, b) -> dict[str, float]:
        return {k: b[1][k] - a[1].get(k, 0.0) for k in b[1]}


def gpu_utilization() -> int:
    lib, h = _nv()
    u = _Util()
    return int(u.gpu) if lib.nvmlDeviceGetUtilizationRates(h, ctypes.byref(u)) == 0 else -1


def gpu_compute_pids() -> list[int]:
    lib, h = _nv()
    n = ctypes.c_uint(64)
    buf = (_ProcInfo * 64)()
    for fn in ("nvmlDeviceGetComputeRunningProcesses_v3", "nvmlDeviceGetComputeRunningProcesses_v2"):
        f = getattr(lib, fn, None)
        if f is None:
            continue
        n = ctypes.c_uint(64)
        if f(h, ctypes.byref(n), buf) == 0:
            pids = sorted(int(buf[i].pid) for i in range(n.value))
            if all(os.path.exists(f"/proc/{p}") for p in pids):
                return pids
            break  # implausible pid: layout mismatch, use nvidia-smi
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    return sorted(int(x) for x in out.split() if x.strip().isdigit())


BASELINE: set[int] = set()


def set_baseline() -> list[int]:
    """Freeze B, the compute processes present when the start guard passes."""
    BASELINE.clear()
    BASELINE.update(p for p in gpu_compute_pids() if p != os.getpid())
    return sorted(BASELINE)


def foreign_gpu_pids() -> set[int]:
    return {p for p in gpu_compute_pids() if p != os.getpid() and p not in BASELINE}


def gpu_state() -> dict:
    """Descriptive GPU state: board power, SM clock, temperature, P-state, memory."""
    lib, h = _nv()
    p, c, t, ps = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint(), ctypes.c_int()
    m = _Mem()
    lib.nvmlDeviceGetPowerUsage(h, ctypes.byref(p))
    lib.nvmlDeviceGetClockInfo(h, 1, ctypes.byref(c))          # NVML_CLOCK_SM
    lib.nvmlDeviceGetTemperature(h, 0, ctypes.byref(t))        # NVML_TEMPERATURE_GPU
    lib.nvmlDeviceGetPerformanceState(h, ctypes.byref(ps))
    lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(m))
    return {"power_w": p.value / 1000, "sm_mhz": c.value, "temp_c": t.value,
            "pstate": ps.value, "mem_used_mb": m.used / 2**20}


def _python_cpu_ticks() -> dict[int, tuple[str, int]]:
    out = {}
    for d in glob.glob("/proc/[0-9]*"):
        try:
            with open(d + "/stat") as f:
                s = f.read()
            comm = s[s.index("(") + 1:s.rindex(")")]
            fields = s[s.rindex(")") + 2:].split()
            if "python" not in comm.lower():
                continue
            with open(d + "/cmdline", "rb") as f:
                cmd = f.read().split(b"\0")[0].decode(errors="replace")
            out[int(d[6:])] = (cmd or comm, int(fields[11]) + int(fields[12]))
        except (OSError, ValueError, IndexError):
            continue
    return out


def python_ticks() -> dict[int, tuple[str, int]]:
    """CPU ticks of every Python process (start/end of a window)."""
    return _python_cpu_ticks()


def busy_between(a: dict, b: dict, dt: float, threshold: float = 20.0,
                 exclude_pid: int | None = None) -> list[tuple[float, int, str]]:
    """Other Python processes above ``threshold`` %CPU between two tick maps."""
    me = exclude_pid or os.getpid()
    hz = os.sysconf("SC_CLK_TCK")
    rows = []
    for pid, (cmd, ticks) in b.items():
        if pid == me or pid not in a or dt <= 0:
            continue
        pct = 100.0 * (ticks - a[pid][1]) / hz / dt
        if pct > threshold:
            rows.append((pct, pid, cmd))
    return rows


def busy_python(threshold: float = 20.0, exclude_pid: int | None = None,
                interval: float = 1.0) -> list[tuple[float, int, str]]:
    """Other Python processes above ``threshold`` %CPU over ``interval`` s."""
    a = _python_cpu_ticks()
    time.sleep(interval)
    return busy_between(a, _python_cpu_ticks(), interval, threshold, exclude_pid)


def vm_cpu_percent() -> float:
    """No OrbStack VM on KXKM; kept for log compatibility (always 0)."""
    return 0.0


def n10c_active(campaign_dir: str = N10C_DIR, minutes: float = 10.0) -> str:
    pg = subprocess.run(["pgrep", "-f", "run_campaign"], capture_output=True, text=True).stdout.split()
    pg = [p for p in pg if int(p) != os.getpid()]
    if pg:
        return f"run_campaign pids {pg}"
    now = time.time()
    for f in glob.glob(campaign_dir + "/**", recursive=True):
        try:
            if os.path.isfile(f) and now - os.path.getmtime(f) < minutes * 60:
                return f"recent campaign file {f}"
        except OSError:
            pass
    return ""


def pause_reason() -> str:
    """Non-empty if the pair must wait (prereg section 4, pause rule)."""
    r = n10c_active()
    if r:
        return f"N10-C: {r}"
    fp = foreign_gpu_pids()
    if fp:
        return f"foreign GPU compute pids {sorted(fp)}"
    u = gpu_utilization()
    if u < 0 or u >= 10:  # -1: failed read, fail closed
        return f"GPU utilisation {u} %"
    return ""
