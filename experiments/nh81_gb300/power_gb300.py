"""NVML instrument and host guards for the NH-81 GB300 descriptive run.

Drop-in for ``power_nvml`` (same functions as used by the frozen
``measure.py`` at tag ``prereg-nh81-m3pro-20261030``), injected by
``measure_gb300.py``. Differences, all from the GB300 prereg:

* GPU handle by PCI bus ID (``NH81_GPU_PCI_BUS_ID``), never by index;
  MIG must be off; process queries restricted to that GPU, fail closed.
* Rest threshold T and instrument read from ``rest_threshold.json``
  (``NH81_REST_THRESHOLD``), never a constant.
* Instrument: NVML total-energy counter, or the pre-declared
  power-integration fallback; read faults invalidate the window and
  three faults end the run.
* CPU rule over all processes (not only Python), host CPU outside our
  process, 1-min load average; allowed-process list instead of a
  baseline frozen at guard time.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import threading
import time

import gb300_rules as R
from power import rail_energy  # noqa: F401  (re-exported for measure.py)

NVML_SUCCESS = 0
NVML_ERROR_NOT_SUPPORTED = 3
_NVML = None
_HANDLE = None
_UUID = ""
HZ = os.sysconf("SC_CLK_TCK")
ALLOWED = {x for x in os.environ.get("NH81_GB300_ALLOWED", "").split(",") if x}
PAUSE_PATTERNS = [x for x in os.environ.get(
    "NH81_GB300_PAUSE_PATTERNS", "run_campaign,run.sh,compare_nh01,procgen").split(",") if x]
FAULTS = 0                  # instrument faults since start
_FAULTS_SEEN = 0            # faults already attributed to a window
_SNAPS: list[dict] = []     # last meter snapshots (time, energy, cap)
_TICKS: list[dict] = []     # last CPU snapshots (stat, own)
_CFG: dict | None = None


class _ProcInfo(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]


class _Util(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _Mem(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


def cfg() -> dict:
    """rest_threshold.json written by rest_probe_gb300.py."""
    global _CFG
    if _CFG is None:
        path = os.environ.get("NH81_REST_THRESHOLD", "")
        _CFG = json.load(open(path)) if path else {}
    return _CFG


def rest_max_w() -> float:
    t = cfg().get("T_w")
    if not t:
        raise RuntimeError("NH81_REST_THRESHOLD missing or without T_w")
    return float(t)


def _nv():
    global _NVML, _HANDLE, _UUID
    if _NVML is None:
        bus = os.environ.get("NH81_GPU_PCI_BUS_ID", "")
        if not bus:
            raise RuntimeError("NH81_GPU_PCI_BUS_ID is required on the GB300")
        lib = ctypes.CDLL("libnvidia-ml.so.1")
        if lib.nvmlInit_v2() != NVML_SUCCESS:
            raise RuntimeError("nvmlInit_v2 failed")
        h = ctypes.c_void_p()
        if lib.nvmlDeviceGetHandleByPciBusId_v2(bus.encode(), ctypes.byref(h)) != NVML_SUCCESS:
            raise RuntimeError(f"no NVML device at PCI bus id {bus}")
        buf = ctypes.create_string_buffer(96)
        lib.nvmlDeviceGetUUID(h, buf, 96)
        _NVML, _HANDLE, _UUID = lib, h, buf.value.decode()
    return _NVML, _HANDLE


def uuid() -> str:
    _nv()
    return _UUID


def mig_mode() -> str:
    lib, h = _nv()
    cur, pend = ctypes.c_uint(), ctypes.c_uint()
    rc = lib.nvmlDeviceGetMigMode(h, ctypes.byref(cur), ctypes.byref(pend))
    if rc == NVML_ERROR_NOT_SUPPORTED:
        return "not_supported"
    if rc != NVML_SUCCESS:
        return f"error_{rc}"
    return "enabled" if cur.value else "disabled"


def device_name() -> str:
    lib, h = _nv()
    buf = ctypes.create_string_buffer(96)
    lib.nvmlDeviceGetName(h, buf, 96)
    return buf.value.decode()


def check_identity() -> dict:
    """In-process GPU identity check (prereg 4.3); raises on mismatch."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis != uuid():
        raise SystemExit(f"GPU selection mismatch: CUDA_VISIBLE_DEVICES={vis!r} != {uuid()}")
    mig = mig_mode()
    if mig == "enabled" or mig.startswith("error"):
        raise SystemExit(f"MIG mode {mig}: not measured")
    cu = ctypes.CDLL("libcuda.so.1")
    if cu.cuInit(0) != 0:
        raise SystemExit("cuInit failed")
    n = ctypes.c_int()
    cu.cuDeviceGetCount(ctypes.byref(n))
    if n.value != 1:
        raise SystemExit(f"GPU selection mismatch: {n.value} CUDA devices visible")
    dev = ctypes.c_int()
    cu.cuDeviceGet(ctypes.byref(dev), 0)
    bus = ctypes.create_string_buffer(64)
    cu.cuDeviceGetPCIBusId(bus, 64, dev)
    raw = (ctypes.c_ubyte * 16)()
    cu.cuDeviceGetUuid(raw, dev)
    cuda_uuid = R.uuid_from_bytes(bytes(raw))
    want_bus = R.parse_bus_id(os.environ["NH81_GPU_PCI_BUS_ID"])
    if R.parse_bus_id(bus.value.decode()) != want_bus or cuda_uuid != uuid():
        raise SystemExit(f"GPU selection mismatch: CUDA {bus.value.decode()} {cuda_uuid} "
                         f"vs NVML {os.environ['NH81_GPU_PCI_BUS_ID']} {uuid()}")
    return {"pci_bus_id": os.environ["NH81_GPU_PCI_BUS_ID"], "uuid": uuid(),
            "name": device_name(), "mig": mig, "cuda_pci_bus_id": bus.value.decode()}


# --------------------------------------------------------------------------
# Instrument
# --------------------------------------------------------------------------

def _fault(why: str) -> None:
    global FAULTS
    FAULTS += 1
    print(json.dumps({"kind": "instrument_fault", "why": why, "faults": FAULTS}), flush=True)


def counter_j() -> float | None:
    lib, h = _nv()
    e = ctypes.c_ulonglong()
    if lib.nvmlDeviceGetTotalEnergyConsumption(h, ctypes.byref(e)) != NVML_SUCCESS:
        return None
    return e.value * 1e-3


def board_power_w() -> float:
    lib, h = _nv()
    p = ctypes.c_uint()
    return p.value / 1000 if lib.nvmlDeviceGetPowerUsage(h, ctypes.byref(p)) == NVML_SUCCESS else -1.0


def enforced_cap_w() -> float:
    lib, h = _nv()
    p = ctypes.c_uint()
    return p.value / 1000 if lib.nvmlDeviceGetEnforcedPowerLimit(h, ctypes.byref(p)) == NVML_SUCCESS else -1.0


class _Integrator:
    """Power-integration fallback: trapezoid of nvmlDeviceGetPowerUsage every 0.1 s."""

    def __init__(self) -> None:
        self.e, self.t, self.p = 0.0, None, None
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            w, now = board_power_w(), time.time()
            with self.lock:
                if w < 0:
                    _fault("power read failed")
                elif self.t is not None:
                    if now - self.t > R.MAX_SAMPLE_GAP_S:
                        _fault(f"sample gap {now - self.t:.2f} s")
                    self.e += 0.5 * (w + self.p) * (now - self.t)
                if w >= 0:
                    self.t, self.p = now, w
            time.sleep(0.1)

    def value(self) -> float:
        with self.lock:
            return self.e


class EnergyMeter:
    """Cumulative GB300 energy with IOReport-compatible snapshots."""

    def __init__(self) -> None:
        self.mode = cfg().get("instrument", "counter")
        self.last = None
        self.integ = _Integrator() if self.mode == "power_integrated" else None
        if self.integ is not None:
            time.sleep(0.5)

    def _read(self) -> float:
        if self.integ is not None:
            return self.integ.value()
        e = counter_j()
        if e is None:
            _fault("counter read failed")
            return self.last if self.last is not None else 0.0
        if self.last is not None and e < self.last:
            _fault("counter decreased")
        self.last = e
        return e

    def snapshot(self) -> tuple[float, dict[str, float]]:
        e, t = self._read(), time.time()
        _SNAPS.append({"t": t, "e": e, "cap": enforced_cap_w()})
        del _SNAPS[:-2]
        return t, {"GPU Energy": e}

    @staticmethod
    def delta_joules(a, b) -> dict[str, float]:
        return {k: b[1][k] - a[1].get(k, 0.0) for k in b[1]}


# --------------------------------------------------------------------------
# Processes, CPU, guards
# --------------------------------------------------------------------------

def gpu_utilization() -> int:
    lib, h = _nv()
    u = _Util()
    return int(u.gpu) if lib.nvmlDeviceGetUtilizationRates(h, ctypes.byref(u)) == NVML_SUCCESS else -1


def _comm(pid: int) -> str:
    try:
        return open(f"/proc/{pid}/comm").read().strip()
    except OSError:
        return "?"


def gpu_pids() -> list[int] | None:
    """Compute and graphics processes on the GB300 only; None on failure."""
    lib, h = _nv()
    pids: set[int] = set()
    for fn in ("nvmlDeviceGetComputeRunningProcesses_v3", "nvmlDeviceGetGraphicsRunningProcesses_v3"):
        f = getattr(lib, fn, None)
        buf = (_ProcInfo * 256)()
        n = ctypes.c_uint(256)
        if f is None or f(h, ctypes.byref(n), buf) != NVML_SUCCESS:
            break
        got = [int(buf[i].pid) for i in range(n.value)]
        if not all(os.path.exists(f"/proc/{p}") for p in got):
            break  # implausible pid: layout mismatch
        pids |= set(got)
    else:
        return sorted(pids)
    r = subprocess.run(["nvidia-smi", f"--id={uuid()}", "--query-compute-apps=pid",
                        "--format=csv,noheader"], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return sorted(int(x) for x in r.stdout.split() if x.strip().isdigit())


def set_baseline() -> list[int]:
    """No frozen baseline on the GB300: allowed names only (prereg 4.4)."""
    return []


def foreign_gpu_pids() -> set[int]:
    p = gpu_pids()
    if p is None:
        return {-1}  # failed query: fail closed
    return {x for x in p if x != os.getpid() and _comm(x) not in ALLOWED}


def gpu_state() -> dict:
    lib, h = _nv()
    c, t, ps = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_int()
    m = _Mem()
    lib.nvmlDeviceGetClockInfo(h, 1, ctypes.byref(c))
    lib.nvmlDeviceGetTemperature(h, 0, ctypes.byref(t))
    lib.nvmlDeviceGetPerformanceState(h, ctypes.byref(ps))
    lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(m))
    return {"power_w": board_power_w(), "sm_mhz": c.value, "temp_c": t.value,
            "pstate": ps.value, "mem_used_mb": m.used / 2**20}


def _stat_busy() -> int:
    f = open("/proc/stat").readline().split()[1:]
    v = [int(x) for x in f]
    return sum(v) - v[3] - v[4]  # minus idle and iowait


def _own_ticks() -> int:
    s = open("/proc/self/stat").read()
    f = s[s.rindex(")") + 2:].split()
    return int(f[11]) + int(f[12]) + int(f[13]) + int(f[14])


def python_ticks() -> dict[int, tuple[str, int]]:
    """CPU ticks of every process (name 'python_ticks' kept for measure.py)."""
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            s = open(f"/proc/{d}/stat").read()
            name = s[s.index("(") + 1:s.rindex(")")]
            f = s[s.rindex(")") + 2:].split()
            out[int(d)] = (name, int(f[11]) + int(f[12]))
        except (OSError, ValueError, IndexError):
            continue
    _TICKS.append({"t": time.time(), "stat": _stat_busy(), "own": _own_ticks()})
    del _TICKS[:-2]
    return out


def busy_between(a: dict, b: dict, dt: float, threshold: float = R.CPU_PROC_MAX_PCT,
                 exclude_pid: int | None = None) -> list[tuple[float, int, str]]:
    return R.cpu_busy(a, b, dt, HZ, {exclude_pid or os.getpid()}, threshold)


def busy_python(threshold: float = R.CPU_PROC_MAX_PCT, exclude_pid: int | None = None,
                interval: float = 1.0) -> list[tuple[float, int, str]]:
    a = python_ticks()
    time.sleep(interval)
    return busy_between(a, python_ticks(), interval, threshold, exclude_pid)


def host_cores_last() -> float:
    if len(_TICKS) < 2:
        return float("inf")
    a, b = _TICKS
    return R.host_foreign_cores(a["stat"], b["stat"], a["own"], b["own"], HZ, b["t"] - a["t"])


def load1() -> float:
    return float(open("/proc/loadavg").read().split()[0])


def vm_cpu_percent() -> float:
    return 0.0


def n10c_active(*_a, **_k) -> str:
    """Other studies of the slot (pause patterns), in place of N10-C."""
    me = os.getpid()
    for pat in PAUSE_PATTERNS:
        out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout.split()
        out = [p for p in out if int(p) != me]
        if out:
            return f"{pat} pids {out}"
    return ""


def pause_reason() -> str:
    r = n10c_active()
    if r:
        return f"slot job: {r}"
    fp = foreign_gpu_pids()
    if fp:
        return f"foreign GPU pids {sorted(fp)}"
    w = board_power_w()
    if w < 0 or w >= rest_max_w():
        return f"board power {w:.1f} W"
    if load1() > R.LOAD1_MAX:
        return f"load average {load1():.2f}"
    python_ticks()
    time.sleep(1.0)
    python_ticks()
    if host_cores_last() > R.CPU_HOST_MAX_CORES:
        return f"host CPU {host_cores_last():.2f} cores"
    return ""


def other_gpus() -> list[dict]:
    lib, _ = _nv()
    n = ctypes.c_uint()
    lib.nvmlDeviceGetCount_v2(ctypes.byref(n))
    out = []
    for i in range(n.value):
        h = ctypes.c_void_p()
        if lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) != NVML_SUCCESS:
            continue
        buf = ctypes.create_string_buffer(96)
        lib.nvmlDeviceGetUUID(h, buf, 96)
        if buf.value.decode() == uuid():
            continue
        p, c = ctypes.c_uint(), ctypes.c_uint()
        rp = lib.nvmlDeviceGetPowerUsage(h, ctypes.byref(p))
        rc = lib.nvmlDeviceGetEnforcedPowerLimit(h, ctypes.byref(c))
        out.append({"uuid": buf.value.decode(), "power_w": p.value / 1000 if rp == 0 else -1,
                    "cap_w": c.value / 1000 if rc == 0 else -1})
    return out


def window_extra() -> dict:
    """GB300 fields added to every window record (measure.window hook)."""
    global _FAULTS_SEEN
    invalid = FAULTS > _FAULTS_SEEN
    _FAULTS_SEEN = FAULTS
    cap_changed = False
    if len(_SNAPS) == 2:
        a, b = _SNAPS
        if cfg().get("instrument", "counter") == "counter" and b["t"] - a["t"] >= 1.0 and b["e"] <= a["e"]:
            _fault("counter stalled over a window")
            invalid = True
            _FAULTS_SEEN = FAULTS
        cap_changed = a["cap"] != b["cap"]
    if FAULTS >= R.MAX_INSTRUMENT_FAULTS:
        raise SystemExit(5)  # instrument failure: run ends (prereg 4.1)
    return {"rest_rule": "gb300", "rest_max_w": rest_max_w(), "load1": load1(),
            "load_max": R.LOAD1_MAX, "host_foreign_cores": host_cores_last(),
            "host_cores_max": R.CPU_HOST_MAX_CORES, "instrument_invalid": invalid,
            "cap_changed": cap_changed, "cap_w": _SNAPS[-1]["cap"] if _SNAPS else -1,
            "instrument": cfg().get("instrument", "counter"), "other_gpus": other_gpus()}
