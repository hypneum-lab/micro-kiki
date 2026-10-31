"""Pure rules of the NH-81 GB300 descriptive run (no GPU, no MLX).

Prereg ``docs/prereg/nh81-gb300/2026-10-31-nh81-gb300-descriptive.md``
(PI decisions 2026-10-31). Imported by the host module, the probe, the
wrapper, the summary writer and the tests.
"""

from __future__ import annotations

import math
import re

K_REST = 1.25                # T = K_REST x P_rest (PI 2026-10-31)
STAB_S = 600.0               # thermal stabilisation before P_rest (10 min)
CPU_PROC_MAX_PCT = 20.0      # any foreign process, % of one core over a window
CPU_HOST_MAX_CORES = 1.0     # host CPU outside our process, core-equivalents
LOAD1_MAX = 4.0              # 1-min load average
COUNTER_TOL = 0.10           # counter vs power-usage agreement
GUARD_BUDGET_S = 3600.0      # start-guard window
PAUSE_BUDGET_S = 3600.0      # cumulative pause budget
DEADLINE_S = 2 * 3600.0      # global deadline from the first probe
MAX_INSTRUMENT_FAULTS = 3
MAX_SAMPLE_GAP_S = 1.0       # power-integration fallback


def threshold(p_rest_w: float, k: float = K_REST) -> float:
    if not (p_rest_w > 0 and math.isfinite(p_rest_w)):
        raise ValueError(f"invalid P_rest {p_rest_w}")
    return k * p_rest_w


def max_sliding_mean(t: list[float], e: list[float], width: float = 20.0) -> float:
    """Largest mean power over any ``width``-s window, from cumulative
    energy readings ``e`` (J) at times ``t`` (s), windows starting at each
    reading and ending at the first reading >= start + width."""
    best, j = -1.0, 0
    for i in range(len(t)):
        while j < len(t) and t[j] - t[i] < width:
            j += 1
        if j == len(t):
            break
        best = max(best, (e[j] - e[i]) / (t[j] - t[i]))
    if best < 0:
        raise ValueError("series shorter than the window")
    return best


def counter_consistent(counter_w: float, sampled_w: float, tol: float = COUNTER_TOL) -> bool:
    return sampled_w > 0 and abs(counter_w - sampled_w) <= tol * sampled_w


def parse_bus_id(s: str) -> tuple[int, int, int, int]:
    """'00000009:01:00.0' (NVML) or '0009:01:00.0' (CUDA) -> ints."""
    m = re.fullmatch(r"\s*([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+)\.([0-7])\s*", s)
    if not m:
        raise ValueError(f"bad PCI bus id {s!r}")
    return tuple(int(x, 16) for x in m.groups())  # type: ignore[return-value]


def uuid_from_bytes(b: bytes) -> str:
    """16 CUDA UUID bytes -> NVML form 'GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'."""
    h = b.hex()
    return f"GPU-{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def cpu_busy(a: dict, b: dict, dt: float, hz: int, exclude: set[int],
             threshold_pct: float = CPU_PROC_MAX_PCT) -> list[tuple[float, int, str]]:
    """Foreign processes above ``threshold_pct`` between two tick maps
    {pid: (name, ticks)}; a process absent from ``a`` counts from zero."""
    rows = []
    if dt <= 0:
        return rows
    for pid, (name, ticks) in b.items():
        if pid in exclude:
            continue
        pct = 100.0 * (ticks - (a[pid][1] if pid in a else 0)) / hz / dt
        if pct > threshold_pct:
            rows.append((pct, pid, name))
    return rows


def host_foreign_cores(stat_a: int, stat_b: int, own_a: int, own_b: int,
                       hz: int, dt: float) -> float:
    """Busy host CPU (jiffies from /proc/stat) minus our process, in cores."""
    if dt <= 0:
        return float("inf")
    return max(0.0, ((stat_b - stat_a) - (own_b - own_a)) / hz / dt)


def window_ok_gb300(r: dict) -> bool:
    return (not r.get("foreign_gpu_pids") and not r.get("instrument_invalid")
            and not r.get("cap_changed") and not r.get("watch_errors"))


def rest_ok_gb300(r: dict) -> bool:
    if r.get("rest_rule") != "gb300":
        raise ValueError("not a GB300 record")
    w = r["gpu_j"] / r["dur_s"] if r.get("dur_s", 0) > 0 else -1.0
    return (0 <= w <= r["rest_max_w"] and not r["busy_python"]
            and 0 <= r["load1"] <= r["load_max"]
            and r["host_foreign_cores"] <= r["host_cores_max"]
            and window_ok_gb300(r))
