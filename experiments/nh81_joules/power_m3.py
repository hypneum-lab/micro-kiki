"""IOReport instrument and host guards for the macM3 (M3 Pro) replication.

Selected in ``measure.py`` with ``NH81_HOST=m3pro`` (prereg
``docs/prereg/2026-10-30-nh81-m3pro-replication.md``). Energy comes from
the same IOReport "Energy Model" reader as macM1 (``power.IOReportEnergy``,
GPU rail). The guards differ from macM1 (prereg section 3):

* other heavy processes (python, ffmpeg, bouba-sens) are judged on their
  mean CPU over a window, from cumulative CPU time (``ps -o time``);
* GPU-rail power over a window must stay under ``REST_MAX_W``;
* the 1-min load average must stay under ``LOAD_MAX``.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import time

from power import IOReportEnergy, gpu_utilization, rail_energy, vm_cpu_percent  # noqa: F401

REST_MAX_W = 0.25   # GPU-rail rest power (W), prereg 2026-10-30-nh81-m3pro section 3
LOAD_MAX = 6.0      # 1-min load average, same section
HEAVY = re.compile(r"python|ffmpeg|bouba", re.I)
N10C_DIR = os.path.expanduser(os.environ.get("NH81_N10C_DIR", "~/scratch/n10c"))
REST_RULE = "m3pro"

_METER: IOReportEnergy | None = None


def _meter() -> IOReportEnergy:
    global _METER
    if _METER is None:
        _METER = IOReportEnergy()
    return _METER


class EnergyMeter(IOReportEnergy):
    pass


def _cputime(s: str) -> float:
    """``ps -o time`` value ([[dd-]hh:]mm:ss.cc) in seconds."""
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [float(x.replace(",", ".")) for x in s.split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return days * 86400 + sec


def python_ticks() -> dict[int, tuple[str, float]]:
    """Cumulative CPU seconds of heavy-pattern processes (start/end of a window)."""
    out = subprocess.run(["ps", "-Ao", "pid=,time=,comm="], capture_output=True, text=True).stdout
    rows = {}
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and HEAVY.search(parts[2]):
            try:
                rows[int(parts[0])] = (parts[2], _cputime(parts[1]))
            except ValueError:
                pass
    return rows


def busy_between(a: dict, b: dict, dt: float, threshold: float = 20.0,
                 exclude_pid: int | None = None) -> list[tuple[float, int, str]]:
    me = exclude_pid or os.getpid()
    rows = []
    for pid, (cmd, t) in b.items():
        if pid == me or pid not in a or dt <= 0:
            continue
        pct = 100.0 * (t - a[pid][1]) / dt
        if pct > threshold:
            rows.append((pct, pid, cmd))
    return rows


def busy_python(threshold: float = 20.0, exclude_pid: int | None = None,
                interval: float = 2.0) -> list[tuple[float, int, str]]:
    a = python_ticks()
    time.sleep(interval)
    return busy_between(a, python_ticks(), interval, threshold, exclude_pid)


def load1() -> float:
    return os.getloadavg()[0]


def gpu_power_w(interval: float = 1.0) -> float:
    """GPU-rail power (W) over ``interval`` s; -1 on a failed read."""
    try:
        m = _meter()
        a = m.snapshot()
        time.sleep(interval)
        b = m.snapshot()
        return rail_energy(m.delta_joules(a, b))["gpu_j"] / (b[0] - a[0])
    except Exception:
        return -1.0


board_power_w = gpu_power_w  # name used by measure.start_guard_power


def set_baseline() -> list[int]:
    """No GPU process list on macOS; kept for the shared guard code."""
    return []


def n10c_active(campaign_dir: str = N10C_DIR, minutes: float = 10.0) -> str:
    pg = subprocess.run(["pgrep", "-f", "run_campaign"], capture_output=True, text=True).stdout.split()
    pg = [p for p in pg if int(p) != os.getpid()]
    if pg:
        return f"run_campaign pids {pg}"
    now = time.time()
    for f in glob.glob(campaign_dir + "/runs/**", recursive=True):
        try:
            if os.path.isfile(f) and now - os.path.getmtime(f) < minutes * 60:
                return f"recent N10-C file {f}"
        except OSError:
            pass
    return ""


def pause_reason() -> str:
    r = n10c_active()
    if r:
        return f"N10-C: {r}"
    w = gpu_power_w()
    if w < 0 or w >= REST_MAX_W:
        return f"GPU rail {w:.3f} W"
    ld = load1()
    if ld >= LOAD_MAX:
        return f"load average {ld:.2f}"
    return ""


def window_extra() -> dict:
    """Fields added to every window record (rest rule of prereg section 3)."""
    return {"rest_rule": REST_RULE, "rest_max_w": REST_MAX_W, "load_max": LOAD_MAX,
            "load1": load1()}
