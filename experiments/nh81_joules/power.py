"""Energy instruments and host guards for NH-81 (Apple silicon).

Two instruments read the same hardware energy counters:

* ``IOReportEnergy`` -- sudo-less reader of the IOReport "Energy Model"
  group (the source ``powermetrics`` itself uses). Counters are cumulative
  energies, so the energy of a window is the delta of two snapshots; no
  power integration is needed.
* ``parse_powermetrics_plist`` -- parser for a ``powermetrics -f plist``
  log started by the PI under sudo (see the preregistration, section 5).
  Energy of a window is the sum of per-sample GPU energy whose sample
  interval falls inside the window.

Host guards (``gpu_utilization``, ``busy_python``, ``n10c_active``) implement
the stop/wait rules of the preregistration.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import plistlib
import re
import subprocess
import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# IOReport (sudo-less)
# ---------------------------------------------------------------------------

_UNIT_TO_J = {"mJ": 1e-3, "uJ": 1e-6, "nJ": 1e-9, "J": 1.0}


class IOReportEnergy:
    """Cumulative energy counters of the IOReport "Energy Model" group."""

    def __init__(self) -> None:
        cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
        io = ctypes.cdll.LoadLibrary("/usr/lib/libIOReport.dylib")
        vp, i64 = ctypes.c_void_p, ctypes.c_int64
        cf.CFStringCreateWithCString.restype = vp
        cf.CFStringCreateWithCString.argtypes = [vp, ctypes.c_char_p, ctypes.c_uint32]
        cf.CFStringGetCString.restype = ctypes.c_bool
        cf.CFStringGetCString.argtypes = [vp, ctypes.c_char_p, i64, ctypes.c_uint32]
        cf.CFDictionaryGetValue.restype = vp
        cf.CFDictionaryGetValue.argtypes = [vp, vp]
        cf.CFDictionaryGetCount.restype = i64
        cf.CFDictionaryGetCount.argtypes = [vp]
        cf.CFDictionaryCreateMutableCopy.restype = vp
        cf.CFDictionaryCreateMutableCopy.argtypes = [vp, i64, vp]
        cf.CFArrayGetCount.restype = i64
        cf.CFArrayGetCount.argtypes = [vp]
        cf.CFArrayGetValueAtIndex.restype = vp
        cf.CFArrayGetValueAtIndex.argtypes = [vp, i64]
        cf.CFRelease.argtypes = [vp]
        io.IOReportCopyChannelsInGroup.restype = vp
        io.IOReportCopyChannelsInGroup.argtypes = [vp, vp, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        io.IOReportCreateSubscription.restype = vp
        io.IOReportCreateSubscription.argtypes = [vp, vp, ctypes.POINTER(vp), ctypes.c_uint64, vp]
        io.IOReportCreateSamples.restype = vp
        io.IOReportCreateSamples.argtypes = [vp, vp, vp]
        io.IOReportCreateSamplesDelta.restype = vp
        io.IOReportCreateSamplesDelta.argtypes = [vp, vp, vp]
        io.IOReportSimpleGetIntegerValue.restype = i64
        io.IOReportSimpleGetIntegerValue.argtypes = [vp, ctypes.c_int32]
        for fn in ("IOReportChannelGetChannelName", "IOReportChannelGetUnitLabel",
                   "IOReportChannelGetGroup"):
            getattr(io, fn).restype = vp
            getattr(io, fn).argtypes = [vp]
        self.cf, self.io = cf, io
        chans = io.IOReportCopyChannelsInGroup(self._s("Energy Model"), None, 0, 0, 0)
        if not chans:
            raise RuntimeError("IOReport: no 'Energy Model' channels")
        self._chans = cf.CFDictionaryCreateMutableCopy(None, cf.CFDictionaryGetCount(chans), chans)
        self._subbed = vp()
        self._sub = io.IOReportCreateSubscription(None, self._chans, ctypes.byref(self._subbed), 0, None)
        if not self._sub:
            raise RuntimeError("IOReport: subscription failed")
        self._key = self._s("IOReportChannels")
        self.snapshot()  # the first sample after subscribing can be stale
        time.sleep(0.2)

    def _s(self, text: str):
        return self.cf.CFStringCreateWithCString(None, text.encode(), 0x08000100)

    def _str(self, ref) -> str:
        if not ref:
            return ""
        buf = ctypes.create_string_buffer(256)
        self.cf.CFStringGetCString(ref, buf, 256, 0x08000100)
        return buf.value.decode(errors="replace")

    def _absolute(self, sample) -> dict[str, float]:
        arr = self.cf.CFDictionaryGetValue(sample, self._key)
        out: dict[str, float] = {}
        for i in range(self.cf.CFArrayGetCount(arr)):
            item = self.cf.CFArrayGetValueAtIndex(arr, i)
            name = self._str(self.io.IOReportChannelGetChannelName(item))
            unit = self._str(self.io.IOReportChannelGetUnitLabel(item)).strip()
            val = self.io.IOReportSimpleGetIntegerValue(item, 0)
            out[name] = out.get(name, 0.0) + val * _UNIT_TO_J.get(unit, float("nan"))
        return out

    def snapshot(self) -> tuple[float, dict[str, float]]:
        """(unix time, cumulative energy per channel in J)."""
        s = self.io.IOReportCreateSamples(self._sub, self._chans, None)
        t = time.time()
        vals = self._absolute(s)
        self.cf.CFRelease(s)
        return t, vals

    @staticmethod
    def delta_joules(a, b) -> dict[str, float]:
        """Energy per channel (J) between snapshots ``a`` and ``b``.

        The counters are cumulative, so this is a plain difference (the
        IOReportCreateSamplesDelta path returned zeros on macOS 27).
        """
        return {k: b[1][k] - a[1].get(k, 0.0) for k in b[1]}


def rail_energy(channels: dict[str, float]) -> dict[str, float]:
    """Aggregate IOReport channels into GPU / ANE / CPU / DRAM rails (J).

    GPU = "GPU Energy" (nJ resolution) or, if absent, "GPU0" (mJ); the
    "GPU SRAM" channel is kept separate. CPU = "CPU Energy" (all clusters).
    Each rail is read from one channel only, so nothing is double counted.
    """
    def pick(*names: str) -> float:
        for n in names:
            if n in channels:
                return channels[n]
        return 0.0

    gpu = pick("GPU Energy", "GPU0", "GPU")
    ane = pick("ANE0", "ANE")
    cpu = pick("CPU Energy")
    dram = pick("DRAM0", "DRAM")
    return {"gpu_j": gpu, "ane_j": ane, "cpu_j": cpu, "dram_j": dram,
            "gpu_sram_j": pick("GPU SRAM0", "GPU SRAM")}


# ---------------------------------------------------------------------------
# powermetrics (PI-run, sudo)
# ---------------------------------------------------------------------------

POWERMETRICS_CMD = (
    "sudo powermetrics --samplers gpu_power,cpu_power,ane_power -i 100 "
    "-f plist -o ~/scratch/nh81/pm.plist"
)


@dataclass
class PMSample:
    t_end: float      # unix time at end of sample
    dt: float         # sample interval (s)
    gpu_j: float
    ane_j: float
    cpu_j: float


def parse_powermetrics_plist(blob: bytes) -> list[PMSample]:
    """Parse a NUL-separated ``powermetrics -f plist`` stream."""
    out: list[PMSample] = []
    for chunk in blob.split(b"\x00"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            d = plistlib.loads(chunk)
        except Exception:  # truncated last sample
            continue
        dt = d.get("elapsed_ns", 0) / 1e9
        ts = d.get("timestamp")
        t_end = ts.timestamp() if hasattr(ts, "timestamp") else float("nan")
        p = d.get("processor", {})
        gpu = d.get("gpu", {})

        def energy(key_e: str, key_p: str, *srcs) -> float:
            for s in srcs:
                if key_e in s:
                    return float(s[key_e]) * 1e-3          # mJ
            for s in srcs:
                if key_p in s:
                    return float(s[key_p]) * 1e-3 * dt     # mW * s
            return 0.0

        out.append(PMSample(t_end, dt,
                            energy("gpu_energy", "gpu_power", gpu, p),
                            energy("ane_energy", "ane_power", p),
                            energy("cpu_energy", "cpu_power", p)))
    return out


def window_energy(samples: list[PMSample], t0: float, t1: float) -> dict[str, float]:
    """Energy (J) inside [t0, t1], prorating samples that straddle edges."""
    acc = {"gpu_j": 0.0, "ane_j": 0.0, "cpu_j": 0.0, "covered_s": 0.0}
    for s in samples:
        a, b = s.t_end - s.dt, s.t_end
        ov = max(0.0, min(b, t1) - max(a, t0))
        if ov <= 0 or s.dt <= 0:
            continue
        f = ov / s.dt
        acc["gpu_j"] += f * s.gpu_j
        acc["ane_j"] += f * s.ane_j
        acc["cpu_j"] += f * s.cpu_j
        acc["covered_s"] += ov
    return acc


# ---------------------------------------------------------------------------
# Host guards
# ---------------------------------------------------------------------------

def gpu_utilization() -> int:
    """GPU "Device Utilization %" from ioreg (0-100)."""
    txt = subprocess.run(["ioreg", "-r", "-c", "IOAccelerator", "-d", "1"],
                         capture_output=True, text=True).stdout
    m = re.search(r'"Device Utilization %"=(\d+)', txt)
    return int(m.group(1)) if m else -1


def process_table() -> list[tuple[float, int, str]]:
    """(pcpu, pid, command) for all processes."""
    txt = subprocess.run(["ps", "-Ao", "pcpu=,pid=,comm="], capture_output=True, text=True).stdout
    rows = []
    for line in txt.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3:
            try:
                rows.append((float(parts[0].replace(",", ".")), int(parts[1]), parts[2]))
            except ValueError:
                pass
    return rows


def busy_python(threshold: float = 20.0, exclude_pid: int | None = None) -> list[tuple[float, int, str]]:
    """Other Python processes above ``threshold`` %CPU."""
    me = exclude_pid or os.getpid()
    return [r for r in process_table()
            if "python" in r[2].lower() and r[0] > threshold and r[1] != me]


def vm_cpu_percent() -> float:
    """%CPU of OrbStack processes (the production VM `srv`); descriptive only."""
    return sum(r[0] for r in process_table() if "orbstack" in r[2].lower())


def n10c_active(campaign_dir: str = "~/scratch/n10c-run/campaign", minutes: float = 10.0) -> str:
    """Non-empty reason string if an N10-C campaign runs or ran recently."""
    pg = subprocess.run(["pgrep", "-f", "run_campaign"], capture_output=True, text=True).stdout.split()
    pg = [p for p in pg if int(p) != os.getpid()]
    if pg:
        return f"run_campaign pids {pg}"
    now = time.time()
    for f in glob.glob(os.path.expanduser(campaign_dir) + "/**", recursive=True):
        try:
            if os.path.isfile(f) and now - os.path.getmtime(f) < minutes * 60:
                return f"recent campaign file {f}"
        except OSError:
            pass
    return ""
