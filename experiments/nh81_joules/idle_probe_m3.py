"""Read-only idle probe on macM3 (M3 Pro), no model loaded.

GPU-rail energy (IOReport) over the probe, GPU utilisation (ioreg) every
0.5 s, 1-min load average, and mean CPU of heavy-pattern processes
(python, ffmpeg, bouba) over the probe.

Usage: python idle_probe_m3.py --out idle_probe.json [--seconds 120]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import power_m3 as p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=120.0)
    a = ap.parse_args()
    m = p.EnergyMeter()
    tk0 = p.python_ticks()
    s0 = m.snapshot()
    util, load, per10 = [], [], []
    last = s0
    while time.time() - s0[0] < a.seconds:
        util.append(p.gpu_utilization())
        load.append(os.getloadavg()[0])
        time.sleep(0.5)
        if len(util) % 20 == 0:  # 10-s GPU-rail power
            cur = m.snapshot()
            per10.append(p.rail_energy(m.delta_joules(last, cur))["gpu_j"] / (cur[0] - last[0]))
            last = cur
    s1 = m.snapshot()
    rails = p.rail_energy(m.delta_joules(s0, s1))
    dur = s1[0] - s0[0]
    busy = p.busy_between(tk0, p.python_ticks(), dur)
    out = {"t0": s0[0], "t1": s1[0], "dur_s": dur, "gpu_j": rails["gpu_j"],
           "gpu_w_mean": rails["gpu_j"] / dur, "gpu_w_10s": per10,
           "gpu_util_mean": float(np.mean(util)), "gpu_util_max": int(max(util)),
           "load1_mean": float(np.mean(load)), "load1_max": float(max(load)),
           "heavy_over_20pct": [(round(x[0], 1), x[1], x[2]) for x in busy]}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
