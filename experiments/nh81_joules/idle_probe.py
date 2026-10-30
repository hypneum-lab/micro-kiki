"""Read-only 60 s idle probe on KXKM (replication prereg section 4).

No model is loaded. Samples NVML utilisation, board power, compute
processes and the cumulative energy counter every 0.5 s.

Usage: python idle_probe.py --out idle_probe.json [--seconds 60]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import power_nvml as p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    a = ap.parse_args()
    e0, t0 = p.total_energy_j(), time.time()
    util, pw, pids = [], [], set()
    while time.time() - t0 < a.seconds:
        util.append(p.gpu_utilization())
        st = p.gpu_state()
        pw.append(st["power_w"])
        pids |= set(p.gpu_compute_pids())
        time.sleep(0.5)
    e1, t1 = p.total_energy_j(), time.time()
    out = {"t0": t0, "t1": t1, "energy_j": e1 - e0, "mean_power_w": (e1 - e0) / (t1 - t0),
           "sampled_power_w_mean": float(np.mean(pw)), "gpu_util_mean": float(np.mean(util)),
           "gpu_util_max": int(max(util)), "compute_pids": sorted(pids),
           "busy_python": [r[2] for r in p.busy_python()], "n10c": p.n10c_active(),
           "state_end": p.gpu_state()}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
