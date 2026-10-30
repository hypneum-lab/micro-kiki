"""Recompute held-out losses of the measured checkpoints on the current device.

Replication prereg 2026-10-28, section 3: same evaluation as training
(``train.held_out_loss``: 512 fixed windows, seeds 1 / 2), then the parent
iso-quality rule and the 0.005-nat reproduction check against macM1.

Usage:
    python isoquality.py --dense ann1.npz --spiking spkt.npz --data split/ \
        --macm1 results/nh81/selection.json --out isoquality.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import mlx.core as mx
import numpy as np

from analyze import ISO_MARGIN
from model_mlx import LM
from train import EVAL_SEED, held_out_loss

REPRO_TOL = 0.005  # nats


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True)
    ap.add_argument("--spiking", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--macm1", required=True, help="results/nh81/selection.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mx.set_memory_limit(6 * 2**30)
    ref = json.load(open(a.macm1))
    held = {k: np.fromfile(os.path.join(a.data, f"{k}.i32"), dtype=np.int32) for k in ("val", "test")}
    out = {"device": str(mx.default_device()), "mlx": mx.__version__,
           "tf32": os.environ.get("MLX_ENABLE_TF32", "default"), "models": {}}
    for key, path, arm, rkey in (("dense", a.dense, "ann1", "dense_star"),
                                 ("spiking", a.spiking, "spkt", "spiking")):
        m = LM(arm=arm)
        m.load_weights(path)
        m.eval()
        rec = {"path": path, "sha256": sha256(path)}
        for split in ("val", "test"):
            loss, fire = held_out_loss(m, held[split], m.cfg["SEQ"], EVAL_SEED[split])
            rec[f"{split}_nats"], rec[f"{split}_fire"] = loss, fire
            rec[f"{split}_macm1"] = ref[f"{rkey}_{split}"]
            rec[f"{split}_dev"] = loss - ref[f"{rkey}_{split}"]
        out["models"][key] = rec
    d, s = out["models"]["dense"], out["models"]["spiking"]
    gap = s["test_nats"] - d["test_nats"]
    out["gap_test_nats"] = gap
    out["iso_quality"] = abs(gap) <= ISO_MARGIN
    out["iso_direction"] = "iso" if abs(gap) <= ISO_MARGIN else (
        "spiking_worse" if gap > 0 else "spiking_better")
    out["repro_max_dev"] = max(abs(r[f"{k}_dev"]) for r in (d, s) for k in ("val", "test"))
    out["repro_ok"] = out["repro_max_dev"] <= REPRO_TOL
    out["spiking_test_fire"] = s["test_fire"]
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0 if out["repro_ok"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
