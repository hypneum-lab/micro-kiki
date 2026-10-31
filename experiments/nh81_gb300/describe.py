"""Descriptive GB300 summary (prereg section 5, PI decisions 2026-10-31).

Pairs are built by the frozen ``analyze.build_pairs`` (from
``NH81_CODE_DIR``) with the GB300 rest/window rules. No ``decision`` or
``catalogue`` field. Pairs whose net energy is <= 0 for either model are
excluded from the net R-hat and counted (PI rule); the gross ratio uses
every valid pair.

Usage: python describe.py --measure measure.jsonl --out summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.environ.get("NH81_CODE_DIR", "")]

import gb300_rules as R  # noqa: E402
import analyze  # noqa: E402

analyze.rest_ok = R.rest_ok_gb300
analyze.window_ok = R.window_ok_gb300

BLOCKS = [("W1", "fused"), ("W2", "fused"), ("W1", "event"), ("W2", "event")]


def describe_block(pairs: list[dict]) -> dict:
    valid = [p for p in pairs if p["valid"]]
    pos = [p for p in valid if p["dense_j_per_tok"] > 0 and p["spiking_j_per_tok"] > 0]
    res = {"pairs_total": len(pairs), "pairs_valid": len(valid),
           "pairs_nonpositive_net_excluded": len(valid) - len(pos),
           "fewer_than_10_valid": len(valid) < analyze.MIN_PAIRS}
    if valid:
        g = np.array([p["ratio_gross"] for p in valid])
        res["R_gross_hat"] = analyze.geomean(g)
        res["rest_gpu_w_mean"] = float(np.mean([p["rest_gpu_w"] for p in valid]))
        for k in ("dense", "spiking"):
            res[f"{k}_tok_s_mean"] = float(np.mean([p[f"{k}_tok_s"] for p in valid]))
            res[f"{k}_gross_j_per_tok_mean"] = float(np.mean([p[f"{k}_gross_j_per_tok"] for p in valid]))
    if pos:
        r = np.array([p["dense_j_per_tok"] / p["spiking_j_per_tok"] for p in pos])
        d = np.array([p["spiking_j_per_tok"] - p["dense_j_per_tok"] for p in pos])
        res.update({"R_hat": analyze.geomean(r), "R_ci95": analyze.bootstrap(r, analyze.geomean),
                    "diff_j_per_tok_mean": float(d.mean()),
                    "diff_ci95": analyze.bootstrap(d, np.mean),
                    "dense_j_per_tok_mean": float(np.mean([p["dense_j_per_tok"] for p in pos])),
                    "spiking_j_per_tok_mean": float(np.mean([p["spiking_j_per_tok"] for p in pos]))})
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = analyze.read_jsonl(a.measure)
    out = {"status": "descriptive (no verdict, no cross-host inference)", "blocks": {}}
    for w, path in BLOCKS:
        pairs = analyze.build_pairs(recs, w, path)
        out["blocks"][f"{w}_{path}"] = {"stats": describe_block(pairs), "pairs": pairs}
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: v["stats"] for k, v in out["blocks"].items()}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
