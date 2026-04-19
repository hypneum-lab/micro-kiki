#!/usr/bin/env python3
"""Numerical verification of the Holevo + Fano bound on 10-class routing.

Loads the same MiniLM embeddings as C1, estimates I(X; Y) at several projection
dimensions, computes the acc_upper_bound, and compares to the empirically
measured VQC + classical baselines from C1 results.

Writes results/c5-info-bound.json and docs/paper-a/c5-bound-figure.pdf.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.routing.info_bound import (
    acc_upper_bound,
    estimate_mi_bits,
    holevo_capacity_bits,
)

logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings-npz", type=Path, default=Path("results/.c1-cache.npz"))
    p.add_argument("--c1-results", type=Path, default=Path("results/c1-classical-vs-vqc.json"))
    p.add_argument("--output", type=Path, default=Path("results/c5-info-bound.json"))
    p.add_argument("--figure", type=Path, default=Path("docs/paper-a/c5-bound-figure.pdf"))
    p.add_argument("--pca-dims", default="2,4,8,16,32,64,128,384")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.embeddings_npz.exists():
        logger.error("embeddings cache not found at %s (run C1 first to populate it)",
                     args.embeddings_npz)
        return 2
    cache = np.load(args.embeddings_npz)
    X, y = cache["embeddings"], cache["labels"]
    n_classes = int(y.max()) + 1
    logger.info("loaded X shape=%s, K=%d", X.shape, n_classes)

    from sklearn.decomposition import PCA

    pca_dims = [int(d) for d in args.pca_dims.split(",") if d.strip()]
    mi_per_dim: dict[int, float] = {}
    for d in pca_dims:
        if d >= X.shape[1]:
            Xp = X
        else:
            Xp = PCA(n_components=d, random_state=0).fit_transform(X)
        mi = estimate_mi_bits(Xp, y)
        mi_per_dim[d] = mi
        logger.info("  dim=%3d  MI ~ %.3f bits  (Holevo cap: %.1f)",
                    d, mi, holevo_capacity_bits(d))

    # Our architecture: 4-qubit VQC with learned 384->4 projection
    # The Holevo cap is 4 bits. The actual MI in the projected space is
    # bounded by MI_PCA at dim=4 as an upper proxy.
    mi_4d = mi_per_dim[4] if 4 in mi_per_dim else None
    bound_vqc = acc_upper_bound(n_qubits=4, n_classes=n_classes,
                                mi_estimate_bits=mi_4d) if mi_4d is not None else None

    # Empirical C1 measurements
    c1_acc = None
    if args.c1_results.exists():
        c1 = json.loads(args.c1_results.read_text())
        c1_acc = {k: v["accuracy_mean"] for k, v in c1["aggregated"].items()}
        logger.info("C1 empirical: torch_vqc=%.3f logreg_pca=%.3f mlp=%.3f logreg=%.3f",
                    c1_acc.get("torch_vqc", 0), c1_acc.get("logreg_pca", 0),
                    c1_acc.get("mlp", 0), c1_acc.get("logreg", 0))

    out = {
        "config": {"n_classes": n_classes, "n_samples": int(len(X))},
        "mi_per_pca_dim_bits": {str(k): v for k, v in mi_per_dim.items()},
        "holevo_cap_4qubit_bits": holevo_capacity_bits(4),
        "bound_vqc_4qubit": bound_vqc,
        "c1_empirical_acc": c1_acc,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", args.output)

    # Figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dims_sorted = sorted(pca_dims)
    mis = [mi_per_dim[d] for d in dims_sorted]
    bounds = [acc_upper_bound(n_qubits=4, n_classes=n_classes, mi_estimate_bits=m)
              for m in mis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(dims_sorted, mis, "o-", color="#3366cc", linewidth=1.5, markersize=6,
             label="estimated I(X_PCA; Y)")
    ax1.axhline(y=holevo_capacity_bits(4), color="red", linestyle="--",
                label="Holevo cap (4 qubits)")
    ax1.axhline(y=np.log2(n_classes), color="gray", linestyle=":",
                label=f"H(Y) = log2({n_classes})")
    ax1.set_xscale("log")
    ax1.set_xlabel("PCA dimension")
    ax1.set_ylabel("Mutual information (bits)")
    ax1.set_title("MI vs embedding dimension")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(dims_sorted, bounds, "s-", color="#cc3366", linewidth=1.5, markersize=6,
             label="Holevo+Fano upper bound (4 qubits)")
    if c1_acc:
        for name, acc in c1_acc.items():
            ax2.axhline(y=acc, linestyle="--", linewidth=0.8,
                        label=f"{name} empirical = {acc:.2f}")
    ax2.set_xscale("log")
    ax2.set_xlabel("PCA dimension of features fed to bound")
    ax2.set_ylabel("Accuracy upper bound")
    ax2.set_title("Bound vs empirical C1 accuracies")
    ax2.legend(loc="lower right", fontsize=7)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.figure, bbox_inches="tight")
    logger.info("wrote %s", args.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
