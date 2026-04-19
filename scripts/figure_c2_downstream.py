#!/usr/bin/env python3
"""Generate c2-downstream-figure.pdf from c2-downstream.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("results/c2-downstream.json"))
    p.add_argument("--output", type=Path, default=Path("docs/paper-a/c2-downstream-figure.pdf"))
    args = p.parse_args()

    meta = json.loads(args.input.read_text())
    data = meta["results"] if "results" in meta else meta  # back-compat
    order = ["random", "vqc", "oracle"]
    colors = ["#bbbbbb", "#6699ff", "#66bb77"]
    means = [data[n]["mean_score"] for n in order]
    corr_means = [data[n]["mean_score_when_routed_correct"] for n in order]
    wrong_means = [data[n]["mean_score_when_routed_wrong"] for n in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(order))
    ax1.bar(x, means, color=colors, edgecolor="black", linewidth=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(order)
    ax1.set_ylabel("Mean judge score (0-5)")
    ax1.set_title("Downstream quality by router")
    ax1.set_ylim(0, 5)
    for i, m in enumerate(means):
        ax1.text(i, m + 0.05, f"{m:.2f}", ha="center", fontsize=9)

    width = 0.35
    ax2.bar(x - width / 2, corr_means, width, label="correct route", color="#66bb77")
    ax2.bar(x + width / 2, wrong_means, width, label="wrong route", color="#cc5555")
    ax2.set_xticks(x)
    ax2.set_xticklabels(order)
    ax2.set_ylabel("Mean judge score (0-5)")
    ax2.set_title("Conditional on routing correctness")
    ax2.set_ylim(0, 5)
    ax2.legend()

    self_bias = meta.get("config", {}).get("self_judging", False)
    if self_bias:
        fig.suptitle("C2 downstream eval (SELF-JUDGING disclosed — same model gen + judge)",
                     fontsize=9, y=1.02)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
