#!/usr/bin/env python3
"""Post-hoc diagnostic on results/c2-downstream.json.

Produces three analyses (per-domain gap, correctness-stratified, top-10
qualitative) plus two matplotlib PDFs and a machine-readable JSON summary.

Usage:
    uv run python scripts/c2_diagnostic.py \\
        --input results/c2-downstream.json \\
        --out-json results/c2-diagnostic.json \\
        --out-per-domain-pdf docs/paper-a/c2-diagnostic-per-domain.pdf \\
        --out-stratified-pdf docs/paper-a/c2-diagnostic-stratified.pdf \\
        --out-top10-md docs/paper-a/c2-diagnostic-top10.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_DEFAULT_DOMAINS = [
    "dsp", "electronics", "emc", "embedded", "freecad",
    "kicad-dsl", "platformio", "power", "spice", "stm32",
]


def analyze_per_domain(data: dict, domains: list[str]) -> dict:
    """Return per-domain mean scores and gaps for vqc, oracle, random routers."""
    out: dict = {}
    for d in domains:
        per_domain: dict[str, float] = {}
        for router in ("oracle", "vqc", "random"):
            records = [
                r for r in data["results"][router]["per_query"]
                if r["expected_domain"] == d
            ]
            if not records:
                per_domain[f"mean_{router}"] = 0.0
                continue
            per_domain[f"mean_{router}"] = sum(r["score"] for r in records) / len(records)
        per_domain["gap_oracle_vs_vqc"] = per_domain["mean_oracle"] - per_domain["mean_vqc"]
        per_domain["gap_vqc_vs_random"] = per_domain["mean_vqc"] - per_domain["mean_random"]
        per_domain["n"] = sum(
            1 for r in data["results"]["oracle"]["per_query"] if r["expected_domain"] == d
        )
        out[d] = per_domain
    return out


def analyze_stratified(data: dict) -> dict:
    """Split queries by VQC correctness, report per-router mean score in each bucket."""
    vqc_per_query = data["results"]["vqc"]["per_query"]
    oracle_per_query = data["results"]["oracle"]["per_query"]
    random_per_query = data["results"]["random"]["per_query"]

    correct_idx = [i for i, r in enumerate(vqc_per_query) if r["correct_route"]]
    wrong_idx = [i for i, r in enumerate(vqc_per_query) if not r["correct_route"]]

    def _bucket(indices: list[int]) -> dict:
        if not indices:
            return {
                "n": 0,
                "vqc_mean_score": 0.0,
                "oracle_mean_score": 0.0,
                "random_mean_score": 0.0,
            }
        return {
            "n": len(indices),
            "vqc_mean_score": sum(vqc_per_query[i]["score"] for i in indices) / len(indices),
            "oracle_mean_score": sum(oracle_per_query[i]["score"] for i in indices) / len(indices),
            "random_mean_score": sum(random_per_query[i]["score"] for i in indices) / len(indices),
        }

    return {
        "vqc_correct": _bucket(correct_idx),
        "vqc_wrong": _bucket(wrong_idx),
    }


def top_10_by_gap(data: dict, k: int = 10) -> list[dict]:
    """Return top-k queries by oracle_score - vqc_score, descending, ties stable."""
    oracle_pq = data["results"]["oracle"]["per_query"]
    vqc_pq = data["results"]["vqc"]["per_query"]
    random_pq = data["results"]["random"]["per_query"]

    rows = []
    for i, (o, v, r) in enumerate(zip(oracle_pq, vqc_pq, random_pq)):
        rows.append({
            "index": i,
            "question": o["question"],
            "expected_domain": o["expected_domain"],
            "oracle_score": o["score"],
            "oracle_answer": o["answer"],
            "vqc_routed_domain": v["routed_domain"],
            "vqc_score": v["score"],
            "vqc_answer": v["answer"],
            "random_routed_domain": r["routed_domain"],
            "random_score": r["score"],
            "random_answer": r["answer"],
            "gap": o["score"] - v["score"],
        })

    # Sort: primary key = gap desc, secondary = index asc (stability)
    rows.sort(key=lambda x: (-x["gap"], x["index"]))
    return rows[:k]


def _render_per_domain_pdf(per_domain: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    domains_sorted = sorted(per_domain.keys(),
                            key=lambda d: per_domain[d]["gap_oracle_vs_vqc"],
                            reverse=True)
    ov = [per_domain[d]["gap_oracle_vs_vqc"] for d in domains_sorted]
    vr = [per_domain[d]["gap_vqc_vs_random"] for d in domains_sorted]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(domains_sorted))
    width = 0.38
    ax.bar(x - width / 2, ov, width, label="oracle - vqc", color="#3366cc", edgecolor="black", linewidth=0.6)
    ax.bar(x + width / 2, vr, width, label="vqc - random", color="#cc3366", edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(domains_sorted, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score gap (rubric points)")
    ax.set_title("C2 diagnostic: per-domain score gaps")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _render_stratified_pdf(stratified: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    buckets = ["vqc_correct", "vqc_wrong"]
    routers = ["vqc", "oracle", "random"]
    values = np.array([
        [stratified[b][f"{r}_mean_score"] for r in routers] for b in buckets
    ])

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(buckets))
    width = 0.27
    colors = {"vqc": "#6699ff", "oracle": "#66bb77", "random": "#bbbbbb"}
    for j, r in enumerate(routers):
        ax.bar(x + (j - 1) * width, values[:, j], width, label=r, color=colors[r],
               edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={stratified[b]['n']})" for b in buckets])
    ax.set_ylabel("Mean judge score (0-5)")
    ax.set_ylim(0, 5)
    ax.set_title("C2 diagnostic: scores stratified by VQC routing correctness")
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _render_top10_md(top_gaps: list[dict], out_path: Path) -> None:
    lines = ["# C2 diagnostic — Top-10 queries by oracle-vqc score gap",
             "",
             "*Auto-generated; append human-observed patterns at the bottom.*",
             ""]
    for k, row in enumerate(top_gaps, 1):
        lines.extend([
            f"## #{k} — gap = {row['gap']}",
            "",
            f"**Question:** {row['question']}",
            "",
            f"**Expected domain:** `{row['expected_domain']}`",
            "",
            f"### Oracle (routed to `{row['expected_domain']}`, score {row['oracle_score']})",
            "",
            "> " + row["oracle_answer"][:500].replace("\n", "\n> "),
            "",
            f"### VQC (routed to `{row['vqc_routed_domain']}`, score {row['vqc_score']})",
            "",
            "> " + row["vqc_answer"][:500].replace("\n", "\n> "),
            "",
            f"### Random (routed to `{row['random_routed_domain']}`, score {row['random_score']})",
            "",
            "> " + row["random_answer"][:500].replace("\n", "\n> "),
            "",
            "---",
            "",
        ])
    lines.extend([
        "## Patterns observed (hand-written by reviewer)",
        "",
        "_Edit after reading the 10 pairs above. Candidate patterns: persona mismatch,",
        "technical depth gap, answer length/tone, off-topic drift, hallucinated code._",
        "",
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def run(
    *,
    input_path: Path,
    out_json: Path,
    out_per_domain_pdf: Path,
    out_stratified_pdf: Path,
    out_top10_md: Path,
    domains: list[str],
) -> int:
    data = json.loads(Path(input_path).read_text())
    per_domain = analyze_per_domain(data, domains=domains)
    stratified = analyze_stratified(data)
    top_gaps = top_10_by_gap(data, k=10)

    _render_per_domain_pdf(per_domain, out_per_domain_pdf)
    _render_stratified_pdf(stratified, out_stratified_pdf)
    _render_top10_md(top_gaps, out_top10_md)

    # Strip the large answer fields from the JSON summary
    top_gaps_slim = [
        {k: v for k, v in row.items() if not k.endswith("_answer")}
        for row in top_gaps
    ]

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps({
        "per_domain": per_domain,
        "stratified": stratified,
        "top_gaps": top_gaps_slim,
        "config": {
            "domains": domains,
            "input": str(input_path),
            "n_queries": len(data["results"]["oracle"]["per_query"]),
        },
    }, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-per-domain-pdf", type=Path, required=True)
    p.add_argument("--out-stratified-pdf", type=Path, required=True)
    p.add_argument("--out-top10-md", type=Path, required=True)
    p.add_argument("--domains", default=",".join(_DEFAULT_DOMAINS))
    args = p.parse_args()

    return run(
        input_path=args.input,
        out_json=args.out_json,
        out_per_domain_pdf=args.out_per_domain_pdf,
        out_stratified_pdf=args.out_stratified_pdf,
        out_top10_md=args.out_top10_md,
        domains=[d.strip() for d in args.domains.split(",") if d.strip()],
    )


if __name__ == "__main__":
    raise SystemExit(main())
