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
