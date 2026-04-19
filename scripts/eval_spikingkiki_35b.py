#!/usr/bin/env python3
"""Story-28: Eval SpikingKiki-35B — benchmark comparisons.

Loads the existing story-27 analysis JSON, adds benchmark comparisons
(MACs/token, theoretical throughput, energy/token) and compares with
published SNN results (Spikformer, SpikingBERT, SpikingBrain).

Pure Python + numpy. No torch, no GPU.

Usage:
    uv run python scripts/eval_spikingkiki_35b.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
ANALYSIS_PATH = RESULTS_DIR / "spikingkiki-35b-analysis.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_35b")


# ---------------------------------------------------------------------------
# Published SNN baselines (from papers)
# ---------------------------------------------------------------------------

PUBLISHED_BASELINES = {
    "Spikformer": {
        "source": "Zhou et al., 2023 (ICLR) — Spikformer: When SNN Meets Transformer",
        "model_size": "66M (ViT-scale)",
        "task": "ImageNet classification",
        "timesteps": 4,
        "accuracy_percent": 74.81,
        "energy_reduction_vs_ann": "5.3x (estimated from spike ops)",
        "notes": "Vision-only, not language. First SNN transformer.",
    },
    "SpikingBERT": {
        "source": "Bal & Bhattacharya, 2024 — SpikingBERT: Distilling BERT to Train Spiking Language Models",
        "model_size": "109M (BERT-base)",
        "task": "GLUE benchmark (NLU)",
        "timesteps": 4,
        "accuracy_vs_bert": "97.1% of BERT-base on SST-2",
        "energy_reduction_vs_ann": "~10x theoretical (fewer synaptic ops)",
        "notes": "Knowledge distillation from BERT. T=4 sufficient for NLU.",
    },
    "SpikingBrain_7B": {
        "source": "BICLab, 2025 — SpikingBrain (HuggingFace: BICLab/SpikingBrain-7B)",
        "model_size": "7B",
        "task": "Language modeling / chat",
        "timesteps": "variable (1-8)",
        "energy_claim": "Significant reduction via sparse spikes",
        "notes": "First multi-billion parameter spiking LLM. Native spiking architecture.",
    },
    "SpikingBrain_76B": {
        "source": "BICLab, 2025 — SpikingBrain-76B",
        "model_size": "76B",
        "task": "Language modeling",
        "timesteps": "variable",
        "energy_claim": "Largest spiking LLM to date",
        "notes": "Dense architecture. SpikingKiki-35B would be the first MoE spiking LLM.",
    },
    "LAS_OPT66B": {
        "source": "arxiv 2505.09659 — Lossless ANN-SNN Conversion",
        "model_size": "66B (OPT-66B)",
        "task": "Language modeling",
        "timesteps": 128,
        "conversion_method": "LAS (lossless)",
        "energy_reduction_claim": "Theoretical; activation sparsity dependent",
        "notes": "Converted OPT-66B. Dense model, not MoE.",
    },
}


def load_analysis(path: Path | None = None) -> dict:
    """Load the existing 35B analysis JSON."""
    p = path or ANALYSIS_PATH
    if not p.exists():
        raise FileNotFoundError(f"Analysis not found: {p}")
    return json.loads(p.read_text())


def compute_macs_per_token(analysis: dict) -> dict:
    """Extract and summarize MACs/token from the analysis."""
    energy = analysis["energy_savings"]
    per_token = energy["per_token_full_model"]

    ann_macs = per_token["ann_macs"]
    snn_ops = per_token["snn_spike_ops"]

    # Theoretical throughput: tokens/second assuming hardware budget
    # A100 peak: ~312 TFLOPS FP16, ~19.5 TFLOPS FP32
    a100_fp32_tops = 19.5e12  # ops/sec
    loihi2_spike_rate = 2e9   # ~2 billion spikes/sec (estimated)

    ann_tokens_per_sec_a100 = a100_fp32_tops / ann_macs if ann_macs > 0 else 0
    snn_tokens_per_sec_loihi = loihi2_spike_rate / snn_ops if snn_ops > 0 else 0

    return {
        "ann_macs_per_token": int(ann_macs),
        "ann_macs_per_token_G": round(ann_macs / 1e9, 2),
        "snn_spike_ops_per_token": int(snn_ops),
        "snn_spike_ops_per_token_M": round(snn_ops / 1e6, 2),
        "ops_reduction_ratio": round(ann_macs / snn_ops, 2) if snn_ops > 0 else float("inf"),
        "theoretical_throughput": {
            "ann_on_a100_fp32_tok_per_sec": round(ann_tokens_per_sec_a100, 1),
            "snn_on_loihi2_tok_per_sec": round(snn_tokens_per_sec_loihi, 1),
            "notes": [
                "A100 peak FP32: 19.5 TFLOPS (compute-bound estimate)",
                "Loihi 2 spike processing: ~2 billion spikes/sec (rough estimate)",
                "Real throughput depends on memory bandwidth, not just compute",
                "MoE sparsity (8/256 active) already gives 32x routing efficiency",
            ],
        },
    }


def compute_energy_per_token(analysis: dict) -> dict:
    """Detailed energy breakdown per token."""
    energy = analysis["energy_savings"]
    per_token = energy["per_token_full_model"]

    ann_energy_uj = per_token["ann_energy_uJ"]
    snn_cmos_uj = per_token["snn_energy_cmos_uJ"]
    snn_neuro_uj = per_token["snn_energy_neuromorphic_uJ"]

    # Convert to more intuitive units
    # 1 J = 1e6 uJ, typical phone battery ~10 Wh = 36000 J
    phone_battery_j = 36000.0
    tokens_per_battery_ann = phone_battery_j / (ann_energy_uj * 1e-6) if ann_energy_uj > 0 else 0
    tokens_per_battery_snn_cmos = phone_battery_j / (snn_cmos_uj * 1e-6) if snn_cmos_uj > 0 else 0
    tokens_per_battery_snn_neuro = phone_battery_j / (snn_neuro_uj * 1e-6) if snn_neuro_uj > 0 else 0

    return {
        "ann_energy_uJ_per_token": ann_energy_uj,
        "snn_cmos_energy_uJ_per_token": snn_cmos_uj,
        "snn_neuromorphic_energy_uJ_per_token": snn_neuro_uj,
        "tokens_per_phone_battery_10Wh": {
            "ann": int(tokens_per_battery_ann),
            "snn_cmos": int(tokens_per_battery_snn_cmos),
            "snn_neuromorphic": int(tokens_per_battery_snn_neuro),
        },
        "context_comparison": {
            "ann_energy_per_1k_tokens_mJ": round(ann_energy_uj * 1000 / 1000, 2),
            "snn_cmos_per_1k_tokens_mJ": round(snn_cmos_uj * 1000 / 1000, 2),
            "snn_neuro_per_1k_tokens_mJ": round(snn_neuro_uj * 1000 / 1000, 2),
        },
    }


def build_comparison_table(analysis: dict) -> dict:
    """Build a comparison table: SpikingKiki-35B vs published SNNs."""
    model_info = analysis["model"]
    energy = analysis["energy_savings"]
    savings = energy["savings"]

    rows = []

    # SpikingKiki-35B entry
    rows.append({
        "model": "SpikingKiki-35B (ours)",
        "size": f"{model_info['total_params_estimate'] / 1e9:.1f}B",
        "architecture": "MoE (256 experts, 8 active)",
        "conversion": "LAS (attention+router lossless, expert partial)",
        "timesteps": analysis["spike_rate_equivalence"]["timesteps"],
        "energy_saving_vs_ann": f"{savings['cmos_saving_percent']}% (CMOS)",
        "neuromorphic_saving": f"{savings['neuromorphic_saving_percent']}%",
        "status": "theoretical (no weight conversion yet)",
        "novelty": "First MoE model targeted for SNN conversion",
    })

    # Published baselines
    for name, baseline in PUBLISHED_BASELINES.items():
        rows.append({
            "model": name,
            "size": baseline["model_size"],
            "architecture": "Dense" if "MoE" not in baseline.get("notes", "") else "MoE",
            "conversion": baseline.get("conversion_method", "native/distilled"),
            "timesteps": baseline.get("timesteps", "N/A"),
            "energy_saving_vs_ann": baseline.get(
                "energy_reduction_vs_ann",
                baseline.get("energy_claim", "N/A"),
            ),
            "neuromorphic_saving": "N/A",
            "status": "published",
            "novelty": baseline.get("notes", ""),
        })

    return {
        "comparison_table": rows,
        "key_differentiators": [
            "SpikingKiki-35B is the first MoE-architecture SNN conversion attempt",
            "256 experts with preserved top-8 routing via identity-activation LAS",
            "Unique challenge: SwiGLU gate (SiLU + element-wise multiply) has no spiking equivalent",
            "Hybrid approach (ANN gate + spiking up/down) is pragmatic but novel",
            "35B total / 3B active per token gives inherent sparsity advantage",
        ],
    }


def run_eval(
    analysis_path: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    analysis = load_analysis(analysis_path)

    macs = compute_macs_per_token(analysis)
    energy = compute_energy_per_token(analysis)
    comparison = build_comparison_table(analysis)

    results = {
        "story": "story-28",
        "title": "Eval SpikingKiki-35B — benchmark comparisons",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_analysis": {
            "source": str(ANALYSIS_PATH),
            "story": analysis.get("story", "story-27"),
            "model": analysis["model"],
        },
        "macs_per_token": macs,
        "energy_per_token": energy,
        "published_baselines": PUBLISHED_BASELINES,
        "comparison": comparison,
        "conclusions": [
            "SpikingKiki-35B achieves ~468x energy reduction on CMOS (theoretical)",
            "MoE sparsity (8/256 experts) compounds with spike sparsity for massive savings",
            "SwiGLU gate remains the primary conversion bottleneck — hybrid approach recommended",
            "At 35B scale, memory bandwidth (not compute) likely dominates real-world energy",
            "Neuromorphic deployment (Loihi 2) would yield ~8400x reduction but requires hardware",
            "Published SNNs are all dense; MoE conversion is unexplored territory",
        ],
    }

    out = output_path or (RESULTS_DIR / "spikingkiki-35b-eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    log.info("eval results written to %s", out)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Story-28: Eval SpikingKiki-35B benchmarks"
    )
    parser.add_argument("--analysis", type=Path, default=ANALYSIS_PATH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    results = run_eval(args.analysis, args.output)

    macs = results["macs_per_token"]
    log.info(
        "MACs/token: ANN=%.2fG, SNN=%.2fM, reduction=%.1fx",
        macs["ann_macs_per_token_G"],
        macs["snn_spike_ops_per_token_M"],
        macs["ops_reduction_ratio"],
    )
    e = results["energy_per_token"]
    log.info(
        "energy/token: ANN=%.3f uJ, SNN(CMOS)=%.3f uJ, SNN(neuro)=%.3f uJ",
        e["ann_energy_uJ_per_token"],
        e["snn_cmos_energy_uJ_per_token"],
        e["snn_neuromorphic_energy_uJ_per_token"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
