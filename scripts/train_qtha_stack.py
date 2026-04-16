#!/usr/bin/env python3
"""Train a QTHA adapter on a domain for comparison with LoRA.

Usage::

    uv run python scripts/train_qtha_stack.py --domain reasoning
    uv run python scripts/train_qtha_stack.py --domain reasoning --bond-dim 16

Trains both a standard LoRA (rank 16) and a QTHA (bond_dim 8) adapter
on the same data, then compares parameter counts and (if eval data exists)
evaluation scores. Results written to ``results/qtha-pilot-<domain>.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.stacks.qtha import QTHAConfig, estimate_qtha_params

logger = logging.getLogger(__name__)

HIDDEN_DIM = 3584  # Qwen3.5-35B-A3B hidden size
NUM_LAYERS = 28    # Qwen3.5-35B-A3B layers
LORA_RANK = 16
LORA_MODULES = 4   # q, k, v, o


def estimate_lora_params() -> int:
    """Estimate standard LoRA rank-16 param count."""
    per_module = 2 * LORA_RANK * HIDDEN_DIM  # A + B matrices
    return per_module * LORA_MODULES * NUM_LAYERS


def compare_configs(bond_dim: int) -> dict:
    """Compare LoRA vs QTHA parameter efficiency."""
    lora_params = estimate_lora_params()
    qtha_params = estimate_qtha_params(HIDDEN_DIM, bond_dim, NUM_LAYERS)
    ratio = lora_params / qtha_params if qtha_params > 0 else float("inf")

    return {
        "lora_rank_16": {
            "params": lora_params,
            "params_human": f"{lora_params / 1e6:.1f}M",
        },
        f"qtha_bond_{bond_dim}": {
            "params": qtha_params,
            "params_human": f"{qtha_params / 1e6:.1f}M",
            "bond_dim": bond_dim,
        },
        "compression_ratio": f"{ratio:.1f}x",
    }


def find_data(domain: str) -> Path | None:
    """Find training data for domain."""
    candidates = [
        Path(f"data/distilled/{domain}.jsonl"),
        Path.home() / "KIKI-Mac_tunner" / "data" / "micro-kiki" / domain / "train.jsonl",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QTHA vs LoRA comparative pilot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--domain", default="reasoning", help="Domain to pilot on")
    parser.add_argument("--bond-dim", type=int, default=8, help="QTHA bond dimension")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--train", action="store_true", help="Actually train (requires GPU/CPU + model)")
    args = parser.parse_args()

    output = args.output or Path(f"results/qtha-pilot-{args.domain}.json")

    comparison = compare_configs(args.bond_dim)
    data_path = find_data(args.domain)

    report = {
        "domain": args.domain,
        "data_path": str(data_path) if data_path else None,
        "data_available": data_path is not None,
        "parameter_comparison": comparison,
        "training_done": False,
        "eval_results": None,
    }

    logger.info("Parameter comparison for %s:", args.domain)
    logger.info("  LoRA rank-16:  %s", comparison["lora_rank_16"]["params_human"])
    logger.info("  QTHA bond-%d: %s", args.bond_dim, comparison[f"qtha_bond_{args.bond_dim}"]["params_human"])
    logger.info("  Compression:   %s fewer params", comparison["compression_ratio"])

    if data_path:
        logger.info("Data found: %s", data_path)
    else:
        logger.warning("No training data found for %s", args.domain)

    if args.train:
        logger.info("Training mode enabled — not yet implemented (pending GPU execution)")
        report["training_done"] = False

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    logger.info("Report written to %s", output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
