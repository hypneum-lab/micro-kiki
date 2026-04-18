#!/usr/bin/env python
"""Measure per-layer cos-angle (and optional win-rate) between two LoRA stacks.

Phase 1a of OPLoRA: angle-only (informational), status ``angle_only_partial``.
Phase 1b (this script, with new flags): full gate — angle AND win-rate with
AND-logic rollback. See ``.omc/brainstorm-oplora.md``.

Angle-only (back-compat, phase 1a):
    python scripts/measure_forgetting.py \\
        --prior-adapter path/to/prior/adapter_model.safetensors \\
        --new-adapter   path/to/new/adapter_model.safetensors \\
        [--output results/forgetting-<stack>.json]

Full gate (phase 1b):
    python scripts/measure_forgetting.py \\
        --prior-adapter ... --new-adapter ... \\
        --eval-dataset data/eval/<stack>.jsonl \\
        --generate-fn-module src.serving.vllm_client:generate \\
        --winrate-baseline-score 0.82 \\
        [--output results/forgetting-<stack>.json]

Full-gate logic: ``angle_degrees_mean < 30° AND winrate_drop > 0.03`` →
``gate_status = "fail"`` and exit code 1. Otherwise ``gate_status = "pass"``
and exit code 0. Angle-only (no win-rate inputs) keeps exit 0 (informational).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

# Ensure ``src`` is importable when running from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.forgetting import (  # noqa: E402
    ANGLE_THRESHOLD,
    WINRATE_DROP_THRESHOLD,
    measure_forgetting_signal,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure LoRA-delta cos-angle (and optional win-rate) between "
            "two adapter stacks."
        )
    )
    parser.add_argument(
        "--prior-adapter",
        required=True,
        type=Path,
        help="Path to prior stack adapter_model.safetensors",
    )
    parser.add_argument(
        "--new-adapter",
        required=True,
        type=Path,
        help="Path to newly-trained stack adapter_model.safetensors",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON report.",
    )
    parser.add_argument(
        "--eval-dataset",
        type=Path,
        default=None,
        help="JSONL of {prompt, reference} for the prior domain (phase 1b).",
    )
    parser.add_argument(
        "--generate-fn-module",
        type=str,
        default=None,
        help=(
            "Python path to generation callable, e.g. "
            "'src.serving.vllm_client:generate' (phase 1b)."
        ),
    )
    parser.add_argument(
        "--winrate-baseline-score",
        type=float,
        default=None,
        help="Reference win-rate from the prior adapter (phase 1b).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Send loguru to stderr; keep stdout clean for JSON.
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    args = _parse_args(argv)

    # Validate win-rate flag coherence: all three or none.
    winrate_flags = (
        args.eval_dataset,
        args.generate_fn_module,
        args.winrate_baseline_score,
    )
    some_set = any(f is not None for f in winrate_flags)
    all_set = all(f is not None for f in winrate_flags)
    if some_set and not all_set:
        logger.error(
            "--eval-dataset, --generate-fn-module and "
            "--winrate-baseline-score must all be provided together."
        )
        return 2

    report = measure_forgetting_signal(
        prior_adapter_path=args.prior_adapter,
        new_adapter_path=args.new_adapter,
        eval_dataset=args.eval_dataset if all_set else None,
        generate_fn=args.generate_fn_module if all_set else None,
        winrate_baseline=args.winrate_baseline_score if all_set else None,
    )

    status = report.get("gate_status")
    if status == "fail":
        logger.warning(
            "GATE FAIL: %s",
            report.get("warning") or "angle<30° AND winrate_drop>0.03",
        )
    elif status == "angle_only_partial":
        mean = report.get("angle_degrees_mean")
        if isinstance(mean, (int, float)) and mean < ANGLE_THRESHOLD:
            logger.warning(
                "mean angle %.2f° < %.1f° — potential forgetting; "
                "run win-rate eval for full gate (drop threshold %.2f).",
                mean,
                ANGLE_THRESHOLD,
                WINRATE_DROP_THRESHOLD,
            )

    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        logger.info("wrote {}", args.output)

    # Full-gate fail → non-zero exit for CI. angle_only and pass → 0.
    return 1 if status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
