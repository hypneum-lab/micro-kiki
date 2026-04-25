#!/usr/bin/env python3
"""MBPP pass@1 comparison: base vs python (original) vs python-v2 (clean data).

Runs MBPP eval on three configurations sequentially and produces a summary
comparing pass@1 against the base model, to verify whether the python-v2
adapter (trained without stubs) fixes the ~-10pp regression seen in python v1.

Usage:
    python scripts/eval_mbpp_python_v2.py \
        --base-model /Users/clems/KIKI-Mac_tunner/models/Qwen3.6-35B-A3B \
        --adapter-python /Users/clems/KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b-v4-sota/python \
        --adapter-python-v2 /Users/clems/KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b-v4-sota/python-v2 \
        --output results/mbpp-python-v2-comparison.json \
        --n 100

Skip individual configurations with --skip-base / --skip-python / --skip-python-v2.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Reuse all core eval logic from eval_mbpp_v4
sys.path.insert(0, str(Path(__file__).parent))
from eval_mbpp_v4 import (
    _build_prompt,
    _ensure_fixture,
    _extract_completion,
    _load_mlx,
    _run_tests,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mbpp_python_v2")

# ---------------------------------------------------------------------------
# Single-configuration eval loop
# ---------------------------------------------------------------------------


def _eval_config(
    label: str,
    problems: list[dict[str, Any]],
    base_model: Path,
    adapter: Path | None,
    max_tokens: int,
) -> dict[str, Any]:
    """Run MBPP on one configuration; return result dict (no file I/O)."""
    result: dict[str, Any] = {
        "label": label,
        "base_model": str(base_model),
        "adapter": str(adapter) if adapter is not None else None,
        "n_problems": len(problems),
        "max_tokens": max_tokens,
        "per_problem": [],
    }

    try:
        gen = _load_mlx(base_model, adapter, max_tokens)
    except ImportError as e:
        logger.error("mlx_lm unavailable for %s: %s", label, e)
        result["status"] = "mlx_unavailable"
        result["error"] = str(e)
        result["pass@1"] = None
        return result

    passed = 0
    total_gen_s = 0.0
    for i, prob in enumerate(problems, 1):
        prompt = _build_prompt(prob)
        t_g = time.monotonic()
        try:
            raw = gen(prompt)
        except Exception as e:  # noqa: BLE001
            logger.warning("[%s] gen failed for %s: %s", label, prob["task_id"], e)
            raw = ""
        gen_s = time.monotonic() - t_g
        total_gen_s += gen_s

        completion = _extract_completion(raw, prompt)
        ok, err = _run_tests(
            prob.get("test_setup_code", ""), completion, prob["test_list"]
        )
        logger.info(
            "[%s %d/%d] %s: %s (%.1fs)",
            label,
            i,
            len(problems),
            prob["task_id"],
            "PASS" if ok else "FAIL",
            gen_s,
        )
        if ok:
            passed += 1
        result["per_problem"].append(
            {
                "task_id": prob["task_id"],
                "passed": ok,
                "gen_s": round(gen_s, 2),
                "completion": completion[:800],
                "error": (err[:200] if err else ""),
            }
        )

    result["pass@1"] = passed / len(problems) if problems else 0.0
    result["total_gen_s"] = round(total_gen_s, 1)
    result["status"] = "ok"
    logger.info(
        "[%s] DONE pass@1=%.3f (%d/%d) in %.1fs",
        label,
        result["pass@1"],
        passed,
        len(problems),
        total_gen_s,
    )
    return result


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_summary(configs: list[dict[str, Any]]) -> None:
    base_score: float | None = None
    for cfg in configs:
        if cfg["label"] == "base":
            base_score = cfg.get("pass@1")
            break

    print("\n" + "=" * 60)
    print("MBPP pass@1 — python v1 vs v2 comparison")
    print("=" * 60)
    header = f"{'Config':<16} {'pass@1':>8} {'Δ vs base':>12} {'Status':<14}"
    print(header)
    print("-" * 60)
    for cfg in configs:
        score = cfg.get("pass@1")
        status = cfg.get("status", "unknown")
        if score is None:
            score_str = "  N/A"
            delta_str = "       N/A"
        else:
            score_str = f"{score * 100:6.1f}%"
            if base_score is not None and cfg["label"] != "base":
                delta = (score - base_score) * 100
                sign = "+" if delta >= 0 else ""
                delta_str = f"  {sign}{delta:.1f}pp"
            else:
                delta_str = "        —"
        print(f"{cfg['label']:<16} {score_str:>8} {delta_str:>12}  {status:<14}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare MBPP pass@1 for base / python / python-v2 adapters."
    )
    ap.add_argument("--base-model", required=True, type=Path, help="Qwen3.6-35B path")
    ap.add_argument(
        "--adapter-python",
        type=Path,
        default=Path(
            "/Users/clems/KIKI-Mac_tunner/output/micro-kiki/"
            "lora-qwen36-35b-v4-sota/python"
        ),
        help="Path to the original python adapter (v1)",
    )
    ap.add_argument(
        "--adapter-python-v2",
        type=Path,
        default=Path(
            "/Users/clems/KIKI-Mac_tunner/output/micro-kiki/"
            "lora-qwen36-35b-v4-sota/python-v2"
        ),
        help="Path to the retrained python adapter (v2, clean data)",
    )
    ap.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Optional JSONL fixture; auto-downloaded if omitted",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("results/mbpp-python-v2-comparison.json"),
    )
    ap.add_argument("--n", type=int, default=100, help="Number of MBPP problems")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip base-model eval (useful if already cached)",
    )
    ap.add_argument("--skip-python", action="store_true", help="Skip python v1 eval")
    ap.add_argument(
        "--skip-python-v2", action="store_true", help="Skip python v2 eval"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify fixture + args without loading the model",
    )
    args = ap.parse_args()

    fixture = _ensure_fixture(args.fixture)

    problems: list[dict[str, Any]] = []
    with fixture.open() as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    problems = problems[: args.n]
    logger.info("loaded %d problems (N=%d)", len(problems), args.n)

    if args.dry_run:
        logger.warning("dry-run — skipping MLX inference")
        dry_configs = []
        if not args.skip_base:
            dry_configs.append(
                {"label": "base", "adapter": None, "pass@1": None, "status": "dry_run"}
            )
        if not args.skip_python:
            dry_configs.append(
                {
                    "label": "python",
                    "adapter": str(args.adapter_python),
                    "pass@1": None,
                    "status": "dry_run",
                }
            )
        if not args.skip_python_v2:
            dry_configs.append(
                {
                    "label": "python-v2",
                    "adapter": str(args.adapter_python_v2),
                    "pass@1": None,
                    "status": "dry_run",
                }
            )
        out = {
            "n_problems": len(problems),
            "max_tokens": args.max_tokens,
            "base_model": str(args.base_model),
            "configs": dry_configs,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        logger.info("dry-run wrote %s", args.output)
        return 0

    configs: list[dict[str, Any]] = []

    if not args.skip_base:
        logger.info("--- evaluating: base ---")
        configs.append(
            _eval_config("base", problems, args.base_model, None, args.max_tokens)
        )

    if not args.skip_python:
        logger.info("--- evaluating: python (original) ---")
        configs.append(
            _eval_config(
                "python", problems, args.base_model, args.adapter_python, args.max_tokens
            )
        )

    if not args.skip_python_v2:
        if not args.adapter_python_v2.exists():
            logger.warning(
                "python-v2 adapter not found at %s — skipping", args.adapter_python_v2
            )
        else:
            logger.info("--- evaluating: python-v2 (clean data) ---")
            configs.append(
                _eval_config(
                    "python-v2",
                    problems,
                    args.base_model,
                    args.adapter_python_v2,
                    args.max_tokens,
                )
            )

    out = {
        "n_problems": len(problems),
        "max_tokens": args.max_tokens,
        "base_model": str(args.base_model),
        "configs": configs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("results written to %s", args.output)

    _print_summary(configs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
