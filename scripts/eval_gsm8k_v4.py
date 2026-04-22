#!/usr/bin/env python3
"""GSM8K grade-school math eval for V4 SOTA adapters.

Loads the base Qwen3.6-35B-A3B model via mlx_lm, optionally applies a LoRA
adapter, then generates deterministic completions for GSM8K test problems and
extracts the final numeric answer to compute accuracy (pass@1).

Mirrors the CLI shape of eval_humaneval_v4.py. Fixture is auto-downloaded
from HF (`openai/gsm8k`, `main`, split=test) on first run and cached to
~/.cache/micro-kiki-evals/gsm8k_test.jsonl. Pass an explicit --fixture to
override.

Usage (on Studio):
    python scripts/eval_gsm8k_v4.py \
        --base-model /Users/clems/KIKI-Mac_tunner/models/Qwen3.6-35B-A3B \
        --adapter /Users/clems/KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b-v4-sota/math \
        --output /tmp/gsm8k-math.json \
        --label math \
        --n 50
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gsm8k_v4")

DEFAULT_CACHE = Path.home() / ".cache" / "micro-kiki-evals" / "gsm8k_test.jsonl"

# ---------------------------------------------------------------------------
# Fixture generation
# ---------------------------------------------------------------------------


def _ensure_fixture(fixture: Path | None) -> Path:
    """Return fixture path; materialize from HF if missing."""
    target = fixture if fixture is not None else DEFAULT_CACHE
    if target.exists() and target.stat().st_size > 0:
        logger.info("reusing fixture at %s", target)
        return target

    logger.info("materializing GSM8K fixture at %s", target)
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            f"pip install datasets (needed to build GSM8K fixture): {e}"
        ) from e

    target.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    with target.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            rec = {
                "task_id": f"gsm8k/{i}",
                "question": row["question"],
                "answer": row["answer"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("wrote %d problems to %s", len(ds), target)
    return target


# ---------------------------------------------------------------------------
# Prompt & answer parsing
# ---------------------------------------------------------------------------


def _build_prompt(problem: dict[str, Any]) -> str:
    return f"{problem['question']}\nAnswer:"


_RE_HASHES = re.compile(r"####\s*(-?[\d,\.]+)")
_RE_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def _normalize_number(s: str) -> str | None:
    """Strip commas and trailing periods; return canonical string or None."""
    if s is None:
        return None
    s = s.strip().replace(",", "")
    # Drop trailing period (end-of-sentence, not decimal)
    if s.endswith(".") and s.count(".") == 1:
        s = s[:-1]
    if s in ("", "-", ".", "-."):
        return None
    try:
        # Prefer int when possible for exact comparison.
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except ValueError:
        return None


def _extract_answer(text: str) -> str | None:
    """Pull the final number out of a GSM8K answer-style string.

    1. Prefer the segment after '####' if present.
    2. Otherwise, take the last number in the text.
    """
    if not text:
        return None
    m = _RE_HASHES.search(text)
    if m:
        return _normalize_number(m.group(1))
    nums = _RE_NUMBER.findall(text)
    if not nums:
        return None
    return _normalize_number(nums[-1])


# ---------------------------------------------------------------------------
# MLX loader (lazy)
# ---------------------------------------------------------------------------


def _load_mlx(base_model: Path, adapter: Path | None, max_tokens: int):
    """Returns generate_fn(prompt)->str."""
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    logger.info("loading base model from %s", base_model)
    t0 = time.monotonic()
    model, tokenizer = load(str(base_model))
    logger.info("base loaded in %.1fs", time.monotonic() - t0)

    if adapter is not None:
        from mlx_lm.tuner.utils import load_adapters

        t1 = time.monotonic()
        logger.info("applying adapter: %s", adapter)
        load_adapters(model, str(adapter))
        logger.info("adapter applied in %.1fs", time.monotonic() - t1)

    sampler = make_sampler(temp=0.0)

    def _gen(prompt: str) -> str:
        return generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )

    return _gen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True, type=Path)
    p.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="LoRA adapter directory. Omit for base-only run.",
    )
    p.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="JSONL fixture path. If omitted, auto-downloaded to "
        f"{DEFAULT_CACHE}.",
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--label", required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse fixture only; do not load the model.",
    )
    args = p.parse_args()

    fixture = _ensure_fixture(args.fixture)

    problems: list[dict[str, Any]] = []
    with fixture.open() as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    problems = problems[: args.n]
    logger.info("loaded %d problems from %s", len(problems), fixture)

    out: dict[str, Any] = {
        "label": args.label,
        "base_model": str(args.base_model),
        "adapter": str(args.adapter) if args.adapter else None,
        "n_problems": len(problems),
        "max_tokens": args.max_tokens,
        "per_problem": [],
    }

    if args.dry_run:
        logger.warning("dry run — skipping mlx_lm")
        out["pass@1"] = None
        out["status"] = "dry_run"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        logger.info("dry-run wrote %s (N=%d)", args.output, len(problems))
        return 0

    try:
        gen = _load_mlx(args.base_model, args.adapter, args.max_tokens)
    except ImportError as e:
        logger.error("mlx_lm unavailable: %s", e)
        out["status"] = "mlx_unavailable"
        out["error"] = str(e)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        return 2

    passed = 0
    total_gen_s = 0.0
    for i, prob in enumerate(problems, 1):
        prompt = _build_prompt(prob)
        expected = _extract_answer(prob["answer"])
        t_g = time.monotonic()
        try:
            raw = gen(prompt)
        except Exception as e:  # noqa: BLE001
            logger.warning("gen failed for %s: %s", prob["task_id"], e)
            raw = ""
        gen_s = time.monotonic() - t_g
        total_gen_s += gen_s
        predicted = _extract_answer(raw)
        ok = (expected is not None) and (predicted == expected)
        logger.info(
            "[%d/%d] %s: %s (exp=%s pred=%s, %.1fs)",
            i,
            len(problems),
            prob["task_id"],
            "PASS" if ok else "FAIL",
            expected,
            predicted,
            gen_s,
        )
        if ok:
            passed += 1
        out["per_problem"].append(
            {
                "question_id": prob["task_id"],
                "expected": expected,
                "predicted": predicted,
                "ok": ok,
                "gen_s": round(gen_s, 2),
                "completion": (raw or "")[:800],
            }
        )

    out["pass@1"] = passed / len(problems) if problems else 0.0
    out["total_gen_s"] = round(total_gen_s, 1)
    out["status"] = "ok"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info(
        "DONE label=%s pass@1=%.3f (%d/%d), total_gen=%.1fs, wrote %s",
        args.label,
        out["pass@1"],
        passed,
        len(problems),
        total_gen_s,
        args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
