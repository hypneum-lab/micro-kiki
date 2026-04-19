#!/usr/bin/env python3
"""Story-14: Forgetting check — verify LoRA adapters preserve general knowledge.

Loads base model, then base + each adapter, runs the same general-knowledge
prompts, and compares responses. If the adapter significantly changes general
answers, it indicates catastrophic forgetting.

Metric: response similarity (keyword overlap + length ratio). A future version
can compare output logits directly.

Usage:
    # Single domain:
    .venv/bin/python3 scripts/check_forgetting.py --domain kicad-dsl

    # All 10 domains:
    .venv/bin/python3 scripts/check_forgetting.py --all

    # Custom prompts file:
    .venv/bin/python3 scripts/check_forgetting.py --all --prompts data/forgetting-prompts.jsonl

Designed for Mac Studio M3 Ultra 512 GB (MLX). Restart-friendly: saves results
incrementally to results/forgetting-check.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ALL_DOMAINS = [
    "kicad-dsl", "spice", "emc", "stm32", "embedded",
    "freecad", "platformio", "power", "dsp", "electronics",
]

BASE_MODEL_PATH = str(PROJECT_ROOT / "models" / "qwen3.5-35b-a3b")
ADAPTERS_DIR = PROJECT_ROOT / "outputs" / "stacks"

RESULTS_PATH = PROJECT_ROOT / "results" / "forgetting-check.json"

# General-knowledge prompts that should produce stable answers regardless
# of which LoRA adapter is loaded.
DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Explain Newton's three laws of motion in simple terms.",
    "Write a Python function to compute the Fibonacci sequence.",
    "What is the difference between TCP and UDP?",
    "Summarize the key events of World War II in 3 sentences.",
]

# Similarity thresholds
LENGTH_RATIO_THRESHOLD = 0.3   # response length ratio must be > this
KEYWORD_OVERLAP_THRESHOLD = 0.4  # keyword overlap must be > this
OVERALL_PASS_THRESHOLD = 0.5   # combined similarity must be > this

MAX_TOKENS = 256


# ---------------------------------------------------------------------------
# MLX model loading
# ---------------------------------------------------------------------------

def _setup_metal():
    """Configure Metal buffer limits for Mac Studio 512 GB."""
    import mlx.core as mx
    mx.set_memory_limit(460 * 1024 * 1024 * 1024)  # 460 GB
    mx.set_cache_limit(32 * 1024 * 1024 * 1024)    # 32 GB


def _load_model(model_path: str, adapter_path: str | None = None):
    """Load model (and optionally adapter) via mlx_lm."""
    from mlx_lm import load

    kwargs = {"model_path": model_path}
    if adapter_path:
        kwargs["adapter_path"] = adapter_path
    model, tokenizer = load(**kwargs)
    return model, tokenizer


def _generate(model, tokenizer, prompt: str, max_tokens: int = MAX_TOKENS) -> str:
    """Generate a response using mlx_lm."""
    from mlx_lm import generate

    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    response = generate(
        model,
        tokenizer,
        prompt=chat_prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    return response


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------

def _tokenize_simple(text: str) -> set[str]:
    """Split text into lowercase word tokens."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def compute_similarity(base_response: str, adapted_response: str) -> dict:
    """Compare two responses and return similarity metrics.

    Returns dict with:
        - length_ratio: min(len_a, len_b) / max(len_a, len_b)
        - keyword_overlap: Jaccard similarity of word tokens
        - combined: average of the two
        - passed: True if combined > OVERALL_PASS_THRESHOLD
    """
    len_base = max(len(base_response), 1)
    len_adapted = max(len(adapted_response), 1)
    length_ratio = min(len_base, len_adapted) / max(len_base, len_adapted)

    tokens_base = _tokenize_simple(base_response)
    tokens_adapted = _tokenize_simple(adapted_response)

    if tokens_base or tokens_adapted:
        intersection = tokens_base & tokens_adapted
        union = tokens_base | tokens_adapted
        keyword_overlap = len(intersection) / len(union) if union else 0.0
    else:
        keyword_overlap = 1.0  # both empty

    combined = (length_ratio + keyword_overlap) / 2.0
    passed = combined > OVERALL_PASS_THRESHOLD

    return {
        "length_ratio": round(length_ratio, 4),
        "keyword_overlap": round(keyword_overlap, 4),
        "combined": round(combined, 4),
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(prompts_path: str | None) -> list[str]:
    """Load general-knowledge prompts from JSONL file or use defaults.

    Expected JSONL format: {"prompt": "..."} per line.
    Returns up to 5 prompts.
    """
    if prompts_path and Path(prompts_path).exists():
        prompts = []
        with open(prompts_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if "prompt" in entry:
                        prompts.append(entry["prompt"])
                except json.JSONDecodeError:
                    continue
                if len(prompts) >= 5:
                    break
        if prompts:
            logger.info("Loaded %d prompts from %s", len(prompts), prompts_path)
            return prompts

    logger.info("Using %d hardcoded general-knowledge prompts", len(DEFAULT_PROMPTS))
    return DEFAULT_PROMPTS


# ---------------------------------------------------------------------------
# Forgetting check for one domain
# ---------------------------------------------------------------------------

def check_domain(
    domain: str,
    prompts: list[str],
    base_responses: list[str],
    model_path: str,
) -> dict:
    """Run forgetting check for a single domain adapter.

    Args:
        domain: domain name (e.g. "kicad-dsl")
        prompts: list of general-knowledge prompts
        base_responses: pre-computed base model responses (same order as prompts)
        model_path: path to base model

    Returns:
        Dict with domain, per-prompt results, aggregate metrics, and PASS/FAIL.
    """
    adapter_dir = ADAPTERS_DIR / f"stack-{domain}"
    adapter_path = adapter_dir / "adapters.safetensors"

    if not adapter_path.exists():
        # Try alternate name
        adapter_path = adapter_dir / "adapter_model.safetensors"

    if not adapter_path.exists():
        logger.warning("[%s] No adapter found at %s — SKIP", domain, adapter_dir)
        return {
            "domain": domain,
            "status": "SKIP",
            "reason": f"adapter not found at {adapter_dir}",
            "prompts": [],
        }

    logger.info("[%s] Loading base + adapter from %s", domain, adapter_dir)
    import mlx.core as mx
    model, tokenizer = _load_model(model_path, adapter_path=str(adapter_dir))

    prompt_results = []
    for i, (prompt, base_resp) in enumerate(zip(prompts, base_responses)):
        logger.info("[%s] Prompt %d/%d: %s", domain, i + 1, len(prompts), prompt[:60])
        adapted_resp = _generate(model, tokenizer, prompt)
        sim = compute_similarity(base_resp, adapted_resp)
        prompt_results.append({
            "prompt": prompt,
            "base_response_len": len(base_resp),
            "adapted_response_len": len(adapted_resp),
            **sim,
        })

    # Free memory
    del model, tokenizer
    mx.clear_memory_cache()

    # Aggregate
    if prompt_results:
        avg_combined = sum(r["combined"] for r in prompt_results) / len(prompt_results)
        all_passed = all(r["passed"] for r in prompt_results)
    else:
        avg_combined = 0.0
        all_passed = False

    domain_passed = all_passed and avg_combined > OVERALL_PASS_THRESHOLD
    status = "PASS" if domain_passed else "FAIL"

    logger.info(
        "[%s] %s — avg_similarity=%.3f, all_passed=%s",
        domain, status, avg_combined, all_passed,
    )

    return {
        "domain": domain,
        "status": status,
        "avg_combined_similarity": round(avg_combined, 4),
        "all_prompts_passed": all_passed,
        "prompts": prompt_results,
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def load_existing_results() -> dict:
    """Load previously saved incremental results."""
    if RESULTS_PATH.exists():
        try:
            return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"domains": {}, "base_responses": []}


def save_results(results: dict):
    """Save results incrementally."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Results saved to %s", RESULTS_PATH)


def main(argv: list[str] | None = None) -> int:
    global RESULTS_PATH
    parser = argparse.ArgumentParser(
        description="Forgetting check: verify LoRA adapters preserve general knowledge.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--domain", help="Single domain to check")
    group.add_argument("--all", action="store_true", help="Check all 10 domains")

    parser.add_argument(
        "--prompts", default=None,
        help="Path to JSONL with general-knowledge prompts (default: hardcoded)",
    )
    parser.add_argument(
        "--model", default=BASE_MODEL_PATH,
        help=f"Base model path (default: {BASE_MODEL_PATH})",
    )
    parser.add_argument(
        "--output", default=str(RESULTS_PATH),
        help=f"Output JSON path (default: {RESULTS_PATH})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from previous run (skip already-checked domains)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    RESULTS_PATH = Path(args.output)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    domains = ALL_DOMAINS if args.all else [args.domain]
    prompts = load_prompts(args.prompts)

    # Setup Metal
    _setup_metal()

    # Load or resume results
    if args.resume:
        results = load_existing_results()
    else:
        results = {"domains": {}, "base_responses": []}

    # Generate base model responses (or reuse from previous run)
    if results.get("base_responses") and len(results["base_responses"]) == len(prompts):
        logger.info("Reusing %d cached base responses", len(results["base_responses"]))
        base_responses = results["base_responses"]
    else:
        logger.info("Generating base model responses (%d prompts)...", len(prompts))
        import mlx.core as mx
        model, tokenizer = _load_model(args.model)
        base_responses = []
        for i, prompt in enumerate(prompts):
            logger.info("Base prompt %d/%d: %s", i + 1, len(prompts), prompt[:60])
            resp = _generate(model, tokenizer, prompt)
            base_responses.append(resp)
        del model, tokenizer
        mx.clear_memory_cache()

        results["base_responses"] = base_responses
        save_results(results)

    # Check each domain
    t0 = time.time()
    for domain in domains:
        if args.resume and domain in results["domains"]:
            logger.info("[%s] Already checked — skipping (--resume)", domain)
            continue

        domain_result = check_domain(domain, prompts, base_responses, args.model)
        results["domains"][domain] = domain_result
        save_results(results)  # incremental save after each domain

    elapsed = time.time() - t0

    # Summary
    total = len(results["domains"])
    passed = sum(1 for d in results["domains"].values() if d.get("status") == "PASS")
    skipped = sum(1 for d in results["domains"].values() if d.get("status") == "SKIP")
    failed = total - passed - skipped

    results["summary"] = {
        "total_domains": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "elapsed_seconds": round(elapsed, 1),
    }
    save_results(results)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Forgetting Check Summary")
    print(f"{'='*60}")
    for domain, result in results["domains"].items():
        status = result.get("status", "?")
        sim = result.get("avg_combined_similarity", 0)
        marker = "OK" if status == "PASS" else ("--" if status == "SKIP" else "XX")
        print(f"  [{marker}] {domain:20s}  similarity={sim:.3f}  {status}")
    print(f"{'='*60}")
    print(f"  PASS={passed}  FAIL={failed}  SKIP={skipped}  ({elapsed:.0f}s)")
    print(f"  Results: {RESULTS_PATH}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
