#!/usr/bin/env python3
"""Benchmark niche LoRA vs raw 35B — per-domain scoring via 480B judge.

Loads 50 eval prompts from data/prompts-expanded/<domain>.jsonl, stubs
inference for base/sft/dpo/grpo variants, and scores responses via HTTP POST
to the 480B judge at localhost:8481.

SCAFFOLD: inference calls are stubbed — judge scoring is real when reachable.

Usage:
    uv run python scripts/eval_niche_vs_base.py --domain kicad-dsl
    uv run python scripts/eval_niche_vs_base.py --all
    uv run python scripts/eval_niche_vs_base.py --all --compare sft,dpo,grpo
    uv run python scripts/eval_niche_vs_base.py --domain emc \\
        --judge-url http://localhost:8481 \\
        --output results/niche-vs-base.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------

ALL_DOMAINS: list[str] = [
    "kicad-dsl",
    "spice",
    "emc",
    "stm32",
    "embedded",
    "freecad",
    "platformio",
    "power",
    "dsp",
    "electronics",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default judge endpoint (480B teacher)
DEFAULT_JUDGE_URL = "http://localhost:8481"

# Max prompts to load per domain
MAX_PROMPTS = 50

# Judge request timeout in seconds
JUDGE_TIMEOUT = 30.0

# ---------------------------------------------------------------------------
# Judge scoring
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = (
    "Rate this response on technical accuracy (0-10): "
    "Domain: {domain} "
    "Question: {prompt} "
    "Response: {response} "
    "Return ONLY a number 0-10."
)


def _parse_score(text: str) -> float:
    """Extract a float 0-10 from judge output, normalise to 0.0-1.0."""
    text = text.strip()
    # Try to find a number (integer or decimal) in the response
    match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
    if match:
        raw = float(match.group(1))
        # Clamp to [0, 10] then normalise
        raw = max(0.0, min(10.0, raw))
        return round(raw / 10.0, 3)
    logger.debug("Could not parse score from judge output: %r", text)
    return 0.5  # fallback


def judge_score(
    domain: str,
    prompt: str,
    response: str,
    judge_url: str,
    client: httpx.Client,
) -> tuple[float, str]:
    """Call the 480B judge and return (normalised_score, status).

    Returns (0.5, 'fallback') when the judge is unavailable.
    """
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        domain=domain,
        prompt=prompt,
        response=response,
    )
    payload = {
        "model": "judge",
        "messages": [{"role": "user", "content": judge_prompt}],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    try:
        resp = client.post(
            f"{judge_url}/v1/chat/completions",
            json=payload,
            timeout=JUDGE_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        score = _parse_score(content)
        return score, "judge"
    except httpx.HTTPStatusError as exc:
        logger.warning("Judge HTTP error: %s — using fallback 0.5", exc)
        return 0.5, "fallback"
    except httpx.RequestError as exc:
        logger.debug("Judge unreachable (%s) — using fallback 0.5", exc)
        return 0.5, "fallback"
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning("Unexpected judge response format: %s — using fallback 0.5", exc)
        return 0.5, "fallback"


# ---------------------------------------------------------------------------
# Inference stubs
# ---------------------------------------------------------------------------


def _infer_stub(variant: str, domain: str, prompt: str) -> str:
    """Stub inference — replace with real HTTP call to vLLM / llama-server."""
    return (
        f"[STUB {variant}] This is a placeholder response for domain={domain}. "
        f"Prompt (first 80 chars): {prompt[:80]}"
    )


VARIANTS_ALL = ["base", "sft", "dpo", "grpo"]


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def load_prompts(domain: str) -> list[dict[str, Any]]:
    """Load up to MAX_PROMPTS prompts from data/prompts-expanded/<domain>.jsonl.

    Falls back to data/prompts/<domain>.jsonl if the expanded file is missing.
    """
    expanded_path = PROJECT_ROOT / "data" / "prompts-expanded" / f"{domain}.jsonl"
    fallback_path = PROJECT_ROOT / "data" / "prompts" / f"{domain}.jsonl"

    for path in (expanded_path, fallback_path):
        if path.exists():
            examples: list[dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    if len(examples) >= MAX_PROMPTS:
                        break
            logger.info("[%s] Loaded %d prompts from %s", domain, len(examples), path)
            return examples

    logger.warning("[%s] No prompt file found (checked expanded + fallback)", domain)
    return []


# ---------------------------------------------------------------------------
# Domain evaluation
# ---------------------------------------------------------------------------


def eval_domain(
    domain: str,
    variants: list[str],
    judge_url: str,
    client: httpx.Client,
) -> dict[str, Any]:
    """Evaluate one domain across requested variants.

    Returns a dict with keys: domain, prompts_count, variants (per-variant stats).
    """
    prompts = load_prompts(domain)
    if not prompts:
        logger.warning("[%s] No prompts — skipping", domain)
        return {
            "domain": domain,
            "prompts_count": 0,
            "variants": {v: {"avg_score": None, "status": "no_prompts"} for v in variants},
        }

    variant_scores: dict[str, list[float]] = {v: [] for v in variants}
    variant_statuses: dict[str, list[str]] = {v: [] for v in variants}

    for idx, example in enumerate(prompts):
        prompt_text = example.get("prompt", "")
        if not prompt_text:
            # Support messages format too
            messages = example.get("messages", [])
            user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
            prompt_text = user_msgs[0] if user_msgs else ""

        if not prompt_text:
            logger.debug("[%s] prompt #%d has no text, skipping", domain, idx)
            continue

        for variant in variants:
            response = _infer_stub(variant, domain, prompt_text)
            score, status = judge_score(domain, prompt_text, response, judge_url, client)
            variant_scores[variant].append(score)
            variant_statuses[variant].append(status)

    # Aggregate
    variant_results: dict[str, Any] = {}
    for variant in variants:
        scores = variant_scores[variant]
        statuses = variant_statuses[variant]
        if scores:
            avg = round(sum(scores) / len(scores), 4)
            fallback_pct = round(statuses.count("fallback") / len(statuses), 3)
        else:
            avg = None
            fallback_pct = 1.0
        variant_results[variant] = {
            "avg_score": avg,
            "n_prompts": len(scores),
            "fallback_pct": fallback_pct,
        }
        logger.info(
            "[%s] variant=%s  avg=%.3f  n=%d  fallback=%.0f%%",
            domain, variant,
            avg if avg is not None else float("nan"),
            len(scores),
            fallback_pct * 100,
        )

    return {
        "domain": domain,
        "prompts_count": len(prompts),
        "variants": variant_results,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(
    domains_results: list[dict[str, Any]],
    variants: list[str],
    judge_url: str,
) -> dict[str, Any]:
    """Assemble final JSON report."""
    summary: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "judge_url": judge_url,
        "mode": "scaffold_stub",
        "variants": variants,
        "domains_evaluated": len(domains_results),
    }

    # Best variant per domain
    ranking: list[dict[str, Any]] = []
    for dr in domains_results:
        best_variant = None
        best_score = -1.0
        for v in variants:
            score = dr["variants"].get(v, {}).get("avg_score")
            if score is not None and score > best_score:
                best_score = score
                best_variant = v
        ranking.append({
            "domain": dr["domain"],
            "best_variant": best_variant,
            "best_score": round(best_score, 4) if best_score >= 0 else None,
        })

    summary["ranking"] = ranking

    return {
        "summary": summary,
        "domains": {dr["domain"]: dr for dr in domains_results},
    }


def save_report(report: dict[str, Any], output: str) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Report saved → %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate base 35B vs 35B+LoRA per niche domain. "
            "Inference is stubbed; judge scoring is real when judge is reachable. "
            "Fallback to 0.5 when judge is unavailable."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available domains:\n  " + "\n  ".join(ALL_DOMAINS)
        ),
    )

    domain_group = parser.add_mutually_exclusive_group(required=True)
    domain_group.add_argument(
        "--domain",
        metavar="DOMAIN",
        help="Single domain to evaluate (e.g. kicad-dsl)",
    )
    domain_group.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all 10 niche domains",
    )

    parser.add_argument(
        "--compare",
        metavar="VARIANTS",
        default="base,sft,dpo,grpo",
        help=(
            "Comma-separated variants to compare "
            "(default: base,sft,dpo,grpo)"
        ),
    )
    parser.add_argument(
        "--judge-url",
        default=DEFAULT_JUDGE_URL,
        help=f"480B judge endpoint (default: {DEFAULT_JUDGE_URL})",
    )
    parser.add_argument(
        "--output",
        default="results/niche-vs-base.json",
        help="Output JSON path (default: results/niche-vs-base.json)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    variants = [v.strip() for v in args.compare.split(",") if v.strip()]
    unknown_variants = [v for v in variants if v not in VARIANTS_ALL]
    if unknown_variants:
        logger.warning(
            "Non-standard variants requested: %s (expected one of %s)",
            unknown_variants, VARIANTS_ALL,
        )

    domains = ALL_DOMAINS if args.all else [args.domain]
    if args.domain and args.domain not in ALL_DOMAINS:
        logger.warning(
            "Domain '%s' is not in the standard list. Proceeding anyway.", args.domain
        )

    logger.info(
        "Starting eval: domains=%s  variants=%s  judge=%s",
        "all" if args.all else domains,
        variants,
        args.judge_url,
    )
    logger.warning(
        "SCAFFOLD mode — inference is stubbed. "
        "Replace _infer_stub() with real model calls for production use."
    )

    results: list[dict[str, Any]] = []
    with httpx.Client() as client:
        for domain in domains:
            dr = eval_domain(domain, variants, args.judge_url, client)
            results.append(dr)

    report = build_report(results, variants, args.judge_url)
    save_report(report, args.output)

    summary = report["summary"]
    logger.info(
        "Done: %d domains evaluated, results → %s",
        summary["domains_evaluated"],
        args.output,
    )


if __name__ == "__main__":
    main()
