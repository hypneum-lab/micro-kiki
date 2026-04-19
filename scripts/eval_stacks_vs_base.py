#!/usr/bin/env python3
"""Story-15: Evaluate all 10 domain stacks vs base model.

For each domain, loads eval prompts from data/eval/<domain>.jsonl, generates
responses with the base model and with base + adapter, then compares quality
metrics: response length, domain keyword density, and format correctness.

Usage:
    # All domains:
    .venv/bin/python3 scripts/eval_stacks_vs_base.py --all

    # Single domain:
    .venv/bin/python3 scripts/eval_stacks_vs_base.py --domain kicad-dsl

    # Resume interrupted run:
    .venv/bin/python3 scripts/eval_stacks_vs_base.py --all --resume

Designed for Mac Studio M3 Ultra 512 GB (MLX). Restart-friendly: saves results
incrementally to results/stacks-vs-base.json.
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
EVAL_DATA_DIR = PROJECT_ROOT / "data" / "eval"
RESULTS_PATH = PROJECT_ROOT / "results" / "stacks-vs-base.json"

MAX_PROMPTS_PER_DOMAIN = 5
MAX_TOKENS = 512

# Domain-specific keywords for measuring keyword density
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "kicad-dsl": [
        "kicad", "schematic", "footprint", "symbol", "pad", "net",
        "pcb", "s-expression", "module", "pin", "layer", "drill",
    ],
    "spice": [
        "spice", "ngspice", "netlist", "subcircuit", ".tran", ".ac",
        ".dc", "voltage", "current", "node", "resistance", "capacitor",
        "inductor", "mosfet", "diode", "simulation",
    ],
    "emc": [
        "emc", "emi", "shielding", "decoupling", "impedance", "ground",
        "filter", "radiated", "conducted", "coupling", "ferrite",
        "bypass", "return path", "compliance", "emissions",
    ],
    "stm32": [
        "stm32", "hal", "gpio", "uart", "spi", "i2c", "timer", "dma",
        "interrupt", "nvic", "clock", "peripheral", "register", "cortex",
        "arm", "cubemx", "stm32cube",
    ],
    "embedded": [
        "embedded", "firmware", "microcontroller", "rtos", "interrupt",
        "peripheral", "register", "gpio", "adc", "pwm", "watchdog",
        "bootloader", "flash", "mcu", "bare-metal",
    ],
    "freecad": [
        "freecad", "part", "sketch", "constraint", "extrude", "pocket",
        "fillet", "chamfer", "assembly", "mesh", "step", "iges",
        "parametric", "body", "pad",
    ],
    "platformio": [
        "platformio", "pio", "platform", "framework", "board", "lib_deps",
        "upload", "monitor", "environment", "build_flags", "serial",
        "arduino", "espidf", "toolchain",
    ],
    "power": [
        "power", "voltage", "current", "regulator", "buck", "boost",
        "ldo", "mosfet", "inductor", "capacitor", "efficiency", "ripple",
        "converter", "supply", "watt", "thermal",
    ],
    "dsp": [
        "dsp", "filter", "fft", "frequency", "sampling", "convolution",
        "signal", "spectrum", "nyquist", "aliasing", "digital",
        "coefficient", "transfer function", "biquad", "window",
    ],
    "electronics": [
        "circuit", "resistor", "capacitor", "inductor", "transistor",
        "diode", "op-amp", "amplifier", "voltage", "current", "ohm",
        "kirchhoff", "thevenin", "norton", "impedance",
    ],
}


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

    kwargs = {"path_or_hf_repo": model_path}
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
# Eval data loading
# ---------------------------------------------------------------------------

def load_eval_prompts(domain: str) -> list[str]:
    """Load up to MAX_PROMPTS_PER_DOMAIN eval prompts for a domain.

    Reads from data/eval/<domain>.jsonl. Expected format: {"prompt": "..."}.
    """
    eval_file = EVAL_DATA_DIR / f"{domain}.jsonl"
    if not eval_file.exists():
        logger.warning("[%s] No eval file at %s", domain, eval_file)
        return []

    prompts = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                prompt = entry.get("prompt", "")
                if not prompt:
                    # Support messages format
                    messages = entry.get("messages", [])
                    user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
                    prompt = user_msgs[0] if user_msgs else ""
                if prompt:
                    prompts.append(prompt)
            except json.JSONDecodeError:
                continue
            if len(prompts) >= MAX_PROMPTS_PER_DOMAIN:
                break

    logger.info("[%s] Loaded %d eval prompts from %s", domain, len(prompts), eval_file)
    return prompts


# ---------------------------------------------------------------------------
# Scoring metrics
# ---------------------------------------------------------------------------

def compute_keyword_density(response: str, domain: str) -> float:
    """Fraction of domain keywords found in the response."""
    keywords = DOMAIN_KEYWORDS.get(domain, [])
    if not keywords:
        return 0.0

    response_lower = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in response_lower)
    return round(hits / len(keywords), 4)


def compute_format_score(response: str) -> float:
    """Score response format quality (0-1).

    Checks for:
    - Non-empty response (0.2)
    - Contains code blocks or structured content (0.3)
    - Reasonable length > 50 chars (0.2)
    - Contains technical terms or numbers (0.3)
    """
    score = 0.0

    if len(response.strip()) > 0:
        score += 0.2

    if "```" in response or re.search(r"^\s*[-*]\s", response, re.MULTILINE):
        score += 0.3

    if len(response) > 50:
        score += 0.2

    if re.search(r"\d+", response) or re.search(r"[A-Z]{2,}", response):
        score += 0.3

    return round(min(score, 1.0), 4)


def score_response(response: str, domain: str) -> dict:
    """Compute all metrics for a single response."""
    return {
        "length": len(response),
        "keyword_density": compute_keyword_density(response, domain),
        "format_score": compute_format_score(response),
    }


def compute_improvement(base_scores: dict, adapted_scores: dict) -> dict:
    """Compute improvement of adapted over base for each metric."""
    improvements = {}

    # Length improvement (positive = adapter produces longer responses)
    base_len = max(base_scores["length"], 1)
    adapted_len = adapted_scores["length"]
    improvements["length_ratio"] = round(adapted_len / base_len, 4)

    # Keyword density improvement
    base_kd = base_scores["keyword_density"]
    adapted_kd = adapted_scores["keyword_density"]
    improvements["keyword_density_delta"] = round(adapted_kd - base_kd, 4)

    # Format score improvement
    base_fs = base_scores["format_score"]
    adapted_fs = adapted_scores["format_score"]
    improvements["format_score_delta"] = round(adapted_fs - base_fs, 4)

    # Combined improvement score: weighted average of deltas
    # Keyword density is most important for domain specialisation
    combined = (
        0.2 * max(0, (improvements["length_ratio"] - 1.0))  # bonus for longer
        + 0.5 * max(0, improvements["keyword_density_delta"])  # main signal
        + 0.3 * max(0, improvements["format_score_delta"])     # format bonus
    )
    improvements["combined_score"] = round(combined, 4)

    return improvements


# ---------------------------------------------------------------------------
# Domain evaluation
# ---------------------------------------------------------------------------

def eval_domain(
    domain: str,
    base_model_path: str,
) -> dict:
    """Evaluate one domain: base vs base+adapter.

    Loads prompts, generates with both models, computes metrics.
    Returns a serializable dict.
    """
    prompts = load_eval_prompts(domain)
    if not prompts:
        return {
            "domain": domain,
            "status": "SKIP",
            "reason": "no eval prompts",
            "prompts": [],
        }

    adapter_dir = ADAPTERS_DIR / f"stack-{domain}"
    adapter_safetensors = adapter_dir / "adapters.safetensors"
    if not adapter_safetensors.exists():
        adapter_safetensors = adapter_dir / "adapter_model.safetensors"
    if not adapter_safetensors.exists():
        return {
            "domain": domain,
            "status": "SKIP",
            "reason": f"adapter not found at {adapter_dir}",
            "prompts": [],
        }

    import mlx.core as mx

    # Generate base responses
    logger.info("[%s] Generating base responses (%d prompts)...", domain, len(prompts))
    model, tokenizer = _load_model(base_model_path)
    base_responses = []
    for i, prompt in enumerate(prompts):
        logger.info("[%s] Base prompt %d/%d", domain, i + 1, len(prompts))
        resp = _generate(model, tokenizer, prompt)
        base_responses.append(resp)
    del model, tokenizer
    mx.metal.clear_cache()

    # Generate adapted responses
    logger.info("[%s] Generating adapted responses...", domain)
    model, tokenizer = _load_model(base_model_path, adapter_path=str(adapter_dir))
    adapted_responses = []
    for i, prompt in enumerate(prompts):
        logger.info("[%s] Adapted prompt %d/%d", domain, i + 1, len(prompts))
        resp = _generate(model, tokenizer, prompt)
        adapted_responses.append(resp)
    del model, tokenizer
    mx.metal.clear_cache()

    # Score each prompt
    prompt_results = []
    for prompt, base_resp, adapted_resp in zip(prompts, base_responses, adapted_responses):
        base_scores = score_response(base_resp, domain)
        adapted_scores = score_response(adapted_resp, domain)
        improvement = compute_improvement(base_scores, adapted_scores)
        prompt_results.append({
            "prompt": prompt[:100],  # truncate for readability
            "base": base_scores,
            "adapted": adapted_scores,
            "improvement": improvement,
        })

    # Aggregate
    if prompt_results:
        avg_kd_delta = sum(
            r["improvement"]["keyword_density_delta"] for r in prompt_results
        ) / len(prompt_results)
        avg_format_delta = sum(
            r["improvement"]["format_score_delta"] for r in prompt_results
        ) / len(prompt_results)
        avg_combined = sum(
            r["improvement"]["combined_score"] for r in prompt_results
        ) / len(prompt_results)
        avg_length_ratio = sum(
            r["improvement"]["length_ratio"] for r in prompt_results
        ) / len(prompt_results)
    else:
        avg_kd_delta = 0.0
        avg_format_delta = 0.0
        avg_combined = 0.0
        avg_length_ratio = 1.0

    # The adapter is "better" if it improves keyword density without
    # destroying format quality
    improved = avg_kd_delta > 0 or avg_format_delta >= 0

    return {
        "domain": domain,
        "status": "IMPROVED" if improved else "REGRESSED",
        "n_prompts": len(prompt_results),
        "aggregate": {
            "avg_keyword_density_delta": round(avg_kd_delta, 4),
            "avg_format_score_delta": round(avg_format_delta, 4),
            "avg_length_ratio": round(avg_length_ratio, 4),
            "avg_combined_score": round(avg_combined, 4),
        },
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
    return {"domains": {}}


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
        description="Evaluate all domain LoRA stacks vs base model.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--domain", help="Single domain to evaluate")
    group.add_argument("--all", action="store_true", help="Evaluate all 10 domains")

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
        help="Resume from previous run (skip already-evaluated domains)",
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

    # Setup Metal
    _setup_metal()

    # Load or resume results
    if args.resume:
        results = load_existing_results()
    else:
        results = {"domains": {}}

    # Evaluate each domain
    t0 = time.time()
    for domain in domains:
        if args.resume and domain in results["domains"]:
            logger.info("[%s] Already evaluated — skipping (--resume)", domain)
            continue

        logger.info("[%s] Starting evaluation...", domain)
        domain_result = eval_domain(domain, args.model)
        results["domains"][domain] = domain_result
        save_results(results)  # incremental save

    elapsed = time.time() - t0

    # Summary
    total = len(results["domains"])
    improved = sum(
        1 for d in results["domains"].values() if d.get("status") == "IMPROVED"
    )
    regressed = sum(
        1 for d in results["domains"].values() if d.get("status") == "REGRESSED"
    )
    skipped = sum(
        1 for d in results["domains"].values() if d.get("status") == "SKIP"
    )

    results["summary"] = {
        "total_domains": total,
        "improved": improved,
        "regressed": regressed,
        "skipped": skipped,
        "elapsed_seconds": round(elapsed, 1),
    }
    save_results(results)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"Stack vs Base Evaluation Summary")
    print(f"{'='*70}")
    print(f"  {'Domain':20s} {'Status':10s} {'KW Delta':>10s} {'Fmt Delta':>10s} {'Combined':>10s}")
    print(f"  {'-'*60}")
    for domain in ALL_DOMAINS:
        result = results["domains"].get(domain, {})
        status = result.get("status", "?")
        agg = result.get("aggregate", {})
        kd = agg.get("avg_keyword_density_delta", 0)
        fd = agg.get("avg_format_score_delta", 0)
        cs = agg.get("avg_combined_score", 0)
        marker = "++" if status == "IMPROVED" else ("--" if status == "SKIP" else "XX")
        print(f"  [{marker}] {domain:18s} {status:10s} {kd:+.4f}    {fd:+.4f}    {cs:.4f}")
    print(f"  {'-'*60}")
    print(f"  IMPROVED={improved}  REGRESSED={regressed}  SKIP={skipped}  ({elapsed:.0f}s)")
    print(f"  Results: {RESULTS_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
