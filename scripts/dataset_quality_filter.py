#!/usr/bin/env python3
"""Quality filter for generated training data.

Scores each example on length, domain keyword presence, code blocks,
and absence of refusal patterns. Deduplicates by MD5.

Usage:
    python3 scripts/dataset_quality_filter.py --domain kicad-dsl
    python3 scripts/dataset_quality_filter.py --all --threshold 0.35
    python3 scripts/dataset_quality_filter.py --all --sources mcp-generated,codex-generated
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "data" / "filtered"

ALL_DOMAINS = [
    "kicad-dsl", "spice", "emc", "stm32", "embedded",
    "power", "dsp", "electronics", "freecad", "platformio",
]

DEFAULT_SOURCES = ["mcp-generated", "codex-generated", "distilled-480b"]
DEFAULT_THRESHOLD = 0.3

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "kicad-dsl": ["symbol", "module", "fp_", "pad", "kicad", "schematic", "footprint"],
    "spice": [".model", ".subckt", ".tran", ".ac", "ngspice", "netlist", "MOSFET",
              "simulation", "transient", "AC analysis", "DC sweep", "Monte Carlo",
              "convergence", "Bode", "waveform", "parametric", "step response"],
    "emc": ["CISPR", "EMI", "shielding", "grounding", "filter", "ferrite"],
    "stm32": ["HAL_", "STM32", "GPIO", "UART", "SPI", "DMA", "TIM", "CubeMX"],
    "embedded": ["interrupt", "ISR", "RTOS", "FreeRTOS", "DMA", "firmware", "peripheral"],
    "power": ["buck", "boost", "MOSFET", "inductor", "switching", "regulator"],
    "dsp": ["FFT", "FIR", "IIR", "filter", "spectrum", "convolution"],
    "electronics": ["op-amp", "amplifier", "transistor", "bias", "gain", "bandwidth"],
    "freecad": ["FreeCAD", "Part", "Sketch", "macro", "parametric"],
    "platformio": ["platformio", "pio", "board", "framework", "lib_deps"],
}

REFUSAL_MARKERS = ["I cannot", "As an AI", "I'm not able to", "I am not able to"]


def _get_response_text(example: dict) -> str:
    """Extract assistant response text from an example."""
    messages = example.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    # Fallback: plain text key
    return example.get("response", example.get("output", ""))


def score_example(example: dict, domain: str) -> float:
    """Score an example on a 0.0–1.0 scale.

    Breakdown:
    - Length score:    up to 0.4
    - Keyword score:   up to 0.3
    - Code block:      +0.2
    - Not refusal:     +0.1
    """
    text = _get_response_text(example)
    length = len(text)

    # --- Length score (0.0–0.4) ---
    if length < 50:
        length_score = 0.0
    elif length < 200:
        # Linear ramp 50→200: 0.0→0.3
        length_score = 0.3 * (length - 50) / 150
    elif length <= 2000:
        length_score = 0.3
    elif length <= 5000:
        # Slight bonus for richer answers up to 5000 chars
        length_score = 0.3 + 0.1 * (length - 2000) / 3000
    else:
        # Very long: small penalty to discourage bloat
        length_score = 0.25

    # --- Domain keyword score (0.0–0.3) ---
    keywords = DOMAIN_KEYWORDS.get(domain, [])
    if keywords:
        hits = sum(1 for kw in keywords if kw in text)
        keyword_score = min(0.3, 0.3 * hits / len(keywords))
    else:
        keyword_score = 0.0

    # --- Code block bonus ---
    code_score = 0.2 if "```" in text else 0.0

    # --- Non-refusal bonus ---
    refusal_score = 0.0 if any(m in text for m in REFUSAL_MARKERS) else 0.1

    total = length_score + keyword_score + code_score + refusal_score
    return round(min(1.0, max(0.0, total)), 4)


def _md5(example: dict) -> str:
    text = _get_response_text(example)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def filter_dataset(
    examples: list[dict],
    domain: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict]:
    """Deduplicate by MD5 and keep examples above score threshold."""
    seen_hashes: set[str] = set()
    result: list[dict] = []

    for ex in examples:
        digest = _md5(ex)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)

        score = score_example(ex, domain)
        if score >= threshold:
            result.append(ex)

    return result


def load_examples(domain: str, sources: list[str]) -> list[dict]:
    """Load all examples for a domain from the given source directories."""
    examples: list[dict] = []
    for source in sources:
        src_file = REPO_ROOT / "data" / source / domain / "train.jsonl"
        if not src_file.exists():
            logger.debug("Source not found: %s", src_file)
            continue
        with open(src_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning("Skipping malformed line in %s: %s", src_file, e)
        logger.info("Loaded %s / %s", source, domain)
    return examples


def process_domain(domain: str, args: argparse.Namespace) -> None:
    sources = [s.strip() for s in args.sources.split(",")]
    examples = load_examples(domain, sources)

    if not examples:
        logger.warning("No examples found for domain '%s'", domain)
        return

    logger.info(
        "%s: %d raw examples from %d source(s)", domain, len(examples), len(sources)
    )

    filtered = filter_dataset(examples, domain, threshold=args.threshold)
    kept_pct = 100 * len(filtered) / len(examples) if examples else 0
    logger.info(
        "%s: %d → %d examples kept (%.1f%%, threshold=%.2f)",
        domain,
        len(examples),
        len(filtered),
        kept_pct,
        args.threshold,
    )

    if args.dry_run:
        return

    out_dir = OUTPUT_ROOT / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "train.jsonl"

    with open(out_file, "w") as f:
        for ex in filtered:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info("Written %d examples → %s", len(filtered), out_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quality filter for generated training data.",
    )
    parser.add_argument("--domain", help="Single domain to filter")
    parser.add_argument("--all", action="store_true", help="Filter all domains")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum score to keep an example (default 0.3)",
    )
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help="Comma-separated list of source directories under data/ "
        f"(default: {','.join(DEFAULT_SOURCES)})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show stats without writing output files",
    )
    args = parser.parse_args()

    if args.domain:
        domains = [args.domain]
    elif args.all:
        domains = ALL_DOMAINS
    else:
        parser.print_help()
        return

    for domain in domains:
        process_domain(domain, args)


if __name__ == "__main__":
    main()
