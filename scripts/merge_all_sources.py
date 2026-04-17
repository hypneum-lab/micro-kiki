#!/usr/bin/env python3
"""Unified merge — all 5 data sources into final training set.

Merges ALL data sources per domain into data/final/<domain>/train.jsonl,
deduplicating across sources by MD5 hash of the messages content.

Priority order (highest to lowest):
  1. data/filtered/<domain>/train.jsonl   (Claude+Codex quality-filtered)
  2. data/distilled-480b/<domain>/train.jsonl (480B teacher)
  3. data/merged/<domain>/train.jsonl     (KIKI + HF mascarade)

Usage:
    uv run python scripts/merge_all_sources.py --domain kicad-dsl
    uv run python scripts/merge_all_sources.py --all
    uv run python scripts/merge_all_sources.py --all --stats
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Iterator

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

# ---------------------------------------------------------------------------
# Source layout — ordered by priority (highest first)
# ---------------------------------------------------------------------------

SOURCES: list[tuple[str, str]] = [
    ("claude-cli",     "data/mcp-generated/{domain}/train.jsonl"),
    ("codex-cli",      "data/codex-generated/{domain}/train.jsonl"),
    ("filtered",       "data/filtered/{domain}/train.jsonl"),
    ("distilled-480b", "data/distilled-480b/{domain}/train.jsonl"),
    ("merged",         "data/merged/{domain}/train.jsonl"),
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _messages_hash(example: dict) -> str:
    """Return MD5 hex-digest of the serialised messages list."""
    messages = example.get("messages", [])
    canonical = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def _iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield parsed JSON objects from a .jsonl file, skipping blank lines."""
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d in %s: %s", lineno, path, exc)


# ---------------------------------------------------------------------------
# Core merge logic
# ---------------------------------------------------------------------------


def merge_domain(
    domain: str,
    show_stats: bool = False,
) -> dict[str, int]:
    """Merge all sources for one domain, write to data/final/<domain>/train.jsonl.

    Returns a dict of per-source counts (including 'total' and 'deduped').
    """
    seen_hashes: set[str] = set()
    output_examples: list[dict] = []
    per_source_counts: dict[str, int] = {}

    for source_name, path_template in SOURCES:
        source_path = PROJECT_ROOT / path_template.format(domain=domain)
        if not source_path.exists():
            logger.debug("Source %s not found for domain %s, skipping: %s",
                         source_name, domain, source_path)
            per_source_counts[source_name] = 0
            continue

        count = 0
        added = 0
        for example in _iter_jsonl(source_path):
            count += 1
            h = _messages_hash(example)
            if h not in seen_hashes:
                seen_hashes.add(h)
                output_examples.append(example)
                added += 1

        per_source_counts[source_name] = added
        logger.info(
            "[%s] %s: read=%d  new=%d  duplicates=%d",
            domain, source_name, count, added, count - added,
        )

    # Write output
    out_dir = PROJECT_ROOT / "data" / "final" / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.jsonl"

    with out_path.open("w", encoding="utf-8") as fh:
        for ex in output_examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    total = sum(per_source_counts.values())
    per_source_counts["total"] = total
    per_source_counts["deduped"] = len(output_examples)

    logger.info(
        "[%s] final: %d examples (deduped from %d across all sources) → %s",
        domain, len(output_examples), total, out_path,
    )

    if show_stats:
        _print_stats(domain, per_source_counts)

    return per_source_counts


def _print_stats(domain: str, counts: dict[str, int]) -> None:
    """Pretty-print per-source stats to stdout."""
    print(f"\n{'─' * 50}")
    print(f"  Domain: {domain}")
    print(f"{'─' * 50}")
    for source_name, _ in SOURCES:
        n = counts.get(source_name, 0)
        print(f"  {source_name:<20} {n:>6} examples added")
    print(f"  {'total (raw)':<20} {counts.get('total', 0):>6}")
    print(f"  {'final (deduped)':<20} {counts.get('deduped', 0):>6}")
    print(f"{'─' * 50}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge all 5 data sources per domain into data/final/<domain>/train.jsonl. "
            "Deduplicates across sources by MD5 hash of messages content. "
            "Priority: filtered > distilled-480b > merged."
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
        help="Single domain to merge (e.g. kicad-dsl)",
    )
    domain_group.add_argument(
        "--all",
        action="store_true",
        help="Merge all 10 domains",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print per-source counts after merging",
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

    domains = ALL_DOMAINS if args.all else [args.domain]

    if args.domain and args.domain not in ALL_DOMAINS:
        logger.warning(
            "Domain '%s' is not in the standard list. Proceeding anyway.", args.domain
        )

    grand_total = 0
    grand_deduped = 0

    for domain in domains:
        counts = merge_domain(domain, show_stats=args.stats)
        grand_total += counts.get("total", 0)
        grand_deduped += counts.get("deduped", 0)

    if len(domains) > 1:
        logger.info(
            "All domains done: %d raw examples → %d after dedup (%d duplicates removed)",
            grand_total, grand_deduped, grand_total - grand_deduped,
        )
        if args.stats:
            print(f"\n{'=' * 50}")
            print(f"  Grand total: {grand_total} raw → {grand_deduped} deduped")
            print(f"  ({grand_total - grand_deduped} duplicates removed across all domains)")
            print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
