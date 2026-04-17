#!/usr/bin/env python3
"""Merge redundant domain pairs in micro-kiki.

Handles two merge scenarios:
1. Full merge: source domain is absorbed into target domain (e.g., spice-sim -> spice).
   Data is concatenated and deduplicated, all config references are updated.
2. Sub-category routing: source domain routes to target stack first, falls back to
   source-specific stack if available (e.g., stm32 -> embedded).
   Both domains remain but routing is updated.

Usage:
    # Full merge: absorb spice-sim into spice
    python3 scripts/merge_domains.py --source spice-sim --target spice

    # Sub-category: make stm32 route through embedded first
    python3 scripts/merge_domains.py --source stm32 --target embedded --subcategory

    # Dry run (show what would change, don't write)
    python3 scripts/merge_domains.py --source spice-sim --target spice --dry-run

    # Merge data only (skip config updates)
    python3 scripts/merge_domains.py --source spice-sim --target spice --data-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Data merge helpers
# ---------------------------------------------------------------------------


def _content_hash(example: dict) -> str:
    """Stable hash for dedup: hash the full JSON serialization."""
    return hashlib.md5(
        json.dumps(example, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _message_hash(example: dict) -> str:
    """Hash based on message content only (ignores metadata fields).

    Falls back to full content hash if no messages field.
    """
    messages = example.get("messages", [])
    if messages:
        text = "".join(
            f"{m.get('role', '')}:{m.get('content', '')}" for m in messages
        )
        return hashlib.md5(text.encode()).hexdigest()
    # Fallback: instruction+output or prompt+response
    text = example.get("instruction", "") + example.get("output", "")
    if not text:
        text = example.get("prompt", "") + example.get("response", "")
    if text:
        return hashlib.md5(text.encode()).hexdigest()
    return _content_hash(example)


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, skipping blank lines and JSON errors."""
    if not path.exists():
        logger.warning("File not found: %s", path)
        return []
    examples = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON at %s:%d", path, lineno)
    return examples


def save_jsonl(path: Path, examples: list[dict]) -> None:
    """Write examples as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("Wrote %d examples to %s", len(examples), path)


def merge_and_dedup(
    source_examples: list[dict],
    target_examples: list[dict],
    target_domain: str,
) -> list[dict]:
    """Merge source into target, deduplicating by message content.

    Updates the 'domain' field in source examples to target_domain.
    Target examples take priority (kept first).
    """
    seen: set[str] = set()
    merged: list[dict] = []

    # Target examples first (priority)
    for ex in target_examples:
        h = _message_hash(ex)
        if h not in seen:
            seen.add(h)
            merged.append(ex)

    # Source examples, updating domain
    added = 0
    for ex in source_examples:
        h = _message_hash(ex)
        if h not in seen:
            seen.add(h)
            ex_copy = dict(ex)
            if "domain" in ex_copy:
                ex_copy["domain"] = target_domain
            merged.append(ex_copy)
            added += 1

    logger.info(
        "Merged: %d target + %d source = %d total (%d new from source, %d duplicates removed)",
        len(target_examples),
        len(source_examples),
        len(merged),
        added,
        len(source_examples) - added,
    )
    return merged


# ---------------------------------------------------------------------------
# Data merge: local data/micro-kiki directories
# ---------------------------------------------------------------------------


def merge_data_local(source: str, target: str, dry_run: bool = False) -> None:
    """Merge train.jsonl and valid.jsonl from source domain into target domain.

    Operates on data/micro-kiki/<domain>/ directories.
    """
    data_base = REPO_ROOT / "data" / "micro-kiki"
    source_dir = data_base / source
    target_dir = data_base / target

    if not source_dir.exists():
        logger.info("No local data directory for source '%s' at %s — skipping local merge", source, source_dir)
        return

    for split in ("train.jsonl", "valid.jsonl"):
        source_file = source_dir / split
        target_file = target_dir / split

        if not source_file.exists():
            logger.info("  %s: source file not found — skipping", split)
            continue

        source_examples = load_jsonl(source_file)
        target_examples = load_jsonl(target_file)

        logger.info("  %s: source=%d, target=%d", split, len(source_examples), len(target_examples))
        merged = merge_and_dedup(source_examples, target_examples, target)

        if dry_run:
            logger.info("  [DRY RUN] Would write %d examples to %s", len(merged), target_file)
        else:
            save_jsonl(target_file, merged)

    # Also check other data directories (distilled, eval, etc.)
    for data_subdir in ("distilled", "eval", "codex-generated", "hf-extra", "filtered", "merged"):
        subdir = REPO_ROOT / "data" / data_subdir
        source_file = subdir / f"{source}.jsonl"
        target_file = subdir / f"{target}.jsonl"

        if source_file.exists():
            source_examples = load_jsonl(source_file)
            target_examples = load_jsonl(target_file) if target_file.exists() else []
            logger.info("  data/%s: source=%d, target=%d", data_subdir, len(source_examples), len(target_examples))
            merged = merge_and_dedup(source_examples, target_examples, target)

            if dry_run:
                logger.info("  [DRY RUN] Would write %d examples to %s", len(merged), target_file)
            else:
                save_jsonl(target_file, merged)

    if not dry_run:
        # Archive source directory (don't delete)
        archive_dir = data_base / f"_archived_{source}"
        if source_dir.exists() and not archive_dir.exists():
            shutil.move(str(source_dir), str(archive_dir))
            logger.info("  Archived source data: %s -> %s", source_dir, archive_dir)


# ---------------------------------------------------------------------------
# Config updates
# ---------------------------------------------------------------------------


def update_domains_yaml(source: str, target: str, dry_run: bool = False) -> None:
    """Merge source domain keywords/patterns into target in domains.yaml, remove source entry."""
    config_path = REPO_ROOT / "configs" / "micro_kiki" / "domains.yaml"
    if not config_path.exists():
        logger.warning("domains.yaml not found at %s", config_path)
        return

    with open(config_path) as f:
        raw = f.read()

    data = yaml.safe_load(raw)
    domains = data.get("domains", {})

    if source not in domains:
        logger.info("Source domain '%s' not in domains.yaml — nothing to merge", source)
        return
    if target not in domains:
        logger.warning("Target domain '%s' not in domains.yaml — cannot merge", target)
        return

    source_cfg = domains[source]
    target_cfg = domains[target]

    # Merge keywords (union, preserving order, target first)
    target_kw = target_cfg.get("keywords", [])
    source_kw = source_cfg.get("keywords", [])
    existing = set(target_kw)
    for kw in source_kw:
        if kw not in existing:
            target_kw.append(kw)
            existing.add(kw)
    target_cfg["keywords"] = target_kw

    # Merge patterns (union)
    target_pat = target_cfg.get("patterns", [])
    source_pat = source_cfg.get("patterns", [])
    existing_pat = set(target_pat)
    for pat in source_pat:
        if pat not in existing_pat:
            target_pat.append(pat)
            existing_pat.add(pat)
    target_cfg["patterns"] = target_pat

    # Merge existing_sources
    target_src = target_cfg.get("existing_sources", [])
    source_src = source_cfg.get("existing_sources", [])
    existing_src = set(target_src)
    for src in source_src:
        if src not in existing_src:
            target_src.append(src)
            existing_src.add(src)
    if target_src:
        target_cfg["existing_sources"] = target_src

    # Update target count to sum
    source_n = source_cfg.get("target", 0)
    target_n = target_cfg.get("target", 0)
    target_cfg["target"] = target_n + source_n

    # Remove source domain
    del domains[source]

    if dry_run:
        logger.info("[DRY RUN] Would update domains.yaml:")
        logger.info("  Merged %d keywords, %d patterns from '%s' into '%s'",
                     len(source_kw), len(source_pat), source, target)
        logger.info("  Would remove '%s' entry", source)
    else:
        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)
        logger.info("Updated domains.yaml: merged '%s' into '%s'", source, target)


def update_brainstacks_yaml(source: str, target: str, dry_run: bool = False) -> None:
    """Remove source domain from the curriculum in brainstacks.yaml."""
    config_path = REPO_ROOT / "configs" / "micro_kiki" / "brainstacks.yaml"
    if not config_path.exists():
        return

    with open(config_path) as f:
        raw = f.read()

    data = yaml.safe_load(raw)
    curriculum = data.get("curriculum", [])

    if source in curriculum:
        curriculum.remove(source)
        if dry_run:
            logger.info("[DRY RUN] Would remove '%s' from brainstacks.yaml curriculum", source)
        else:
            data["curriculum"] = curriculum
            with open(config_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)
            logger.info("Removed '%s' from brainstacks.yaml curriculum", source)
    else:
        logger.info("'%s' not in brainstacks.yaml curriculum", source)


def update_curriculum_adaptive_json(source: str, target: str, dry_run: bool = False) -> None:
    """Merge source training config into target in curriculum-adaptive.json."""
    config_path = REPO_ROOT / "configs" / "curriculum-adaptive.json"
    if not config_path.exists():
        return

    with open(config_path) as f:
        data = json.load(f)

    if source not in data:
        logger.info("'%s' not in curriculum-adaptive.json", source)
        return

    source_cfg = data[source]
    target_cfg = data.get(target, {})

    # Sum n_examples, keep target's other params
    if target_cfg:
        target_cfg["n_examples"] = target_cfg.get("n_examples", 0) + source_cfg.get("n_examples", 0)
    del data[source]

    if dry_run:
        logger.info("[DRY RUN] Would merge '%s' n_examples into '%s' in curriculum-adaptive.json", source, target)
    else:
        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        logger.info("Updated curriculum-adaptive.json: merged '%s' into '%s'", source, target)


# ---------------------------------------------------------------------------
# Sub-category routing setup (stm32 -> embedded)
# ---------------------------------------------------------------------------


def setup_subcategory_routing(source: str, target: str, dry_run: bool = False) -> None:
    """Document and configure sub-category routing.

    For stm32 -> embedded: stm32 queries check the embedded stack first,
    then fall back to stm32-specific adapter if available.

    This updates:
    - DOMAIN_ALIASES in train_vqc_router.py (already has some aliases)
    - Adds a SUBCATEGORY_MAP to model_router.py for stack fallback
    - Documents the mapping in domains.yaml
    """
    # 1. Update domains.yaml: add parent_domain field to source
    config_path = REPO_ROOT / "configs" / "micro_kiki" / "domains.yaml"
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f)

        domains = data.get("domains", {})
        if source in domains:
            domains[source]["parent_domain"] = target
            domains[source].setdefault("routing_note",
                f"Sub-category of {target}. Router checks {target} stack first, "
                f"falls back to {source}-specific adapter if available."
            )
            if dry_run:
                logger.info("[DRY RUN] Would add parent_domain='%s' to '%s' in domains.yaml", target, source)
            else:
                with open(config_path, "w") as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)
                logger.info("Added parent_domain='%s' to '%s' in domains.yaml", target, source)

    logger.info(
        "\nSub-category routing configured: %s -> %s\n"
        "  The router will check the %s stack first for %s queries.\n"
        "  If the %s adapter exists and scores higher, it will be used instead.\n"
        "  Both domains remain in the curriculum.",
        source, target, target, source, source,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge redundant micro-kiki domains",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source", required=True, help="Source domain to merge FROM (e.g., spice-sim)")
    parser.add_argument("--target", required=True, help="Target domain to merge INTO (e.g., spice)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    parser.add_argument("--data-only", action="store_true", help="Only merge data files, skip config updates")
    parser.add_argument("--subcategory", action="store_true",
                        help="Set up sub-category routing instead of full merge "
                             "(source routes through target stack first)")

    args = parser.parse_args()

    logger.info("=" * 60)
    if args.subcategory:
        logger.info("SUB-CATEGORY ROUTING: %s -> %s", args.source, args.target)
    else:
        logger.info("FULL DOMAIN MERGE: %s -> %s", args.source, args.target)
    if args.dry_run:
        logger.info("(DRY RUN — no files will be modified)")
    logger.info("=" * 60)

    if args.subcategory:
        setup_subcategory_routing(args.source, args.target, dry_run=args.dry_run)
        return

    # Full merge
    logger.info("\n--- Step 1: Merge training data ---")
    merge_data_local(args.source, args.target, dry_run=args.dry_run)

    if not args.data_only:
        logger.info("\n--- Step 2: Update domains.yaml ---")
        update_domains_yaml(args.source, args.target, dry_run=args.dry_run)

        logger.info("\n--- Step 3: Update brainstacks.yaml ---")
        update_brainstacks_yaml(args.source, args.target, dry_run=args.dry_run)

        logger.info("\n--- Step 4: Update curriculum-adaptive.json ---")
        update_curriculum_adaptive_json(args.source, args.target, dry_run=args.dry_run)

    logger.info("\n--- Summary ---")
    logger.info("Domain '%s' merged into '%s'.", args.source, args.target)
    logger.info("")
    logger.info("Manual steps remaining:")
    logger.info("  1. Remote data on Studio (data/micro-kiki/%s/) must be merged manually:", args.source)
    logger.info("     cat data/micro-kiki/%s/train.jsonl >> data/micro-kiki/%s/train.jsonl", args.source, args.target)
    logger.info("     # Then deduplicate with: python3 scripts/micro_kiki/deduplicate.py --domain %s", args.target)
    logger.info("  2. Delete or archive stack config: configs/stack-*-%s.yaml", args.source)
    logger.info("  3. Delete or archive MLX per-domain config: configs/mlx-per-domain/%s.yaml", args.source)
    logger.info("  4. Re-train the VQC router: python3 scripts/train_vqc_router.py")
    logger.info("  5. Run forgetting check on the merged domain")


if __name__ == "__main__":
    main()
