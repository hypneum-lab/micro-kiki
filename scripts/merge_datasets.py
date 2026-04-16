"""Merge KIKI-Mac_tunner + HuggingFace mascarade datasets for 10 niche domains.

Downloads HF datasets, converts formats, deduplicates, and writes merged
data/merged/<domain>/train.jsonl. All examples in {"messages": [...]} format.

Usage:
    uv run scripts/merge_datasets.py --all
    uv run scripts/merge_datasets.py --domain kicad-dsl
    uv run scripts/merge_datasets.py --domain embedded --kiki-root /custom/path
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset mappings
# ---------------------------------------------------------------------------

ALL_DOMAINS = [
    "kicad-dsl",
    "spice",
    "emc",
    "stm32",
    "embedded",
    "power",
    "dsp",
    "freecad",
    "platformio",
    "iot",
]

# HF repos serving ShareGPT-format files (mascarade-* datasets)
HF_DATASETS: dict[str, tuple[str, str]] = {
    "kicad-dsl": ("electron-rare/mascarade-kicad-dataset", "kicad_chat.jsonl"),
    "spice":     ("electron-rare/mascarade-spice-dataset", "spice_chat.jsonl"),
    "emc":       ("electron-rare/mascarade-emc-dataset", "emc_chat.jsonl"),
    "stm32":     ("electron-rare/mascarade-stm32-dataset", "stm32_chat.jsonl"),
    "embedded":  ("electron-rare/mascarade-embedded-dataset", "embedded_chat.jsonl"),
    "power":     ("electron-rare/mascarade-power-dataset", "power_chat.jsonl"),
    "dsp":       ("electron-rare/mascarade-dsp-dataset", "dsp_chat.jsonl"),
}

# HF repos serving instruction-format files (kill-life-embedded-qa)
KILL_LIFE: dict[str, list[tuple[str, str]]] = {
    "embedded": [
        ("electron-rare/kill-life-embedded-qa", "data/kill_life_embedded_qa.jsonl"),
        ("electron-rare/kill-life-embedded-qa", "data/kb_firmware_qa.jsonl"),
        ("electron-rare/kill-life-embedded-qa", "data/kb_components_qa.jsonl"),
    ],
    "kicad-dsl": [
        ("electron-rare/kill-life-embedded-qa", "data/kb_kicad_qa.jsonl"),
    ],
    "spice": [
        ("electron-rare/kill-life-embedded-qa", "data/kb_spice_qa.jsonl"),
    ],
}

# ---------------------------------------------------------------------------
# Format converters
# ---------------------------------------------------------------------------

def sharegpt_to_messages(conv: dict) -> list[dict] | None:
    """Convert a ShareGPT-format record to OpenAI messages list.

    Input:  {"conversations": [{"from": "human", "value": "..."}, ...]}
    Output: [{"role": "user", "content": "..."}, ...]
    Returns None if the conversation is malformed or empty.
    """
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    raw = conv.get("conversations") or conv.get("messages")
    if not raw:
        return None
    messages = []
    for turn in raw:
        # Already in messages format
        if "role" in turn and "content" in turn:
            messages.append({"role": turn["role"], "content": turn["content"]})
            continue
        frm = turn.get("from", "")
        val = turn.get("value", "")
        role = role_map.get(frm)
        if not role or not val:
            logger.debug("Skipping unknown role/empty value: from=%r", frm)
            continue
        messages.append({"role": role, "content": val})
    return messages if messages else None


def instruction_to_messages(row: dict) -> list[dict] | None:
    """Convert instruction/output format to OpenAI messages list.

    Input:  {"instruction": "...", "input": "...", "output": "..."}
    Output: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    Returns None if required fields are missing.
    """
    instruction = row.get("instruction", "").strip()
    inp = row.get("input", "").strip()
    output = row.get("output", "").strip()
    if not instruction or not output:
        return None
    user_content = f"{instruction}\n\n{inp}".strip() if inp else instruction
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def dedup_by_hash(examples: list[dict]) -> list[dict]:
    """Deduplicate examples by MD5 hash of their JSON content."""
    seen: set[str] = set()
    result: list[dict] = []
    for ex in examples:
        key = hashlib.md5(json.dumps(ex, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            result.append(ex)
    return result


# ---------------------------------------------------------------------------
# HF download helpers
# ---------------------------------------------------------------------------

def _hf_download(repo_id: str, filename: str) -> Path | None:
    """Download a file from a HuggingFace repo, returning local cache path."""
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        logger.info("Downloaded %s / %s → %s", repo_id, filename, local)
        return Path(local)
    except Exception as exc:
        logger.warning("HF download failed for %s / %s: %s", repo_id, filename, exc)
        return None


def load_sharegpt_file(path: Path) -> list[dict]:
    """Load and convert a ShareGPT JSONL file."""
    examples: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("%s line %d: JSON decode error: %s", path.name, lineno, exc)
            continue
        messages = sharegpt_to_messages(record)
        if messages:
            examples.append({"messages": messages})
        else:
            logger.debug("%s line %d: empty/malformed conversation", path.name, lineno)
    return examples


def load_instruction_file(path: Path) -> list[dict]:
    """Load and convert an instruction-format JSONL file."""
    examples: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("%s line %d: JSON decode error: %s", path.name, lineno, exc)
            continue
        # Support both instruction format and pre-converted messages format
        if "messages" in record:
            examples.append(record)
        else:
            messages = instruction_to_messages(record)
            if messages:
                examples.append({"messages": messages})
    return examples


def load_kiki_messages_file(path: Path) -> list[dict]:
    """Load a KIKI-Mac_tunner train.jsonl (already in messages format)."""
    examples: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("%s line %d: JSON decode error: %s", path.name, lineno, exc)
            continue
        # Already {"messages": [...]} — validate minimal structure
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            logger.debug("%s line %d: no valid messages key", path.name, lineno)
            continue
        examples.append({"messages": messages})
    return examples


# ---------------------------------------------------------------------------
# Per-domain merge
# ---------------------------------------------------------------------------

def merge_domain(
    domain: str,
    kiki_root: Path,
    output_root: Path,
) -> dict[str, int]:
    """Merge all sources for one domain, write output, return count breakdown."""
    counts: dict[str, int] = {}
    all_examples: list[dict] = []

    # 1. KIKI-Mac_tunner local data
    kiki_path = kiki_root / domain / "train.jsonl"
    if kiki_path.exists():
        kiki_examples = load_kiki_messages_file(kiki_path)
        logger.info("[%s] KIKI local: %d examples from %s", domain, len(kiki_examples), kiki_path)
        counts["kiki_local"] = len(kiki_examples)
        all_examples.extend(kiki_examples)
    else:
        logger.warning("[%s] KIKI local not found: %s — skipping", domain, kiki_path)
        counts["kiki_local"] = 0

    # 2. HuggingFace mascarade ShareGPT dataset
    if domain in HF_DATASETS:
        repo_id, filename = HF_DATASETS[domain]
        hf_path = _hf_download(repo_id, filename)
        if hf_path:
            hf_examples = load_sharegpt_file(hf_path)
            logger.info("[%s] HF mascarade: %d examples from %s/%s", domain, len(hf_examples), repo_id, filename)
            counts["hf_mascarade"] = len(hf_examples)
            all_examples.extend(hf_examples)
        else:
            counts["hf_mascarade"] = 0
    else:
        counts["hf_mascarade"] = 0

    # 3. Kill-life instruction QA
    if domain in KILL_LIFE:
        kill_life_total = 0
        for repo_id, filename in KILL_LIFE[domain]:
            hf_path = _hf_download(repo_id, filename)
            if hf_path:
                kl_examples = load_instruction_file(hf_path)
                logger.info("[%s] kill-life QA: %d examples from %s/%s", domain, len(kl_examples), repo_id, filename)
                kill_life_total += len(kl_examples)
                all_examples.extend(kl_examples)
            else:
                logger.warning("[%s] kill-life file not found: %s/%s — skipping", domain, repo_id, filename)
        counts["kill_life_qa"] = kill_life_total
    else:
        counts["kill_life_qa"] = 0

    # 4. Deduplicate
    before_dedup = len(all_examples)
    all_examples = dedup_by_hash(all_examples)
    after_dedup = len(all_examples)
    counts["duplicates_removed"] = before_dedup - after_dedup
    counts["total"] = after_dedup

    if not all_examples:
        logger.warning("[%s] No examples after merge — output will be empty", domain)

    # 5. Write output
    out_dir = output_root / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for ex in all_examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info("[%s] Written %d examples → %s", domain, after_dedup, out_path)
    return counts


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Merge KIKI-Mac_tunner + HF mascarade datasets for micro-kiki domains.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="Merge all known domains.",
    )
    group.add_argument(
        "--domain",
        choices=ALL_DOMAINS,
        metavar="DOMAIN",
        help=f"Merge a single domain. Choices: {', '.join(ALL_DOMAINS)}",
    )
    parser.add_argument(
        "--kiki-root",
        default=None,
        metavar="PATH",
        help=(
            "Override path to KIKI-Mac_tunner data root "
            "(default: ~/KIKI-Mac_tunner/data/micro-kiki). "
            "Use this on non-Studio machines where that directory is absent."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="data/merged",
        metavar="PATH",
        help="Root directory for merged outputs (default: data/merged).",
    )
    args = parser.parse_args()

    kiki_root = Path(args.kiki_root).expanduser() if args.kiki_root else Path.home() / "KIKI-Mac_tunner" / "data" / "micro-kiki"
    output_root = Path(args.output_root)

    domains = ALL_DOMAINS if args.all else [args.domain]

    summary: dict[str, dict[str, int]] = {}
    for domain in domains:
        try:
            counts = merge_domain(domain, kiki_root, output_root)
            summary[domain] = counts
        except Exception as exc:
            logger.error("[%s] Unexpected error: %s", domain, exc, exc_info=True)
            summary[domain] = {"error": str(exc)}

    # Print JSON summary to stdout (single print allowed per conventions)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
