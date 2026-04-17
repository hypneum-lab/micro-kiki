#!/usr/bin/env python3
"""Generate synthetic DPO preference pairs from existing SFT training data.

For each domain, takes up to 100 prompts from data/merged/<domain>/train.jsonl.
- "Chosen" = the original assistant response (assumed high quality).
- "Rejected" = a synthetically degraded version (truncation, sentence shuffle,
  wrong-domain keyword injection, or a mix of all three).

Output: data/dpo/<domain>/train.jsonl  (JSONL with prompt/chosen/rejected keys,
        matching the format expected by train_dpo_niches.py)

Usage:
    uv run scripts/gen_dpo_pairs.py
    uv run scripts/gen_dpo_pairs.py --domains embedded,electronics
    uv run scripts/gen_dpo_pairs.py --max-pairs 200
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MERGED_DATA = PROJECT_ROOT / "data" / "merged"
DPO_DATA = PROJECT_ROOT / "data" / "dpo"

DEFAULT_DOMAINS = ["embedded", "electronics", "kicad-dsl", "power", "stm32"]
DEFAULT_MAX_PAIRS = 100

# Keywords foreign to each domain — used for wrong-domain injection.
FOREIGN_KEYWORDS: dict[str, list[str]] = {
    "embedded": [
        "SELECT * FROM", "border-radius", "npm install", "DataFrame",
        "font-size", "JOIN users ON", "pip install django",
    ],
    "electronics": [
        "docker compose", "kubectl apply", "git rebase", "async def",
        "margin-top", "SELECT COUNT", "npm run build",
    ],
    "kicad-dsl": [
        "print('hello')", "std::vector", "REST API", "SELECT *",
        "docker run", "npm test", "background-color",
    ],
    "power": [
        "CSS grid", "React.useState", "git merge", "docker pull",
        "pip install", "SELECT id FROM", "kubectl logs",
    ],
    "stm32": [
        "npm install", "background-image", "SELECT name", "pip freeze",
        "docker build", "git checkout", "React.useEffect",
    ],
}

# Fallback foreign keywords for any domain not explicitly listed.
_DEFAULT_FOREIGN = [
    "SELECT * FROM users", "npm install", "border-radius: 5px",
    "docker compose up", "pip install flask", "git rebase -i",
    "React.createElement",
]


# ---------------------------------------------------------------------------
# Degradation strategies
# ---------------------------------------------------------------------------


def _truncate(text: str) -> str:
    """Truncate to roughly 30-50% of the original, cutting mid-sentence."""
    if len(text) < 40:
        return text[:max(10, len(text) // 3)]
    cut_point = random.randint(len(text) // 4, len(text) // 2)
    # Try to cut in the middle of a word for a more obviously broken result.
    return text[:cut_point].rstrip() + "..."


def _shuffle_sentences(text: str) -> str:
    """Split on sentence boundaries and shuffle the order."""
    # Split on period/newline boundaries, keeping non-empty parts.
    parts = re.split(r'(?<=[.!?\n])\s+', text)
    parts = [p for p in parts if p.strip()]
    if len(parts) <= 2:
        # Too few sentences — reverse instead.
        return text[::-1][:len(text)]  # full reverse (clearly wrong)
    random.shuffle(parts)
    return " ".join(parts)


def _inject_foreign(text: str, domain: str) -> str:
    """Insert 2-3 foreign-domain keywords at random positions."""
    keywords = FOREIGN_KEYWORDS.get(domain, _DEFAULT_FOREIGN)
    n_inject = random.randint(2, 3)
    chosen_kw = random.sample(keywords, min(n_inject, len(keywords)))

    sentences = re.split(r'(?<=[.!?\n])\s+', text)
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return text

    for kw in chosen_kw:
        pos = random.randint(0, len(sentences))
        sentences.insert(pos, kw)

    return " ".join(sentences)


def _mixed_degrade(text: str, domain: str) -> str:
    """Apply truncation + foreign keyword injection."""
    truncated = _truncate(text)
    return _inject_foreign(truncated, domain)


def degrade(text: str, domain: str) -> str:
    """Apply a random degradation strategy to produce a rejected response."""
    strategies = [_truncate, _shuffle_sentences,
                  lambda t: _inject_foreign(t, domain),
                  lambda t: _mixed_degrade(t, domain)]
    strategy = random.choice(strategies)
    result = strategy(text)
    # Ensure rejected differs from chosen.
    if result == text:
        result = _truncate(text)
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_pairs(domain: str, max_pairs: int) -> list[dict]:
    """Load (prompt, assistant_response) from merged training data."""
    data_path = MERGED_DATA / domain / "train.jsonl"
    if not data_path.exists():
        print(f"  WARNING: {data_path} not found, skipping {domain}")
        return []

    records: list[dict] = []
    with data_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue

            prompt = ""
            assistant = ""

            if "messages" in entry:
                for msg in entry["messages"]:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user" and not prompt:
                        prompt = content.strip()
                    elif role == "assistant" and not assistant:
                        assistant = content.strip()
            else:
                prompt = (entry.get("prompt") or entry.get("instruction")
                          or entry.get("input", "")).strip()
                assistant = (entry.get("output") or entry.get("response")
                             or entry.get("completion", "")).strip()

            # Skip entries where the assistant response is too short to degrade.
            if prompt and assistant and len(assistant) >= 30:
                records.append({"prompt": prompt, "chosen": assistant})

            if len(records) >= max_pairs:
                break

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_for_domain(domain: str, max_pairs: int) -> int:
    """Generate DPO pairs for one domain. Returns count written."""
    records = load_pairs(domain, max_pairs)
    if not records:
        print(f"  {domain}: 0 records loaded — skipped")
        return 0

    pairs: list[dict] = []
    for rec in records:
        rejected = degrade(rec["chosen"], domain)
        pairs.append({
            "prompt": rec["prompt"],
            "chosen": rec["chosen"],
            "rejected": rejected,
            "domain": domain,
        })

    out_dir = DPO_DATA / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.jsonl"

    with out_path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"  {domain}: {len(pairs)} pairs -> {out_path}")
    return len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic DPO preference pairs from SFT data.",
    )
    parser.add_argument(
        "--domains",
        default=",".join(DEFAULT_DOMAINS),
        help=f"Comma-separated domain list (default: {','.join(DEFAULT_DOMAINS)})",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=DEFAULT_MAX_PAIRS,
        help=f"Max pairs per domain (default: {DEFAULT_MAX_PAIRS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    print(f"Generating DPO pairs for {len(domains)} domains, max {args.max_pairs} each")
    print(f"  Source: {MERGED_DATA}")
    print(f"  Output: {DPO_DATA}")
    print()

    total = 0
    for domain in domains:
        count = generate_for_domain(domain, args.max_pairs)
        total += count

    print(f"\nTotal: {total} DPO pairs across {len(domains)} domains")


if __name__ == "__main__":
    main()
