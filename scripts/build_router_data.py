"""Build router-v4 train/valid from freshly re-classified domain data.

Reads JSONL files from KIKI-Mac_tunner/data/micro-kiki/classified/<domain>.jsonl,
extracts user prompts, does an 80/20 stratified split, and writes:
  data/router-v4/train.jsonl
  data/router-v4/valid.jsonl

Each output row: {"prompt": str, "domain": str}

Usage:
    python scripts/build_router_data.py [--classified-dir PATH] [--out PATH] [--seed N]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

DEFAULT_CLASSIFIED = Path.home() / "KIKI-Mac_tunner" / "data" / "micro-kiki" / "classified"
DEFAULT_OUT = REPO_ROOT / "data" / "router-v4"

VALID_RATIO = 0.20


def extract_prompt(record: dict) -> str | None:
    """Pull the user prompt from messages[], prompt, or instruction fields."""
    if "messages" in record:
        for msg in record["messages"]:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        return None
    for key in ("prompt", "instruction"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def load_domain(path: Path, domain: str) -> list[str]:
    """Load all unique non-empty prompts from a domain JSONL file."""
    prompts: list[str] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("%s line %d: JSON decode error, skipping", domain, line_no)
                skipped += 1
                continue
            prompt = extract_prompt(record)
            if prompt:
                prompts.append(prompt)
            else:
                skipped += 1
    if skipped:
        logger.warning("%s: skipped %d records (no extractable prompt)", domain, skipped)
    return prompts


def stratified_split(
    prompts: list[str],
    domain: str,
    valid_ratio: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """Shuffle and split into train/valid rows."""
    shuffled = list(prompts)
    rng.shuffle(shuffled)
    n_valid = max(1, round(len(shuffled) * valid_ratio))
    valid_prompts = shuffled[:n_valid]
    train_prompts = shuffled[n_valid:]
    train_rows = [{"prompt": p, "domain": domain} for p in train_prompts]
    valid_rows = [{"prompt": p, "domain": domain} for p in valid_prompts]
    return train_rows, valid_rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build router-v4 data from classified domains")
    parser.add_argument(
        "--classified-dir",
        default=str(DEFAULT_CLASSIFIED),
        help="Directory containing per-domain <domain>.jsonl files",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Output directory for train.jsonl and valid.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=VALID_RATIO,
        help="Fraction of data to put into valid split (default 0.20)",
    )
    args = parser.parse_args()

    classified_dir = Path(args.classified_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    domain_files = sorted(classified_dir.glob("*.jsonl"))
    if not domain_files:
        logger.error("No .jsonl files found in %s", classified_dir)
        sys.exit(1)

    logger.info("Found %d domain files in %s", len(domain_files), classified_dir)

    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"

    n_train_total = 0
    n_valid_total = 0
    domain_stats: list[tuple[str, int, int]] = []

    with train_path.open("w", encoding="utf-8") as f_train, valid_path.open(
        "w", encoding="utf-8"
    ) as f_valid:
        for domain_file in domain_files:
            domain = domain_file.stem  # filename without .jsonl
            prompts = load_domain(domain_file, domain)
            if not prompts:
                logger.warning("%s: no prompts extracted, skipping domain", domain)
                continue

            train_rows, valid_rows = stratified_split(prompts, domain, args.valid_ratio, rng)

            for row in train_rows:
                f_train.write(json.dumps(row, ensure_ascii=False) + "\n")
            for row in valid_rows:
                f_valid.write(json.dumps(row, ensure_ascii=False) + "\n")

            n_train_total += len(train_rows)
            n_valid_total += len(valid_rows)
            domain_stats.append((domain, len(train_rows), len(valid_rows)))
            logger.info("%-16s  train=%5d  valid=%4d", domain, len(train_rows), len(valid_rows))

    logger.info("")
    logger.info("=" * 50)
    logger.info("TOTAL  train=%d  valid=%d  (%.0f%% / %.0f%%)",
                n_train_total, n_valid_total,
                100 * n_train_total / (n_train_total + n_valid_total),
                100 * n_valid_total / (n_train_total + n_valid_total))
    logger.info("Wrote %s", train_path)
    logger.info("Wrote %s", valid_path)


if __name__ == "__main__":
    main()
