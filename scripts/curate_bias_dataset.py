"""Validate and report statistics for the bias training dataset.

Usage:
    uv run python scripts/curate_bias_dataset.py

Exit codes:
    0  — dataset valid: ≥5000 pairs and ≥500 per bias type
    1  — validation failed
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "bias" / "bias_pairs.jsonl"
REQUIRED_FIELDS = {"biased_prompt", "fair_prompt", "bias_type", "expected_behavior"}
VALID_BIAS_TYPES = {
    "confirmation",
    "anchoring",
    "authority",
    "recency",
    "framing",
    "stereotyping",
}
MIN_TOTAL = 5000
MIN_PER_TYPE = 500


def validate_dataset(path: Path) -> int:
    """Load, validate, and report stats for the bias dataset.

    Returns:
        0 if all checks pass, 1 otherwise.
    """
    if not path.exists():
        logger.error("Dataset not found: %s", path)
        return 1

    total = 0
    malformed = 0
    type_counts: Counter[str] = Counter()
    unknown_types: Counter[str] = Counter()

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Line %d: invalid JSON — %s", line_no, exc)
                malformed += 1
                continue

            missing = REQUIRED_FIELDS - set(obj.keys())
            if missing:
                logger.warning("Line %d: missing fields %s", line_no, missing)
                malformed += 1
                continue

            bias_type = obj["bias_type"]
            if bias_type not in VALID_BIAS_TYPES:
                unknown_types[bias_type] += 1
                malformed += 1
                continue

            # Check non-empty strings
            if not all(isinstance(obj[k], str) and obj[k].strip() for k in REQUIRED_FIELDS):
                logger.warning("Line %d: empty field value", line_no)
                malformed += 1
                continue

            type_counts[bias_type] += 1
            total += 1

    # Report
    logger.info("=" * 60)
    logger.info("Bias Dataset Validation Report")
    logger.info("=" * 60)
    logger.info("File: %s", path)
    logger.info("Total valid pairs: %d", total)
    logger.info("Malformed lines:   %d", malformed)
    logger.info("-" * 40)

    for bt in sorted(VALID_BIAS_TYPES):
        count = type_counts.get(bt, 0)
        status = "OK" if count >= MIN_PER_TYPE else "FAIL"
        logger.info("  %-20s %5d  [%s]", bt, count, status)

    if unknown_types:
        logger.info("-" * 40)
        logger.info("Unknown bias types:")
        for bt, count in unknown_types.most_common():
            logger.info("  %-20s %5d", bt, count)

    logger.info("=" * 60)

    # Checks
    ok = True
    if total < MIN_TOTAL:
        logger.error("FAIL: total pairs %d < %d required", total, MIN_TOTAL)
        ok = False

    for bt in VALID_BIAS_TYPES:
        if type_counts.get(bt, 0) < MIN_PER_TYPE:
            logger.error(
                "FAIL: %s has %d pairs < %d required",
                bt,
                type_counts.get(bt, 0),
                MIN_PER_TYPE,
            )
            ok = False

    if ok:
        logger.info("PASS: all checks passed")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(validate_dataset(DATASET_PATH))
