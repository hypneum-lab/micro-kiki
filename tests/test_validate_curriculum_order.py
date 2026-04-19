"""Smoke test for scripts/validate_curriculum_order.py against the live configs."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from validate_curriculum_order import (  # noqa: E402
    FOUNDATION_RANK,
    classify_tiers,
    validate,
)


def test_curriculum_order_ok():
    ok, msg = validate()
    assert ok, msg


def test_foundation_rank_is_32():
    # Guardrail against silent rank-tier remapping.
    assert FOUNDATION_RANK == 32


def test_classify_tiers_partitions_cleanly():
    from validate_curriculum_order import MLX_PER_DOMAIN_DIR, _load_rank_by_domain

    rank_by_domain = _load_rank_by_domain(MLX_PER_DOMAIN_DIR)
    foundations, niches = classify_tiers(rank_by_domain)
    assert foundations, "expected at least one foundation (rank == 32)"
    assert niches, "expected at least one niche (rank < 32)"
    assert foundations.isdisjoint(niches), "foundations and niches must be disjoint"
