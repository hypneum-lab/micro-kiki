"""Smoke test for scripts/validate_rank_schema.py against the live configs."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from validate_rank_schema import ALLOWED_RANKS, load_rank_alpha, validate  # noqa: E402

MLX_PER_DOMAIN_DIR = REPO_ROOT / "configs" / "mlx-per-domain"


def test_rank_schema_ok():
    ok, msg = validate()
    assert ok, msg


def test_each_config_has_rank_in_allowed_set():
    yaml_paths = [
        p
        for p in sorted(MLX_PER_DOMAIN_DIR.iterdir())
        if p.suffix == ".yaml" and not p.name.startswith("_")
    ]
    assert yaml_paths, f"no yaml files found in {MLX_PER_DOMAIN_DIR}"
    for path in yaml_paths:
        rank, alpha = load_rank_alpha(path)
        assert rank is not None, f"{path.name}: missing rank"
        assert alpha is not None, f"{path.name}: missing alpha"
        assert rank in ALLOWED_RANKS, f"{path.name}: rank={rank} not in {ALLOWED_RANKS}"
        assert alpha == 2 * rank, f"{path.name}: alpha={alpha} != 2*rank={2 * rank}"
