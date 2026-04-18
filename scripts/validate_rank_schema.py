"""Validate LoRA rank/alpha schema across per-domain MLX configs.

For each `configs/mlx-per-domain/*.yaml` (expected 32 files), assert:

  * ``lora_parameters.rank`` is one of ``{4, 8, 12, 16, 32}``
  * ``lora_parameters.alpha == 2 * rank`` (per project CLAUDE.md)

Exits 0 if every file passes, 1 otherwise, printing a per-file diff.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MLX_PER_DOMAIN_DIR = REPO_ROOT / "configs" / "mlx-per-domain"

ALLOWED_RANKS: set[int] = {4, 8, 12, 16, 32}


def _iter_domain_config_paths(dir_path: Path) -> list[Path]:
    if not dir_path.is_dir():
        raise ValueError(f"{dir_path}: not a directory")
    out: list[Path] = []
    for child in sorted(dir_path.iterdir()):
        if child.suffix != ".yaml":
            continue
        if child.name.startswith("_") or child.name == "AGENTS.md":
            continue
        out.append(child)
    return out


def load_rank_alpha(path: Path) -> tuple[int | None, int | None]:
    """Return (rank, alpha) from a per-domain yaml, or (None, None) if missing."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return None, None
    lora = data.get("lora_parameters")
    if not isinstance(lora, dict):
        return None, None
    rank = lora.get("rank")
    alpha = lora.get("alpha")
    rank_int = int(rank) if isinstance(rank, (int, float)) else None
    alpha_int = int(alpha) if isinstance(alpha, (int, float)) else None
    return rank_int, alpha_int


def validate() -> tuple[bool, str]:
    """Check rank/alpha for all per-domain configs. Return (ok, message)."""
    paths = _iter_domain_config_paths(MLX_PER_DOMAIN_DIR)
    if not paths:
        return False, f"DRIFT: no per-domain yaml files found in {MLX_PER_DOMAIN_DIR}"

    violations: list[str] = []
    for path in paths:
        rank, alpha = load_rank_alpha(path)
        if rank is None or alpha is None:
            violations.append(
                f"  {path.name}: missing lora_parameters.rank or "
                f"lora_parameters.alpha (rank={rank!r}, alpha={alpha!r})"
            )
            continue
        if rank not in ALLOWED_RANKS:
            violations.append(
                f"  {path.name}: rank={rank} not in {sorted(ALLOWED_RANKS)}"
            )
        expected_alpha = 2 * rank
        if alpha != expected_alpha:
            violations.append(
                f"  {path.name}: alpha={alpha} but rank={rank}, "
                f"expected alpha={expected_alpha}"
            )

    if not violations:
        return (
            True,
            f"OK: {len(paths)} per-domain configs pass rank/alpha schema "
            f"(rank in {sorted(ALLOWED_RANKS)}, alpha == 2*rank).",
        )

    lines = [f"DRIFT: rank/alpha schema violations in {len(violations)} entries:"]
    lines.extend(violations)
    return False, "\n".join(lines)


def main() -> int:
    ok, msg = validate()
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
