"""Validate that the training curriculum orders foundations before niches.

Per project CLAUDE.md the adapter rank encodes tier:
  * rank == 32  => foundation
  * rank <  32  => niche  (typical niches are 4, 8, or 16)

``configs/micro_kiki/brainstacks.yaml`` has a top-level ``curriculum:`` list
giving the sequential training order across all 32 domains. This validator
asserts that every foundation domain appears in the curriculum strictly
before any niche domain (i.e. no niche is scheduled ahead of any foundation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
BRAINSTACKS_YAML = REPO_ROOT / "configs" / "micro_kiki" / "brainstacks.yaml"
MLX_PER_DOMAIN_DIR = REPO_ROOT / "configs" / "mlx-per-domain"

FOUNDATION_RANK = 32


def _load_curriculum(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "curriculum" not in data:
        raise ValueError(f"{path}: missing top-level 'curriculum:' key")
    curriculum = data["curriculum"]
    if not isinstance(curriculum, list):
        raise ValueError(f"{path}: 'curriculum' is not a list")
    out: list[str] = []
    for item in curriculum:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and len(item) == 1:
            (val,) = item.values()
            if isinstance(val, list):
                out.extend(str(d) for d in val)
            else:
                raise ValueError(f"{path}: unexpected curriculum entry: {item!r}")
        else:
            raise ValueError(f"{path}: unexpected curriculum entry: {item!r}")
    return out


def _load_rank_by_domain(dir_path: Path) -> dict[str, int]:
    if not dir_path.is_dir():
        raise ValueError(f"{dir_path}: not a directory")
    out: dict[str, int] = {}
    for child in sorted(dir_path.iterdir()):
        if child.suffix != ".yaml":
            continue
        if child.name.startswith("_") or child.name == "AGENTS.md":
            continue
        with child.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            continue
        lora = data.get("lora_parameters")
        if not isinstance(lora, dict):
            continue
        rank = lora.get("rank")
        if isinstance(rank, (int, float)):
            domain = data["domain"] if isinstance(data.get("domain"), str) else child.stem
            out[domain] = int(rank)
    return out


def classify_tiers(rank_by_domain: dict[str, int]) -> tuple[set[str], set[str]]:
    """Split domains into (foundations, niches) by rank."""
    foundations = {d for d, r in rank_by_domain.items() if r == FOUNDATION_RANK}
    niches = {d for d, r in rank_by_domain.items() if r < FOUNDATION_RANK}
    return foundations, niches


def validate() -> tuple[bool, str]:
    """Check foundations precede niches in curriculum. Return (ok, message)."""
    curriculum = _load_curriculum(BRAINSTACKS_YAML)
    rank_by_domain = _load_rank_by_domain(MLX_PER_DOMAIN_DIR)
    foundations, niches = classify_tiers(rank_by_domain)

    if not foundations:
        return False, "DRIFT: no foundation domains (rank == 32) found in mlx-per-domain/"
    if not niches:
        return False, "DRIFT: no niche domains (rank < 32) found in mlx-per-domain/"

    # Position of each domain in the curriculum (first occurrence wins).
    position: dict[str, int] = {}
    for i, d in enumerate(curriculum):
        position.setdefault(d, i)

    # Check that every foundation appears and every niche appears.
    missing_foundations = sorted(foundations - position.keys())
    missing_niches = sorted(niches - position.keys())
    if missing_foundations or missing_niches:
        lines = ["DRIFT: curriculum missing tier members."]
        if missing_foundations:
            lines.append(f"  missing foundations: {', '.join(missing_foundations)}")
        if missing_niches:
            lines.append(f"  missing niches: {', '.join(missing_niches)}")
        return False, "\n".join(lines)

    last_foundation_pos = max(position[d] for d in foundations)
    first_niche_pos = min(position[d] for d in niches)

    if last_foundation_pos < first_niche_pos:
        return (
            True,
            (
                f"OK: all {len(foundations)} foundations precede all "
                f"{len(niches)} niches (last foundation @ idx {last_foundation_pos}, "
                f"first niche @ idx {first_niche_pos})."
            ),
        )

    # Find the specific offending pairs for the diff.
    late_foundations = sorted(
        (d for d in foundations if position[d] > first_niche_pos),
        key=lambda d: position[d],
    )
    early_niches = sorted(
        (d for d in niches if position[d] < last_foundation_pos),
        key=lambda d: position[d],
    )
    lines = ["DRIFT: curriculum schedules niches before foundations."]
    lines.append(f"  last foundation index: {last_foundation_pos}")
    lines.append(f"  first niche index:     {first_niche_pos}")
    if late_foundations:
        lines.append(
            "  foundations scheduled after first niche: "
            + ", ".join(f"{d}@{position[d]}" for d in late_foundations)
        )
    if early_niches:
        lines.append(
            "  niches scheduled before last foundation: "
            + ", ".join(f"{d}@{position[d]}" for d in early_niches)
        )
    return False, "\n".join(lines)


def main() -> int:
    ok, msg = validate()
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
