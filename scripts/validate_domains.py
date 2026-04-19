"""Validate that the 32-domain list is consistent across the three config mirrors.

Mirrors checked:
  1. configs/micro_kiki/domains.yaml       (dict under top-level `domains:`)
  2. configs/micro_kiki/brainstacks.yaml   (list under top-level `curriculum:`)
  3. configs/mlx-per-domain/*.yaml         (one file per domain; id = filename stem)

Drift between these three silently breaks training. This script is a fail-fast
gate for CI: exits 0 if all three sets are equal, 1 otherwise, printing a
readable diff.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAINS_YAML = REPO_ROOT / "configs" / "micro_kiki" / "domains.yaml"
BRAINSTACKS_YAML = REPO_ROOT / "configs" / "micro_kiki" / "brainstacks.yaml"
MLX_PER_DOMAIN_DIR = REPO_ROOT / "configs" / "mlx-per-domain"


def _load_domains_yaml(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "domains" not in data:
        raise ValueError(f"{path}: missing top-level 'domains:' key")
    domains = data["domains"]
    if not isinstance(domains, dict):
        raise ValueError(f"{path}: 'domains' is not a mapping")
    return set(domains.keys())


def _load_brainstacks_yaml(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "curriculum" not in data:
        raise ValueError(f"{path}: missing top-level 'curriculum:' key")
    curriculum = data["curriculum"]
    if not isinstance(curriculum, list):
        raise ValueError(f"{path}: 'curriculum' is not a list")
    out: set[str] = set()
    for item in curriculum:
        if isinstance(item, str):
            out.add(item)
        elif isinstance(item, dict) and len(item) == 1:
            # Supports nested form {phase_name: [domain, ...]} if introduced later.
            (val,) = item.values()
            if isinstance(val, list):
                out.update(str(d) for d in val)
        else:
            raise ValueError(f"{path}: unexpected curriculum entry: {item!r}")
    return out


def _load_mlx_per_domain(dir_path: Path) -> set[str]:
    if not dir_path.is_dir():
        raise ValueError(f"{dir_path}: not a directory")
    out: set[str] = set()
    for child in sorted(dir_path.iterdir()):
        if child.suffix != ".yaml":
            continue
        if child.name.startswith("_") or child.name == "AGENTS.md":
            continue
        # Prefer an explicit `domain:` key inside if present; else use stem.
        with child.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict) and isinstance(data.get("domain"), str):
            out.add(data["domain"])
        else:
            out.add(child.stem)
    return out


def load_domains_from_each_mirror() -> dict[str, set[str]]:
    """Return the domain id set reported by each of the three mirrors."""
    return {
        "domains_yaml": _load_domains_yaml(DOMAINS_YAML),
        "brainstacks_yaml": _load_brainstacks_yaml(BRAINSTACKS_YAML),
        "mlx_per_domain": _load_mlx_per_domain(MLX_PER_DOMAIN_DIR),
    }


def _fmt_set(s: set[str]) -> str:
    if not s:
        return "(none)"
    return ", ".join(sorted(s))


def validate() -> tuple[bool, str]:
    """Check that all three mirrors agree. Return (ok, message)."""
    mirrors = load_domains_from_each_mirror()
    names = list(mirrors.keys())
    sizes = {n: len(mirrors[n]) for n in names}

    # Are all three sets equal?
    sets = list(mirrors.values())
    if sets[0] == sets[1] == sets[2]:
        return (
            True,
            (
                f"OK: all three mirrors agree on {len(sets[0])} domains "
                f"({', '.join(f'{n}={sizes[n]}' for n in names)})."
            ),
        )

    lines: list[str] = ["DRIFT: domain-list mirrors disagree."]
    lines.append(f"  sizes: {', '.join(f'{n}={sizes[n]}' for n in names)}")
    lines.append("")
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i >= j:
                continue
            a_only = mirrors[a] - mirrors[b]
            b_only = mirrors[b] - mirrors[a]
            if not a_only and not b_only:
                lines.append(f"  {a} == {b}: OK")
                continue
            lines.append(f"  {a} \\ {b}: {_fmt_set(a_only)}")
            lines.append(f"  {b} \\ {a}: {_fmt_set(b_only)}")
    return False, "\n".join(lines)


def main() -> int:
    ok, msg = validate()
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
