#!/usr/bin/env python3
"""Build hybrid adapter directory from V2/V3 eval results.

Creates output/micro-kiki/stacks-hybrid/ with symlinks to either V2 or V3
adapters per domain, based on which version performed better.

Can run on any machine — just provide the correct paths.

Usage:
    # On Studio (default paths):
    uv run python scripts/build_hybrid_adapters.py

    # Custom paths:
    uv run python scripts/build_hybrid_adapters.py \\
        --v2-dir /path/to/stacks-v2 \\
        --v3-dir /path/to/stacks-v3-r16

    # Dry run:
    uv run python scripts/build_hybrid_adapters.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Domain eval results from V2 vs V3 comparison
# ---------------------------------------------------------------------------
# delta = v3_val_loss - v2_val_loss (positive = V3 regressed)

@dataclass
class DomainSpec:
    version: str       # "v2" or "v3"
    reason: str        # human-readable reason
    delta: float       # val_loss delta (v3 - v2), 0.0 for ties/v3-only


# V3 wins (negative delta = V3 improved)
V3_WINS: dict[str, float] = {
    "electronics": -0.55,
    "llm-orch": -0.18,
    "devops": -0.11,
    "security": -0.12,
    "web-frontend": -0.06,
}

# V2 wins (positive delta = V3 regressed)
V2_WINS: dict[str, float] = {
    "spice-sim": 1.51,
    "math": 0.20,
    "kicad-pcb": 0.19,
    "web-backend": 0.17,
    "music-audio": 0.08,
}

# Ties — use V3 (has null-space)
TIES: list[str] = [
    "python", "shell", "embedded", "reasoning", "chat-fr", "docker",
    "kicad-dsl", "spice", "rust", "typescript", "c-cpp", "cmake",
    "platformio", "git", "fastapi", "react-ui", "api-design", "testing",
    "networking", "database", "hardware-desc", "system-design",
    "documentation",
]

# V3-only (no V2 adapter)
V3_ONLY: list[str] = ["components", "llm-ops", "ml-training"]


def build_domain_map() -> dict[str, DomainSpec]:
    """Build the full domain -> spec mapping."""
    domains: dict[str, DomainSpec] = {}

    for domain, delta in V3_WINS.items():
        domains[domain] = DomainSpec(
            version="v3",
            reason=f"v3 wins {delta:+.2f}",
            delta=delta,
        )

    for domain, delta in V2_WINS.items():
        domains[domain] = DomainSpec(
            version="v2",
            reason=f"v3 regressed {delta:+.2f}",
            delta=delta,
        )

    for domain in TIES:
        domains[domain] = DomainSpec(
            version="v3",
            reason="tie (null-space)",
            delta=0.0,
        )

    for domain in V3_ONLY:
        domains[domain] = DomainSpec(
            version="v3",
            reason="v3-only",
            delta=0.0,
        )

    return domains


def read_val_loss(adapter_dir: Path) -> float | None:
    """Try to extract val_loss from adapter training artifacts."""
    for fname in ("training_args.json", "adapter_config.json"):
        fpath = adapter_dir / fname
        if fpath.exists():
            try:
                data = json.loads(fpath.read_text())
                for key in ("val_loss", "final_val_loss", "best_val_loss"):
                    if key in data and data[key] is not None:
                        return float(data[key])
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    return None


def build_hybrid(
    v2_dir: Path,
    v3_dir: Path,
    out_dir: Path,
    dry_run: bool = False,
) -> dict:
    """Build the hybrid adapter directory and manifest."""
    domain_map = build_domain_map()

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "v2_source": str(v2_dir),
        "v3_source": str(v3_dir),
        "domains": {},
    }

    stats = {"v2": 0, "v3": 0, "skip": 0}

    # Header
    print()
    print(f"{'DOMAIN':<20} {'VER':<6} {'VAL_LOSS':<10} {'REASON':<30} {'STATUS'}")
    print(f"{'-'*20} {'-'*6} {'-'*10} {'-'*30} {'-'*8}")

    for domain in sorted(domain_map):
        spec = domain_map[domain]

        # Resolve source path
        if spec.version == "v2":
            src = v2_dir / domain
        else:
            src = v3_dir / domain

        # Check existence
        if not dry_run and not src.exists():
            print(f"{domain:<20} {spec.version:<6} {'—':<10} {spec.reason:<30} SKIP")
            stats["skip"] += 1
            continue

        # Read val_loss if possible
        val_loss = None
        if not dry_run:
            val_loss = read_val_loss(src)

        # Create symlink
        link_path = out_dir / domain
        status = "OK"
        if dry_run:
            status = "DRY-RUN"
        else:
            # Remove existing link/dir
            if link_path.is_symlink() or link_path.exists():
                link_path.unlink()
            link_path.symlink_to(src, target_is_directory=True)

        # Record in manifest
        manifest["domains"][domain] = {
            "version": spec.version,
            "val_loss": val_loss,
            "reason": spec.reason,
        }
        stats[spec.version] += 1

        val_str = f"{val_loss:.4f}" if val_loss is not None else "—"
        print(f"{domain:<20} {spec.version:<6} {val_str:<10} {spec.reason:<30} {status}")

    # Write manifest
    manifest_path = out_dir / "hybrid_manifest.json"
    if not dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # Summary
    total = stats["v2"] + stats["v3"]
    print()
    print("=" * 50)
    print(f"  HYBRID BUILD SUMMARY")
    print(f"  V2 adapters: {stats['v2']}")
    print(f"  V3 adapters: {stats['v3']}")
    print(f"  Skipped:     {stats['skip']}")
    print(f"  Total:       {total}")
    print(f"  Manifest:    {manifest_path}")
    print(f"  Output dir:  {out_dir}")
    if dry_run:
        print("  (DRY RUN — no symlinks created)")
    print("=" * 50)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build hybrid adapter directory from V2/V3 eval results",
    )
    parser.add_argument(
        "--v2-dir",
        type=Path,
        default=Path("/Users/clems/KIKI-Mac_tunner/output/micro-kiki/stacks-v2"),
        help="Path to V2 adapter stacks (default: Studio path)",
    )
    parser.add_argument(
        "--v3-dir",
        type=Path,
        default=Path("/Users/clems/KIKI-Mac_tunner/output/micro-kiki/stacks-v3-r16"),
        help="Path to V3 adapter stacks (default: Studio path)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "micro-kiki" / "stacks-hybrid",
        help="Output directory for hybrid symlinks",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without creating symlinks",
    )
    args = parser.parse_args()

    if not args.dry_run:
        if not args.v2_dir.is_dir():
            print(f"ERROR: V2 directory not found: {args.v2_dir}", file=sys.stderr)
            print("       Run on Studio or pass --v2-dir", file=sys.stderr)
            sys.exit(1)
        if not args.v3_dir.is_dir():
            print(f"ERROR: V3 directory not found: {args.v3_dir}", file=sys.stderr)
            print("       Run on Studio or pass --v3-dir", file=sys.stderr)
            sys.exit(1)

    build_hybrid(args.v2_dir, args.v3_dir, args.out_dir, args.dry_run)


if __name__ == "__main__":
    main()
