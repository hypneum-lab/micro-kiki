"""Guardrail: fail if any LoRA adapter has *all* ``lora_B`` tensors at zero.

The 2026-04-19 pre-pivot MoE-LoRA audit discovered ~35 adapters trained
with the custom ``_moe_lora`` path whose per-expert ``lora_b`` matrices
were stuck at their zero initialisation — the forward contribution
``A @ B = A @ 0 = 0`` collapses the gradient path, and weeks of GPU time
were wasted undetected. See:

    docs/research/2026-04-19-prepivot-moe-lora-audit.md

This validator reads a ``.safetensors`` adapter (or a directory of them),
computes the Frobenius norm of every tensor whose key ends in ``lora_b``
or ``lora_B``, and fails if *every* such norm is below ``epsilon``.

Dependencies: ``numpy`` + ``safetensors`` only (no torch — too heavy for
CI). We use ``safetensors.numpy.load_file`` which handles F32 / F16 /
BF16 natively; the header-parsing-by-hand route was tempting but BF16
needs a careful numpy view trick and a single small buffer load is a
fine tradeoff here.

CLI::

    python scripts/validate_adapter_health.py <adapter.safetensors>
    python scripts/validate_adapter_health.py --adapters-dir <path>
    python scripts/validate_adapter_health.py --epsilon 1e-4 <...>

Exit 0 on healthy, 1 on degenerate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

DEFAULT_EPSILON = 1e-6


def _is_lora_b_key(key: str) -> bool:
    """Return True if ``key`` names a LoRA ``B`` projection tensor.

    Accepts both ``lora_b`` (MLX / pre-pivot MoE-LoRA) and ``lora_B``
    (HF PEFT) spellings, as well as common suffixes (``.weight``).
    """
    lowered = key.lower()
    # strip a trailing '.weight' if present so we match the module name itself
    if lowered.endswith(".weight"):
        lowered = lowered[: -len(".weight")]
    return lowered.endswith("lora_b") or lowered.endswith(".lora_b")


def load_lora_b_norms(adapter_path: Path) -> dict[str, float]:
    """Return ``{key: frobenius_norm}`` for every ``lora_B`` tensor in the file."""
    tensors = load_file(str(adapter_path))
    norms: dict[str, float] = {}
    for key, arr in tensors.items():
        if not _is_lora_b_key(key):
            continue
        # Cast to float32 so BF16/F16 Frobenius norms are computed at full precision.
        as_f32 = np.asarray(arr, dtype=np.float32)
        norms[key] = float(np.linalg.norm(as_f32))
    return norms


def validate(adapter_path: Path, epsilon: float = DEFAULT_EPSILON) -> tuple[bool, str]:
    """Validate a single adapter file. Return ``(ok, message)``."""
    if not adapter_path.is_file():
        return False, f"MISSING: {adapter_path} is not a file"
    norms = load_lora_b_norms(adapter_path)
    if not norms:
        return False, (
            f"NO_LORA_B: {adapter_path} contains no lora_b / lora_B tensors "
            f"(is this an adapter file?)"
        )
    max_norm = max(norms.values())
    nonzero = sum(1 for v in norms.values() if v > epsilon)
    if max_norm <= epsilon:
        return False, (
            f"DEGENERATE: {adapter_path}\n"
            f"  all {len(norms)} lora_B tensors have norm <= epsilon ({epsilon:.1e})\n"
            f"  max norm = {max_norm:.3e}  (likely an untrained adapter — see "
            f"docs/research/2026-04-19-prepivot-moe-lora-audit.md)"
        )
    return True, (
        f"OK: {adapter_path}\n"
        f"  {nonzero}/{len(norms)} lora_B tensors above epsilon ({epsilon:.1e})\n"
        f"  max norm = {max_norm:.4f}"
    )


def _iter_adapter_files(adapters_dir: Path) -> list[Path]:
    """Recursively find ``*.safetensors`` files under ``adapters_dir``."""
    if not adapters_dir.is_dir():
        return []
    return sorted(adapters_dir.rglob("*.safetensors"))


def validate_dir(adapters_dir: Path, epsilon: float = DEFAULT_EPSILON) -> tuple[bool, str]:
    """Validate every safetensors file under ``adapters_dir`` (recursive)."""
    files = _iter_adapter_files(adapters_dir)
    if not files:
        return False, f"NO_ADAPTERS: no *.safetensors under {adapters_dir}"
    failing: list[str] = []
    ok_count = 0
    for path in files:
        ok, msg = validate(path, epsilon=epsilon)
        if ok:
            ok_count += 1
        else:
            failing.append(msg)
    if failing:
        header = (
            f"FAIL: {len(failing)}/{len(files)} adapter(s) under {adapters_dir} "
            f"are degenerate (epsilon={epsilon:.1e}):"
        )
        return False, header + "\n" + "\n".join(failing)
    return True, (
        f"OK: {ok_count}/{len(files)} adapter(s) under {adapters_dir} "
        f"pass lora_B health check (epsilon={epsilon:.1e})."
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-fast check that at least one lora_B tensor has nonzero norm."
    )
    parser.add_argument(
        "adapter",
        nargs="?",
        type=Path,
        help="Path to a single .safetensors adapter file.",
    )
    parser.add_argument(
        "--adapters-dir",
        type=Path,
        default=None,
        help="Recursively validate every .safetensors file under this directory.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help=(
            f"Norm threshold below which lora_B is considered zero "
            f"(default {DEFAULT_EPSILON:.0e})."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.adapters_dir is not None:
        ok, msg = validate_dir(args.adapters_dir, epsilon=args.epsilon)
    elif args.adapter is not None:
        ok, msg = validate(args.adapter, epsilon=args.epsilon)
    else:
        print("ERROR: pass a single .safetensors path or --adapters-dir <path>")
        return 1
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
