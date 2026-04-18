#!/usr/bin/env python
"""Measure per-layer cos-angle between two LoRA adapter stacks.

Phase 1a of OPLoRA experimentation (see .omc/brainstorm-oplora.md).
Angle-only (informational). Win-rate half is deferred to phase 1b.

CLI:
    python scripts/measure_forgetting.py \\
        --prior-adapter path/to/prior/adapter_model.safetensors \\
        --new-adapter   path/to/new/adapter_model.safetensors \\
        [--output results/forgetting-<stack>.json]

For each layer group {q,k,v,o}_proj, reconstruct weight delta = B @ A per
layer, then compute the geometric angle between prior_delta and new_delta
using ``src.eval.forgetting.GradientSubspaceAnalyzer.compute_angle``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from loguru import logger
from safetensors.torch import load_file

# Ensure ``src`` is importable when running from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.forgetting import ANGLE_THRESHOLD, GradientSubspaceAnalyzer  # noqa: E402

PROJ_GROUPS = ("q_proj", "k_proj", "v_proj", "o_proj")

# Regex matches e.g.
#   base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight
#   model.layers.3.self_attn.q_proj.lora_a.weight
_LORA_KEY = re.compile(
    r"(?P<prefix>.+?)\.(?P<proj>[qkvo]_proj)\.lora_(?P<ab>[AaBb])(?:\.default)?\.weight$"
)


def _load_tensors(path: Path) -> dict[str, torch.Tensor]:
    logger.info(f"loading safetensors: {path}")
    return load_file(str(path))


def _extract_deltas(tensors: dict[str, torch.Tensor]) -> dict[str, list[torch.Tensor]]:
    """Group LoRA A/B pairs by projection kind and compute B @ A per layer.

    Returns a mapping ``proj -> [delta_layer_0, delta_layer_1, ...]``.
    Layers with only A or only B (malformed) are skipped with a warning.
    """
    a_matrices: dict[tuple[str, str], torch.Tensor] = {}
    b_matrices: dict[tuple[str, str], torch.Tensor] = {}

    for key, tensor in tensors.items():
        m = _LORA_KEY.match(key)
        if not m:
            continue
        proj = m.group("proj")
        prefix = m.group("prefix")
        ab = m.group("ab").lower()
        bucket = a_matrices if ab == "a" else b_matrices
        bucket[(prefix, proj)] = tensor

    deltas: dict[str, list[torch.Tensor]] = defaultdict(list)
    for (prefix, proj), a in a_matrices.items():
        b = b_matrices.get((prefix, proj))
        if b is None:
            logger.warning(f"missing lora_B for {prefix}.{proj}; skipping")
            continue
        # Standard LoRA: A in (r, in_features), B in (out_features, r).
        # Delta weight = B @ A, shape (out_features, in_features).
        delta = b.float() @ a.float()
        deltas[proj].append(delta)
    for (prefix, proj), _ in b_matrices.items():
        if (prefix, proj) not in a_matrices:
            logger.warning(f"missing lora_A for {prefix}.{proj}; skipping")
    return deltas


def _stack_group(deltas: Iterable[torch.Tensor]) -> torch.Tensor:
    """Flatten each layer's delta to a column and stack → (n_params, n_layers)."""
    cols = [d.reshape(-1) for d in deltas]
    return torch.stack(cols, dim=1)


def compute_angles(
    prior_tensors: dict[str, torch.Tensor],
    new_tensors: dict[str, torch.Tensor],
    analyzer: GradientSubspaceAnalyzer | None = None,
) -> dict[str, float]:
    """Return a dict of per-projection angles (degrees).

    Uses ``GradientSubspaceAnalyzer.compute_angle`` treating each projection
    group's stacked per-layer deltas as the "gradient" matrix (columns =
    samples, rows = params).
    """
    analyzer = analyzer or GradientSubspaceAnalyzer()
    prior_deltas = _extract_deltas(prior_tensors)
    new_deltas = _extract_deltas(new_tensors)

    angles: dict[str, float] = {}
    for proj in PROJ_GROUPS:
        p_list = prior_deltas.get(proj, [])
        n_list = new_deltas.get(proj, [])
        if not p_list or not n_list:
            logger.warning(f"no deltas for {proj}; skipping")
            continue
        p_mat = _stack_group(p_list)
        n_mat = _stack_group(n_list)
        # Row-align if layer counts differ (truncate to min shared columns/rows).
        if p_mat.shape[0] != n_mat.shape[0]:
            min_rows = min(p_mat.shape[0], n_mat.shape[0])
            p_mat = p_mat[:min_rows]
            n_mat = n_mat[:min_rows]
        angle = analyzer.compute_angle(p_mat, n_mat)
        angles[proj] = angle
        logger.info(f"{proj}: angle = {angle:.3f}°")
    return angles


def build_report(angles: dict[str, float]) -> dict[str, object]:
    if not angles:
        return {
            "angle_degrees_mean": float("nan"),
            "angle_degrees_per_layer": {},
            "warning": "no matching LoRA layers found in either adapter",
            "gate_status": "angle_only_partial",
            "note": (
                "Win-rate half not measured; run paired eval for full gate."
            ),
        }
    mean_angle = float(np.mean(list(angles.values())))
    warning: str | None = None
    if mean_angle < ANGLE_THRESHOLD:
        warning = "angle below threshold"
    return {
        "angle_degrees_mean": mean_angle,
        "angle_degrees_per_layer": {k: float(v) for k, v in angles.items()},
        "warning": warning,
        "gate_status": "angle_only_partial",
        "note": "Win-rate half not measured; run paired eval for full gate.",
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure LoRA-delta cos-angle between two adapter stacks."
    )
    parser.add_argument(
        "--prior-adapter",
        required=True,
        type=Path,
        help="Path to prior stack adapter_model.safetensors",
    )
    parser.add_argument(
        "--new-adapter",
        required=True,
        type=Path,
        help="Path to newly-trained stack adapter_model.safetensors",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Send loguru to stderr; keep stdout clean for JSON.
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    args = _parse_args(argv)

    prior_tensors = _load_tensors(args.prior_adapter)
    new_tensors = _load_tensors(args.new_adapter)

    angles = compute_angles(prior_tensors, new_tensors)
    report = build_report(angles)

    if report.get("warning") == "angle below threshold":
        logger.warning(
            f"mean angle {report['angle_degrees_mean']:.2f}° < "
            f"{ANGLE_THRESHOLD}° — potential forgetting; run win-rate eval."
        )

    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        logger.info(f"wrote {args.output}")

    # Angle-only is informational; real gate needs win-rate (phase 1b).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
