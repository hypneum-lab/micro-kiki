"""Forgetting check framework: win-rate delta + gradient subspace angle.

Run after each stack training to detect interference with prior stacks.
Reference: arxiv 2603.02224 (Subspace Geometry, 2026).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForgettingCheckResult:
    prior_stack: str
    new_stack: str
    win_rate_delta: float
    gradient_subspace_angle_deg: float
    passed: bool
    reason: str


def check_forgetting(
    win_rate_delta: float,
    angle_deg: float,
    prior_stack: str,
    new_stack: str,
    max_delta: float = 0.03,
    min_angle: float = 30.0,
) -> ForgettingCheckResult:
    """Check if new stack causes forgetting on prior stack.

    Fails only if BOTH conditions are true:
    - win_rate_delta > max_delta (regression on prior stack)
    - gradient_subspace_angle < min_angle (high interference)
    """
    delta_bad = win_rate_delta > max_delta
    angle_bad = angle_deg < min_angle
    failed = delta_bad and angle_bad

    if failed:
        reason = f"FORGETTING: delta={win_rate_delta:.3f}>{max_delta}, angle={angle_deg:.1f}<{min_angle}"
    elif delta_bad:
        reason = f"Delta high ({win_rate_delta:.3f}) but angle safe ({angle_deg:.1f}°)"
    elif angle_bad:
        reason = f"Angle low ({angle_deg:.1f}°) but delta safe ({win_rate_delta:.3f})"
    else:
        reason = "OK"

    return ForgettingCheckResult(
        prior_stack=prior_stack,
        new_stack=new_stack,
        win_rate_delta=win_rate_delta,
        gradient_subspace_angle_deg=angle_deg,
        passed=not failed,
        reason=reason,
    )


def compute_subspace_angle(lora_a_new, lora_a_prior) -> float:
    """Compute angle between LoRA update subspaces.

    Args:
        lora_a_new: concatenated A matrices of new stack (D, R_new)
        lora_a_prior: concatenated A matrices of prior stack (D, R_prior)

    Returns:
        Angle in degrees. 90° = orthogonal (no interference), 0° = identical.
    """
    import torch
    import math

    q_new, _ = torch.linalg.qr(lora_a_new.float())
    q_prior, _ = torch.linalg.qr(lora_a_prior.float())

    # Principal angle via SVD of Q_new^T @ Q_prior
    cross = q_new.T @ q_prior
    singular_values = torch.linalg.svdvals(cross)
    # Clamp for numerical stability
    max_sv = singular_values[0].clamp(max=1.0)
    angle_rad = torch.acos(max_sv)
    return math.degrees(angle_rad.item())


def save_forgetting_report(
    results: list[ForgettingCheckResult],
    new_stack: str,
    output_dir: str = "results",
) -> Path:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    path = output_dir_path / f"forgetting-{new_stack}.json"
    data = {
        "new_stack": new_stack,
        "timestamp": datetime.now().isoformat(),
        "all_passed": all(r.passed for r in results),
        "checks": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info("Forgetting report saved to %s", path)
    return path
