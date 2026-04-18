"""Span masking for Text-JEPA — contiguous 3-5 token blocks."""
from __future__ import annotations

import numpy as np


def span_mask(
    seq_len: int,
    mask_ratio: float,
    min_span: int,
    max_span: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a boolean mask of shape (seq_len,) with contiguous spans.

    Args:
        seq_len: Sequence length.
        mask_ratio: Target fraction of tokens to mask in [0, 1].
        min_span: Minimum contiguous span length (inclusive).
        max_span: Maximum contiguous span length (inclusive).
        rng: NumPy random generator for reproducibility.

    Returns:
        Boolean ndarray, True where token is masked.
    """
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
    if min_span < 1 or max_span < min_span:
        raise ValueError(f"invalid span bounds: min={min_span}, max={max_span}")

    mask = np.zeros(seq_len, dtype=np.bool_)
    target_masked = int(mask_ratio * seq_len)
    masked = 0
    guard = 0
    max_guard = seq_len * 4

    while masked < target_masked and guard < max_guard:
        span_len = int(rng.integers(min_span, max_span + 1))
        span_len = min(span_len, seq_len)
        start = int(rng.integers(0, seq_len - span_len + 1))
        end = start + span_len

        # Only count new masks
        newly_masked = int((~mask[start:end]).sum())
        mask[start:end] = True
        masked += newly_masked
        guard += 1

    return mask
