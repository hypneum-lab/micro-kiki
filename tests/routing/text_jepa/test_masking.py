"""Span-masking tests — contiguous token blocks, no random single-token masks."""
from __future__ import annotations

import numpy as np
import pytest


def test_span_mask_returns_bool_array_of_seq_len():
    from src.routing.text_jepa.masking import span_mask

    rng = np.random.default_rng(0)
    mask = span_mask(seq_len=32, mask_ratio=0.4, min_span=3, max_span=5, rng=rng)

    assert mask.dtype == np.bool_
    assert mask.shape == (32,)


def test_span_mask_ratio_within_tolerance():
    from src.routing.text_jepa.masking import span_mask

    rng = np.random.default_rng(1)
    mask = span_mask(seq_len=128, mask_ratio=0.4, min_span=3, max_span=5, rng=rng)

    ratio = mask.mean()
    assert 0.30 <= ratio <= 0.55  # allow ±0.15 — short sequences bounce


def test_span_mask_uses_contiguous_spans_only():
    """No isolated masked token surrounded by unmasked — each mask run >= min_span."""
    from src.routing.text_jepa.masking import span_mask

    rng = np.random.default_rng(2)
    mask = span_mask(seq_len=64, mask_ratio=0.4, min_span=3, max_span=5, rng=rng)

    # Find runs of True
    runs = []
    current = 0
    for bit in mask:
        if bit:
            current += 1
        elif current > 0:
            runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)

    assert all(r >= 3 for r in runs), f"Found span shorter than 3: {runs}"


def test_span_mask_rejects_invalid_ratio():
    from src.routing.text_jepa.masking import span_mask

    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        span_mask(seq_len=32, mask_ratio=1.5, min_span=3, max_span=5, rng=rng)


def test_span_mask_seed_reproducible():
    from src.routing.text_jepa.masking import span_mask

    a = span_mask(seq_len=48, mask_ratio=0.4, min_span=3, max_span=5, rng=np.random.default_rng(42))
    b = span_mask(seq_len=48, mask_ratio=0.4, min_span=3, max_span=5, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
