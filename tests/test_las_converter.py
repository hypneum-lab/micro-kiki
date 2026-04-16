"""Tests for the LAS ANN→SNN converter (story-17)."""

from __future__ import annotations

import numpy as np
import pytest

from src.spiking.las_converter import (
    LASConverter,
    SpikingLinear,
    convert_linear,
    verify_equivalence,
)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _ann_linear(w: np.ndarray, b: np.ndarray | None, x: np.ndarray) -> np.ndarray:
    z = x @ w.T
    if b is not None:
        z = z + b
    return _relu(z)


def test_convert_linear_identity_recovery() -> None:
    """Converting a Linear(8,8) with unit max_rate matches the ANN.

    With T=64 timesteps the rate-code quantisation error is ~1/64 per
    element. On a small Linear(8,8) with bounded inputs we expect the
    relative L2 error to sit well under 5%.
    """
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 8)).astype(np.float64) * 0.3
    b = rng.standard_normal(8).astype(np.float64) * 0.05
    snn = convert_linear({"weight": w, "bias": b}, timesteps=64, max_rate=1.0)

    x = np.clip(rng.standard_normal((4, 8)) * 0.3, -0.5, 0.5)
    snn_out = snn(x)
    ann_out = _ann_linear(w, b, x)

    rel_err = np.linalg.norm(snn_out - ann_out) / (np.linalg.norm(ann_out) + 1e-12)
    assert rel_err < 0.05, f"rel_err={rel_err:.4f}"


def test_spiking_linear_shape_and_validation() -> None:
    w = np.zeros((4, 3))
    with pytest.raises(ValueError):
        SpikingLinear(weight=w.reshape(-1), bias=None)
    layer = SpikingLinear(weight=w, bias=None, timesteps=8)
    assert layer.in_features == 3
    assert layer.out_features == 4
    with pytest.raises(ValueError):
        layer.forward(np.zeros(5))  # wrong in-features


def test_convert_model_on_2layer_mlp_preserves_logits() -> None:
    """A 2-layer ReLU MLP keeps its logits within 5% after LAS."""
    rng = np.random.default_rng(1)
    w1 = rng.standard_normal((16, 8)) * 0.25
    b1 = np.zeros(16)
    w2 = rng.standard_normal((4, 16)) * 0.25
    b2 = np.zeros(4)

    def ann_forward(x: np.ndarray) -> np.ndarray:
        h = _relu(x @ w1.T + b1)
        return _relu(h @ w2.T + b2)

    # Cascaded layers compound quantisation error; raise T to keep the
    # 5% story-17 acceptance bar on the full logits.
    converter = LASConverter(timesteps=256, max_rate=1.0)
    snn = converter.convert_model(
        [
            {"weight": w1, "bias": b1},
            {"weight": w2, "bias": b2},
        ]
    )

    x = np.clip(rng.standard_normal((3, 8)) * 0.25, -0.5, 0.5)
    snn_out = snn(x)
    ann_out = ann_forward(x)

    rel_err = np.linalg.norm(snn_out - ann_out) / (np.linalg.norm(ann_out) + 1e-12)
    assert rel_err < 0.05, f"rel_err={rel_err:.4f}"


def test_verify_equivalence_true_for_converted_linear() -> None:
    rng = np.random.default_rng(2)
    w = rng.standard_normal((6, 4)) * 0.2
    b = np.zeros(6)
    snn = convert_linear({"weight": w, "bias": b}, timesteps=256)

    def ann(x: np.ndarray) -> np.ndarray:
        return _ann_linear(w, b, x)

    sample = np.clip(rng.standard_normal((2, 4)) * 0.25, -0.5, 0.5)
    assert verify_equivalence(ann, snn, sample, tol=0.05)


def test_verify_equivalence_false_when_weights_mismatch() -> None:
    """If ann and snn differ materially, verify returns False."""
    rng = np.random.default_rng(3)
    w_snn = rng.standard_normal((6, 4)) * 0.2
    w_ann = rng.standard_normal((6, 4)) * 0.2  # independent weights
    b = np.zeros(6)
    snn = convert_linear({"weight": w_snn, "bias": b}, timesteps=64)

    def ann(x: np.ndarray) -> np.ndarray:
        return _ann_linear(w_ann, b, x)

    sample = np.clip(rng.standard_normal((2, 4)) * 0.25, -0.5, 0.5)
    # Tight tolerance should reject — independent weights cannot match.
    assert not verify_equivalence(ann, snn, sample, tol=1e-3)


def test_converter_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        LASConverter(timesteps=0)
    with pytest.raises(ValueError):
        LASConverter(max_rate=-1.0)
