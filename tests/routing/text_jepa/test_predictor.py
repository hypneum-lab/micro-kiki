"""Tests for the tiny Predictor module."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_predictor_forward_shape():
    from src.routing.text_jepa.predictor import Predictor

    pred = Predictor(latent_dim=128, hidden_dim=32)
    x = torch.randn(4, 16, 128)
    out = pred(x)
    assert out.shape == (4, 16, 128)


def test_predictor_smaller_than_student():
    """Plan spec: predictor ~1/10 of student params."""
    from src.routing.text_jepa.encoder import StudentEncoder
    from src.routing.text_jepa.predictor import Predictor

    student = StudentEncoder(input_dim=384, hidden_dim=256, output_dim=128)
    pred = Predictor(latent_dim=128, hidden_dim=32)

    n_student = sum(p.numel() for p in student.parameters())
    n_pred = sum(p.numel() for p in pred.parameters())
    assert n_pred * 5 < n_student, f"predictor too large: {n_pred} vs student {n_student}"


def test_predictor_output_finite():
    from src.routing.text_jepa.predictor import Predictor

    pred = Predictor(latent_dim=128, hidden_dim=32)
    out = pred(torch.randn(2, 4, 128))
    assert torch.isfinite(out).all()
