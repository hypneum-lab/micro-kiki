"""Tests for the forgetting-gate MLP (story-9)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.cognitive.forgetting_gate import (
    EpisodeFeatures,
    ForgettingGate,
    MLPParams,
    N_FEATURES,
    f1_score,
    generate_synthetic_pairs,
    read_jsonl,
    write_jsonl,
)


def test_feature_array_shape() -> None:
    f = EpisodeFeatures(
        age_hours=3.0, access_count=4,
        conflict_level=0.2, embedding_norm=1.1,
    )
    arr = f.as_array()
    assert arr.shape == (N_FEATURES,)
    assert arr.dtype == np.float32


def test_forward_shape() -> None:
    gate = ForgettingGate(seed=0)
    feats, _ = generate_synthetic_pairs(n=10, seed=1)
    proba = gate.predict_proba(feats)
    assert proba.shape == (10,)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_training_meets_f1_threshold() -> None:
    """Acceptance: test-set F1 >= 0.85 after training."""
    feats, labels = generate_synthetic_pairs(n=2000, seed=0)
    split = int(0.8 * len(feats))
    gate = ForgettingGate(seed=0)
    gate.fit(
        feats[:split], labels[:split],
        epochs=200, batch_size=64, lr=0.05, seed=0,
    )
    preds = gate.predict(feats[split:])
    score = f1_score(labels[split:], preds)
    assert score >= 0.85, f"F1 {score:.3f} below 0.85"


def test_training_loss_decreases() -> None:
    feats, labels = generate_synthetic_pairs(n=300, seed=2)
    gate = ForgettingGate(seed=0)
    history = gate.fit(feats, labels, epochs=120, batch_size=32, lr=0.05)
    assert history[-1] < history[0]


def test_save_and_load_params(tmp_path: Path) -> None:
    gate = ForgettingGate(seed=1)
    feats, labels = generate_synthetic_pairs(n=200, seed=3)
    gate.fit(feats, labels, epochs=50, batch_size=32)
    path = tmp_path / "params.npz"
    gate.params.save(path)
    loaded = MLPParams.load(path)
    np.testing.assert_allclose(loaded.w1, gate.params.w1)
    reload_gate = ForgettingGate(seed=1, params=loaded)
    np.testing.assert_allclose(
        reload_gate.predict_proba(feats[:5]),
        gate.predict_proba(feats[:5]),
    )


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    feats, labels = generate_synthetic_pairs(n=20, seed=7)
    path = tmp_path / "pairs.jsonl"
    write_jsonl(path, feats, labels)
    feats2, labels2 = read_jsonl(path)
    assert labels == labels2
    for a, b in zip(feats, feats2):
        assert a.as_array().tolist() == pytest.approx(b.as_array().tolist())
