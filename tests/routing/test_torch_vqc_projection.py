"""Projection tests: learned 384→n_qubits layer rescues accuracy on hard tasks."""
from __future__ import annotations

import numpy as np
import torch


def _make_hard_task(n_classes: int = 5, samples_per_class: int = 40, input_dim: int = 64, seed: int = 0):
    """Class info lives ONLY in dims [input_dim-6 : input_dim], so truncation to dims [0:n_qubits] sees pure noise."""
    rng = np.random.default_rng(seed)
    class_centers = rng.uniform(0.3, np.pi - 0.3, size=(n_classes, 6))  # mild, in rotation range

    X, y = [], []
    for c in range(n_classes):
        for _ in range(samples_per_class):
            x = rng.normal(0, 0.5, size=input_dim)  # noise everywhere
            x[-6:] = class_centers[c] + rng.normal(0, 0.08, size=6)  # signal in last 6 dims
            X.append(x)
            y.append(c)
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def test_projection_beats_no_projection_on_hard_task():
    """Signal in last 6 dims only: truncation sees pure noise, projection can find it."""
    from src.routing.torch_vqc_router import TorchVQCRouter

    X, y = _make_hard_task(n_classes=5, samples_per_class=40, input_dim=64, seed=3)
    X_t = torch.from_numpy(X).double()
    y_t = torch.from_numpy(y)

    # Without projection: VQC sees X[:, :4] (first 4 dims = pure noise here)
    model_no_proj = TorchVQCRouter(n_qubits=4, n_layers=4, n_classes=5, lr=0.05, seed=0)
    model_no_proj.train_batched(X_t, y_t, epochs=200)
    with torch.no_grad():
        acc_no_proj = (model_no_proj.predict(X_t).numpy() == y).mean()

    # With projection: a 4×64 layer can learn to pick from the informative last dims
    model_proj = TorchVQCRouter(
        n_qubits=4, n_layers=4, n_classes=5, lr=0.05, seed=0, input_dim=64
    )
    model_proj.train_batched(X_t, y_t, epochs=200)
    with torch.no_grad():
        acc_proj = (model_proj.predict(X_t).numpy() == y).mean()

    print(f"\nNo projection acc = {acc_no_proj:.3f}  (chance = {1/5:.3f})")
    print(f"With projection acc = {acc_proj:.3f}")
    assert acc_proj > acc_no_proj + 0.15, (
        f"projection should give ≥15pt boost, got {acc_no_proj:.3f} → {acc_proj:.3f}"
    )


def test_projection_forward_shape():
    """Projection doesn't break shape invariants."""
    from src.routing.torch_vqc_router import TorchVQCRouter

    model = TorchVQCRouter(n_qubits=6, n_layers=6, n_classes=10, input_dim=384, seed=0)
    x = torch.randn(32, 384, dtype=torch.float64)
    logits = model.forward(x)
    assert logits.shape == (32, 10), f"expected (32, 10), got {tuple(logits.shape)}"


def test_projection_params_counted():
    """Projection adds n_qubits × input_dim + n_qubits parameters."""
    from src.routing.torch_vqc_router import TorchVQCRouter

    no_proj = TorchVQCRouter(n_qubits=4, n_layers=6, n_classes=10, seed=0)
    with_proj = TorchVQCRouter(n_qubits=4, n_layers=6, n_classes=10, input_dim=384, seed=0)

    n_no = sum(p.numel() for p in no_proj.parameters())
    n_yes = sum(p.numel() for p in with_proj.parameters())
    expected_extra = 4 * 384 + 4  # projection_w + projection_b
    assert n_yes - n_no == expected_extra, (
        f"expected {expected_extra} extra params, got {n_yes - n_no}"
    )
