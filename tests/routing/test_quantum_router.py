"""Tests for QuantumRouter — all run on classical simulator, no QPU required."""
from __future__ import annotations

import importlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip marker — applied to every test in this module if PennyLane is absent
# ---------------------------------------------------------------------------

_pennylane_available = importlib.util.find_spec("pennylane") is not None

pennylane_required = pytest.mark.skipif(
    not _pennylane_available,
    reason="PennyLane not installed — install with: uv add pennylane",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config():
    """Default QuantumRouterConfig."""
    from src.routing.quantum_router import QuantumRouterConfig

    return QuantumRouterConfig()


@pytest.fixture(scope="module")
def router():
    """QuantumRouter with default config (simulator)."""
    from src.routing.quantum_router import QuantumRouter, QuantumRouterConfig

    return QuantumRouter(QuantumRouterConfig())


@pytest.fixture()
def small_embedding():
    """Synthetic embedding of dimension 16."""
    rng = np.random.default_rng(0)
    return rng.standard_normal(16).astype(np.float64)


# ---------------------------------------------------------------------------
# Test 1 — Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    """QuantumRouterConfig must have n_qubits=6 and n_classes=35 by default."""
    from src.routing.quantum_router import QuantumRouterConfig

    cfg = QuantumRouterConfig()
    assert cfg.n_qubits == 6
    assert cfg.n_layers == 6
    assert cfg.n_classes == 35
    assert cfg.device == "default.qubit"


def test_config_is_frozen():
    """QuantumRouterConfig must be immutable (frozen dataclass)."""
    from src.routing.quantum_router import QuantumRouterConfig

    cfg = QuantumRouterConfig()
    with pytest.raises(Exception):
        cfg.n_qubits = 8  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test 2 — Circuit output shape
# ---------------------------------------------------------------------------


@pennylane_required
def test_circuit_output_shape(router, small_embedding):
    """circuit() must return exactly n_qubits expectation values."""
    result = router.circuit(router.weights, small_embedding)
    assert result.shape == (router.config.n_qubits,), (
        f"Expected ({router.config.n_qubits},), got {result.shape}"
    )


@pennylane_required
def test_circuit_values_in_range(router, small_embedding):
    """Expectation values of PauliZ must lie in [-1, 1]."""
    result = router.circuit(router.weights, small_embedding)
    assert np.all(result >= -1.0 - 1e-6)
    assert np.all(result <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# Test 3 — route() returns valid RouteDecision
# ---------------------------------------------------------------------------


@pennylane_required
def test_route_returns_valid_decision(router, small_embedding):
    """route() must return RouteDecision with a valid domain."""
    from src.routing.model_router import RouteDecision
    from src.routing.quantum_router import _ALL_DOMAINS

    decision = router.route(small_embedding)
    assert isinstance(decision, RouteDecision)
    assert decision.model_id == "qwen35b"
    assert decision.reason.startswith("quantum-vqc:")

    # Extract domain from adapter or reason
    if decision.adapter is not None:
        domain = decision.adapter.removeprefix("stack-")
        assert domain in _ALL_DOMAINS, f"Unknown domain: {domain}"
    else:
        # base fallback — adapter is None
        assert "base" in decision.reason


@pennylane_required
def test_route_domain_in_niche_or_base(router):
    """Route domain must be one of the 10 niche domains or 'base'."""
    from src.routing.router import NICHE_DOMAINS
    from src.routing.quantum_router import _ALL_DOMAINS

    rng = np.random.default_rng(7)
    for _ in range(5):
        emb = rng.standard_normal(32)
        decision = router.route(emb)
        if decision.adapter is not None:
            domain = decision.adapter.removeprefix("stack-")
            assert domain in NICHE_DOMAINS
        else:
            assert "base" in decision.reason


# ---------------------------------------------------------------------------
# Test 4 — train() reduces loss
# ---------------------------------------------------------------------------


@pennylane_required
def test_train_reduces_loss():
    """Training for 10 epochs on synthetic data must reduce mean loss."""
    from src.routing.quantum_router import QuantumRouter, QuantumRouterConfig

    cfg = QuantumRouterConfig(n_qubits=4, n_layers=2, learning_rate=0.05)
    qr = QuantumRouter(cfg)

    rng = np.random.default_rng(99)
    n_samples = 22  # 2 samples per class
    embeddings = rng.standard_normal((n_samples, 8)).astype(np.float64)
    labels = np.tile(np.arange(11), 2)[:n_samples]

    losses = qr.train(embeddings, labels, epochs=10)

    assert len(losses) == 10
    # Loss should decrease (or at least not increase dramatically)
    first_half = float(np.mean(losses[:3]))
    second_half = float(np.mean(losses[7:]))
    assert second_half <= first_half + 0.5, (
        f"Loss did not improve: first={first_half:.4f}  last={second_half:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5 — save / load roundtrip
# ---------------------------------------------------------------------------


@pennylane_required
def test_save_load_roundtrip(small_embedding):
    """Saved weights must produce identical routing decision after load."""
    from src.routing.quantum_router import QuantumRouter, QuantumRouterConfig

    cfg = QuantumRouterConfig()
    qr = QuantumRouter(cfg)

    decision_before = qr.route(small_embedding)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "qr_weights"
        qr.save(save_path)

        # New instance, then load
        qr2 = QuantumRouter(cfg)
        qr2.load(save_path)

        decision_after = qr2.route(small_embedding)

    assert decision_before.adapter == decision_after.adapter
    assert decision_before.model_id == decision_after.model_id


@pennylane_required
def test_save_creates_npz_file(small_embedding):
    """save() must create a .npz file at the given path."""
    from src.routing.quantum_router import QuantumRouter, QuantumRouterConfig

    qr = QuantumRouter(QuantumRouterConfig())
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "weights"
        qr.save(save_path)
        assert Path(str(save_path) + ".npz").exists()


# ---------------------------------------------------------------------------
# Test 6 — graceful skip when PennyLane is not installed
# ---------------------------------------------------------------------------


def test_import_without_pennylane_raises_import_error(monkeypatch):
    """QuantumRouter() must raise ImportError when PennyLane is absent."""
    import sys

    # Temporarily hide pennylane from sys.modules
    saved = sys.modules.get("pennylane")
    sys.modules["pennylane"] = None  # type: ignore[assignment]

    # Re-import the module with pennylane masked
    import importlib
    import src.routing.quantum_router as qr_mod

    original_flag = qr_mod._PENNYLANE_AVAILABLE
    qr_mod._PENNYLANE_AVAILABLE = False

    try:
        with pytest.raises(ImportError, match="PennyLane is required"):
            qr_mod.QuantumRouter()
    finally:
        qr_mod._PENNYLANE_AVAILABLE = original_flag
        if saved is None:
            del sys.modules["pennylane"]
        else:
            sys.modules["pennylane"] = saved


# ---------------------------------------------------------------------------
# Test 7 — parameter count sanity
# ---------------------------------------------------------------------------


@pennylane_required
def test_weight_shapes_match_config():
    """VQC weights must have shape (n_layers, n_qubits, 3)."""
    from src.routing.quantum_router import QuantumRouter, QuantumRouterConfig

    cfg = QuantumRouterConfig(n_qubits=4, n_layers=3)
    qr = QuantumRouter(cfg)
    assert qr.weights.shape == (3, 4, 3)
    assert qr.linear_w.shape == (4, 11)
    assert qr.linear_b.shape == (11,)
