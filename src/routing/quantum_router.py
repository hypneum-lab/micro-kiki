"""Quantum VQC Router — PennyLane variational circuit for domain classification.

Uses 6 qubits to classify text embeddings into 35 domain classes (34 niches + base).
Runs on classical simulator by default; QPU dispatch optional.

Part of the triple-hybrid architecture:
  Quantum VQC Router → SNN SpikingKiki → Classical 35B fallback
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.routing.model_router import RouteDecision
from src.routing.router import NICHE_DOMAINS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PennyLane optional import — graceful fallback if not installed
# ---------------------------------------------------------------------------

try:
    import pennylane as qml  # type: ignore[import]
    _PENNYLANE_AVAILABLE = True
    logger.debug("PennyLane %s loaded", qml.__version__)
except ImportError:  # pragma: no cover
    qml = None  # type: ignore[assignment]
    _PENNYLANE_AVAILABLE = False
    logger.warning("PennyLane not available — QuantumRouter disabled (install pennylane)")

# ---------------------------------------------------------------------------
# Ordered domain list (34 niches + "base" at index 34)
# ---------------------------------------------------------------------------

_NICHE_DOMAIN_LIST: list[str] = sorted(NICHE_DOMAINS)  # 34 entries
_ALL_DOMAINS: list[str] = _NICHE_DOMAIN_LIST + ["base"]  # 35 entries


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantumRouterConfig:
    """Frozen configuration for the Quantum VQC Router.

    Args:
        n_qubits: Number of qubits in the variational circuit (default 6).
            With 35 domains, ceil(log2(35)) = 6 qubits minimum.
        n_layers: Number of strongly-entangling layers (default 6).
        n_classes: Number of output classes — 35 (34 niches + base).
        learning_rate: Gradient descent step size.
        device: PennyLane device name (default "default.qubit" = classical simulator).
    """

    n_qubits: int = 6
    n_layers: int = 6
    n_classes: int = 35
    learning_rate: float = 0.01
    device: str = "default.qubit"


# ---------------------------------------------------------------------------
# Quantum Router
# ---------------------------------------------------------------------------


class QuantumRouter:
    """VQC-based domain classifier for micro-kiki routing.

    Architecture:
    1. Encode embedding → n_qubits angles (AngleEmbedding on first n_qubits features)
    2. Variational layers (StronglyEntanglingLayers)
    3. Measure expectation value of PauliZ on each qubit → n_qubits scalars
    4. Classical linear layer (n_qubits → n_classes) for final classification

    Requires PennyLane to be installed. Raises ``ImportError`` at instantiation
    if PennyLane is unavailable.
    """

    def __init__(self, config: QuantumRouterConfig | None = None) -> None:
        if not _PENNYLANE_AVAILABLE:
            raise ImportError(
                "PennyLane is required for QuantumRouter. "
                "Install it with: uv add pennylane"
            )
        self.config = config or QuantumRouterConfig()
        self._dev = qml.device(self.config.device, wires=self.config.n_qubits)
        self._qnode = qml.QNode(self._circuit_fn, self._dev, diff_method="best")

        # Variational weights: shape (n_layers, n_qubits, 3)
        # StronglyEntanglingLayers needs 3 rotation angles per qubit per layer
        weight_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=self.config.n_layers,
            n_wires=self.config.n_qubits,
        )
        rng = np.random.default_rng(42)
        self.weights: np.ndarray = rng.uniform(
            low=0.0, high=2 * np.pi, size=weight_shape
        )

        # Classical linear head: (n_qubits,) → (n_classes,)
        rng2 = np.random.default_rng(43)
        self.linear_w: np.ndarray = rng2.standard_normal(
            (self.config.n_qubits, self.config.n_classes)
        ) * 0.1
        self.linear_b: np.ndarray = np.zeros(self.config.n_classes)

    # ------------------------------------------------------------------
    # Internal circuit (called via QNode)
    # ------------------------------------------------------------------

    def _circuit_fn(self, weights: np.ndarray, features: np.ndarray) -> list[Any]:
        """Variational circuit: encode + entangle + measure."""
        # AngleEmbedding uses first n_qubits features as rotation angles
        qml.AngleEmbedding(features[: self.config.n_qubits], wires=range(self.config.n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(self.config.n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(self.config.n_qubits)]

    def circuit(self, weights: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Execute VQC and return n_qubits expectation values.

        Args:
            weights: Shape (n_layers, n_qubits, 3).
            features: Embedding vector; only first n_qubits dims are encoded.

        Returns:
            np.ndarray of shape (n_qubits,) with values in [-1, 1].
        """
        result = self._qnode(weights, features)
        return np.array(result)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, embedding: np.ndarray) -> RouteDecision:
        """Classify an embedding into a domain and return a RouteDecision.

        Args:
            embedding: 1D float array, any length ≥ n_qubits.

        Returns:
            RouteDecision with model_id, adapter, and reason.
        """
        qubits = self.circuit(self.weights, embedding)  # (n_qubits,)
        logits = qubits @ self.linear_w + self.linear_b  # (n_classes,)
        probs = _softmax(logits)
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        domain = _ALL_DOMAINS[class_idx]

        if domain == "base":
            return RouteDecision(
                model_id="qwen35b",
                adapter=None,
                reason=f"quantum-vqc: base fallback (conf={confidence:.3f})",
            )

        return RouteDecision(
            model_id="qwen35b",
            adapter=f"stack-{domain}",
            reason=f"quantum-vqc: {domain} (conf={confidence:.3f})",
        )

    def train(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        epochs: int = 50,
    ) -> list[float]:
        """Train VQC weights + linear head via gradient descent.

        Args:
            embeddings: Shape (n_samples, embedding_dim).
            labels: Integer class indices, shape (n_samples,).
            epochs: Number of full-dataset passes.

        Returns:
            List of per-epoch cross-entropy loss values.
        """
        lr = self.config.learning_rate
        losses: list[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for emb, label in zip(embeddings, labels):
                qubits = self.circuit(self.weights, emb)
                logits = qubits @ self.linear_w + self.linear_b
                probs = _softmax(logits)
                loss = -float(np.log(probs[int(label)] + 1e-12))
                epoch_loss += loss

                # Gradient for linear head (analytical)
                d_logits = probs.copy()
                d_logits[int(label)] -= 1.0  # (n_classes,)
                self.linear_w -= lr * np.outer(qubits, d_logits)
                self.linear_b -= lr * d_logits

                # Gradient for VQC weights via parameter-shift rule
                shift = np.pi / 2
                for idx in np.ndindex(*self.weights.shape):
                    w_plus = self.weights.copy()
                    w_plus[idx] += shift
                    w_minus = self.weights.copy()
                    w_minus[idx] -= shift

                    q_plus = self.circuit(w_plus, emb)
                    q_minus = self.circuit(w_minus, emb)

                    # Chain rule: dL/dw_i = (dL/dq) · (dq/dw_i)
                    # dq/dw_i ≈ (q_plus - q_minus) / 2  (parameter-shift)
                    dq_dw = (q_plus - q_minus) / 2.0  # (n_qubits,)

                    # dL/dq via chain through linear + softmax
                    # dL/dlogit = probs - one_hot; dlogit/dq = linear_w[qubit, :]
                    dl_dq = np.dot(self.linear_w, d_logits)  # (n_qubits,)
                    grad = float(np.dot(dl_dq, dq_dw))
                    self.weights[idx] -= lr * grad

            avg_loss = epoch_loss / max(len(embeddings), 1)
            losses.append(avg_loss)
            logger.debug("epoch %d/%d  loss=%.4f", epoch + 1, epochs, avg_loss)

        return losses

    def save(self, path: str | Path) -> None:
        """Serialize weights and linear head to a .npz file.

        Args:
            path: Destination path (`.npz` will be appended if missing).
        """
        path = Path(path)
        np.savez(
            path,
            weights=self.weights,
            linear_w=self.linear_w,
            linear_b=self.linear_b,
            n_qubits=self.config.n_qubits,
            n_layers=self.config.n_layers,
        )
        logger.info("QuantumRouter saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load weights from a .npz file saved by :meth:`save`.

        Args:
            path: Source path (`.npz` suffix handled automatically).
        """
        path = Path(path)
        # numpy appends .npz when saving without extension
        load_path = path if path.suffix == ".npz" else Path(str(path) + ".npz")
        data = np.load(load_path)
        self.weights = data["weights"]
        self.linear_w = data["linear_w"]
        self.linear_b = data["linear_b"]
        logger.info("QuantumRouter loaded from %s", load_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()
