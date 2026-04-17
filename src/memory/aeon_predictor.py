"""AeonPredictor — JEPA-inspired latent predictor on top of AeonSleep.

Adds a small numpy MLP that learns h_t -> h_{t+1} from the temporal
edges of TraceGraph. No torch, no sklearn — same pattern as
ForgettingGate so this runs on GrosMac M5 / 16 GB and on CI.

Public API:
    AeonPredictor(palace, config)
        .ingest_latent(turn_id, h, ts, stack_id=None)
        .predict_next(h_t, horizon=1, stack_id=None) -> np.ndarray
        .recall(query_vec, top_k=10)          # delegates to palace
        .fit_on_buffer(lr=1e-3, epochs=1, batch_size=32)
        .ready -> bool

    LatentMLP(dim, hidden, n_stacks)
        .forward(x, stack_onehot) -> h_hat
        .backward_cosine(x, stack_onehot, target) -> float  # returns loss

    PredictorConfig(frozen dataclass)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.memory.aeonsleep import AeonSleep


@dataclass(frozen=True)
class PredictorConfig:
    """Immutable predictor config."""

    dim: int
    hidden: int = 256
    horizon: int = 1
    n_stacks: int = 16
    cold_start_threshold: int = 500
    seed: int = 0


class LatentMLP:
    """2-layer numpy MLP with skip connection (h_hat = skip(x) + mlp(x))."""

    def __init__(self, dim: int, hidden: int, n_stacks: int, seed: int = 0) -> None:
        raise NotImplementedError("Task 2")


class AeonPredictor:
    """Facade wrapping AeonSleep with a latent predictor."""

    def __init__(self, palace: "AeonSleep", config: PredictorConfig) -> None:
        raise NotImplementedError("Task 5")
