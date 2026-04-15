"""Forgetting gate — small MLP scoring P(keep) per episode.

Pure-Python numpy MLP with two hidden layers. Input features per
episode:

* ``age``          — hours since episode was written.
* ``access_count`` — recall hits over the episode's lifetime.
* ``conflict``     — SleepTagger conflict_level in [0, 1].
* ``emb_norm``     — L2 norm of the episode's embedding (proxy for
  signal strength; stable across length).

Output: probability that the episode is worth keeping after the
next sleep cycle. Trained with binary cross-entropy on synthetic
PI-style pairs (see :mod:`scripts.train_forgetting_gate`).

The training loop is deliberately tiny — no torch, no scikit, just
numpy. That keeps the module runnable on the GrosMac R&D box and on
CI containers.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

FEATURE_NAMES = ("age", "access_count", "conflict", "emb_norm")
N_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


@dataclass
class EpisodeFeatures:
    """Minimal feature vector consumed by :class:`ForgettingGate`."""

    age_hours: float
    access_count: int
    conflict_level: float
    embedding_norm: float

    def as_array(self) -> np.ndarray:
        # Fixed-scale normalisation so the small MLP trains stably without
        # per-run statistics. age_hours/168 scales a week to 1.0;
        # access_count/10 saturates gently; log1p clamps outliers.
        return np.array(
            [
                math.log1p(max(self.age_hours, 0.0)) / 7.0,
                math.log1p(max(self.access_count, 0)) / 3.0,
                float(self.conflict_level),
                float(self.embedding_norm),
            ],
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Stable sigmoid.
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


@dataclass
class MLPParams:
    """Packed parameters for a 2-hidden-layer MLP."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray

    @classmethod
    def init_random(
        cls, input_dim: int, hidden: int = 16, seed: int = 0
    ) -> "MLPParams":
        rng = np.random.default_rng(seed)
        scale = lambda fan_in: math.sqrt(2.0 / fan_in)
        return cls(
            w1=rng.standard_normal((input_dim, hidden)).astype(np.float32)
            * scale(input_dim),
            b1=np.zeros(hidden, dtype=np.float32),
            w2=rng.standard_normal((hidden, hidden)).astype(np.float32)
            * scale(hidden),
            b2=np.zeros(hidden, dtype=np.float32),
            w3=rng.standard_normal((hidden, 1)).astype(np.float32)
            * scale(hidden),
            b3=np.zeros(1, dtype=np.float32),
        )

    def save(self, path: Path | str) -> None:
        np.savez(
            str(path),
            w1=self.w1, b1=self.b1,
            w2=self.w2, b2=self.b2,
            w3=self.w3, b3=self.b3,
        )

    @classmethod
    def load(cls, path: Path | str) -> "MLPParams":
        data = np.load(str(path))
        return cls(**{k: data[k] for k in ("w1", "b1", "w2", "b2", "w3", "b3")})


class ForgettingGate:
    """Small feed-forward classifier predicting P(keep)."""

    def __init__(
        self, *, hidden: int = 16, seed: int = 0,
        params: MLPParams | None = None,
    ) -> None:
        self.hidden = hidden
        self.params = params or MLPParams.init_random(
            N_FEATURES, hidden=hidden, seed=seed
        )

    # ---------------------------------------------------------------- fwd

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        z1 = np.clip(x @ self.params.w1 + self.params.b1, -30.0, 30.0)
        h1 = _relu(z1)
        z2 = np.clip(h1 @ self.params.w2 + self.params.b2, -30.0, 30.0)
        h2 = _relu(z2)
        z3 = np.clip(h2 @ self.params.w3 + self.params.b3, -30.0, 30.0)
        p = _sigmoid(z3).reshape(-1)
        cache = {"x": x, "h1": h1, "h2": h2, "z1": z1, "z2": z2}
        return p, cache

    def predict_proba(
        self, features: Sequence[EpisodeFeatures] | np.ndarray
    ) -> np.ndarray:
        x = self._batch(features)
        p, _ = self._forward(x)
        return p

    def predict(
        self,
        features: Sequence[EpisodeFeatures] | np.ndarray,
        *, threshold: float = 0.5,
    ) -> np.ndarray:
        return (self.predict_proba(features) >= threshold).astype(np.int32)

    # ---------------------------------------------------------------- fit

    def fit(
        self,
        features: Sequence[EpisodeFeatures] | np.ndarray,
        labels: Sequence[int] | np.ndarray,
        *,
        lr: float = 0.05,
        epochs: int = 300,
        batch_size: int = 64,
        seed: int = 0,
        verbose: bool = False,
    ) -> list[float]:
        """Train with mini-batch SGD + BCE loss. Returns loss history."""
        x_all = self._batch(features)
        y_all = np.asarray(labels, dtype=np.float32).reshape(-1)
        if x_all.shape[0] != y_all.shape[0]:
            raise ValueError("features / labels size mismatch")
        rng = np.random.default_rng(seed)
        n = x_all.shape[0]
        history: list[float] = []
        for epoch in range(epochs):
            idx = rng.permutation(n)
            losses: list[float] = []
            for start in range(0, n, batch_size):
                b = idx[start:start + batch_size]
                xb = x_all[b]
                yb = y_all[b]
                p, cache = self._forward(xb)
                eps = 1e-7
                loss = -np.mean(
                    yb * np.log(p + eps)
                    + (1 - yb) * np.log(1 - p + eps)
                )
                losses.append(float(loss))
                self._step(cache, p, yb, lr=lr)
            history.append(float(np.mean(losses)))
            if verbose and epoch % 50 == 0:  # pragma: no cover
                print(f"epoch={epoch} loss={history[-1]:.4f}")
        return history

    def _step(
        self, cache: dict, p: np.ndarray, y: np.ndarray, *, lr: float
    ) -> None:
        x = cache["x"]
        h1 = cache["h1"]
        h2 = cache["h2"]
        z1 = cache["z1"]
        z2 = cache["z2"]
        batch = x.shape[0]

        # dL/dz3 = p - y  (BCE + sigmoid)
        dz3 = (p - y).reshape(-1, 1) / batch
        dw3 = h2.T @ dz3
        db3 = dz3.sum(axis=0)
        dh2 = dz3 @ self.params.w3.T
        dz2 = dh2 * (z2 > 0)
        dw2 = h1.T @ dz2
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.params.w2.T
        dz1 = dh1 * (z1 > 0)
        dw1 = x.T @ dz1
        db1 = dz1.sum(axis=0)

        # Clip gradients to keep the pure-Python trainer stable across
        # runs. The MLP is tiny (4->16->16->1) so clipping rarely fires
        # but it avoids overflow warnings in edge cases.
        clip = 5.0
        for g in (dw1, db1, dw2, db2, dw3, db3):
            np.clip(g, -clip, clip, out=g)

        self.params.w1 -= lr * dw1
        self.params.b1 -= lr * db1
        self.params.w2 -= lr * dw2
        self.params.b2 -= lr * db2
        self.params.w3 -= lr * dw3
        self.params.b3 -= lr * db3

    # --------------------------------------------------------------- utils

    @staticmethod
    def _batch(features) -> np.ndarray:
        if isinstance(features, np.ndarray):
            return features.astype(np.float32, copy=False)
        rows = [f.as_array() for f in features]
        if not rows:
            return np.zeros((0, N_FEATURES), dtype=np.float32)
        return np.stack(rows).astype(np.float32)


# ---------------------------------------------------------------------------
# F1 metric
# ---------------------------------------------------------------------------


def f1_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_pred_arr = np.asarray(y_pred, dtype=np.int32)
    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Synthetic dataset generator (used by tests and the training script)
# ---------------------------------------------------------------------------


def generate_synthetic_pairs(
    n: int = 2000, seed: int = 0
) -> tuple[list[EpisodeFeatures], list[int]]:
    """Generate PI-style (age/access/conflict/norm → keep) pairs."""
    rng = np.random.default_rng(seed)
    feats: list[EpisodeFeatures] = []
    labels: list[int] = []
    for _ in range(n):
        keep = rng.random() < 0.5
        if keep:
            age = float(rng.uniform(0, 200))
            access = int(rng.integers(2, 20))
            conflict = float(rng.uniform(0.0, 0.3))
            norm = float(rng.uniform(0.8, 1.2))
        else:
            age = float(rng.uniform(150, 800))
            access = int(rng.integers(0, 2))
            conflict = float(rng.uniform(0.5, 1.0))
            norm = float(rng.uniform(0.2, 0.8))
        feats.append(
            EpisodeFeatures(
                age_hours=age,
                access_count=access,
                conflict_level=conflict,
                embedding_norm=norm,
            )
        )
        labels.append(1 if keep else 0)
    return feats, labels


# ---------------------------------------------------------------------------
# JSONL IO (used by the trainer script and tests)
# ---------------------------------------------------------------------------


def write_jsonl(path: Path | str, feats: Sequence[EpisodeFeatures], labels: Sequence[int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for f, y in zip(feats, labels):
            row = {
                "age_hours": f.age_hours,
                "access_count": f.access_count,
                "conflict_level": f.conflict_level,
                "embedding_norm": f.embedding_norm,
                "label": int(y),
            }
            fh.write(json.dumps(row) + "\n")


def read_jsonl(path: Path | str) -> tuple[list[EpisodeFeatures], list[int]]:
    path = Path(path)
    feats: list[EpisodeFeatures] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            feats.append(
                EpisodeFeatures(
                    age_hours=float(row["age_hours"]),
                    access_count=int(row["access_count"]),
                    conflict_level=float(row["conflict_level"]),
                    embedding_norm=float(row["embedding_norm"]),
                )
            )
            labels.append(int(row["label"]))
    return feats, labels
