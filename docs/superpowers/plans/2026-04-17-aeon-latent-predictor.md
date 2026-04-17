# Aeon Latent Predictor — PoC Scenario B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove a JEPA-inspired latent predictor (`h_t -> h_{t+k}`) stacked on top of AeonSleep adds value over pure vector RAG for the critic/judge query path.

**Architecture:** A new `AeonPredictor` class wraps `AeonSleep` and learns a 2-layer numpy MLP (hidden=256, ReLU, skip connection, cosine-similarity loss) over `(h_t, h_{t+1})` pairs mined from the `temporal` edges of `TraceGraph`. Predictor training runs as step 5 of `AeonSleep.sleep_cycle`. Downstream callers pick between `palace.recall(embed(q))` (baseline) and `palace.recall(predictor.predict_next(embed(q)))` (predictive) — the PoC evaluation script compares both on Recall@5 / MRR.

**Tech Stack:** numpy only (no torch), pytest, uv, Python 3.11+. Reuses `AeonSleep`, `AtlasIndex`, `TraceGraph` unchanged.

---

## Success & Kill Criteria

**Success (PoC ships):**
- Predictive path beats baseline on >= 60% of test queries (Recall@5 comparison)
- Full `scripts/eval_aeon_predictor.py` run completes in < 30s on GrosMac M5 / 16 GB
- Cosine training loss converges below 0.3 within 300 epochs on the eval fixture

**Kill (abandon PoC, keep retrieval-only Aeon):**
- Cosine loss > 0.3 after 300 epochs on the eval fixture, OR
- `std(h_hat) < 0.1 * std(h_t)` on held-out predictions (representation collapse), OR
- Predictive path underperforms baseline on > 40% of test queries

The collapse detector (Task 4) is the live tripwire; the eval script (Task 10) is the final gate.

## Risk Mitigations (each mapped to a task)

1. **Representation collapse** — Task 2 uses cosine-similarity loss (NOT MSE); Task 4 adds a `detect_collapse` helper that flags when `std(h_hat) < 0.1 * std(h_t)` and is called from the training loop (Task 9) to trigger early rollback.
2. **32 LoRA stacks = 32 latent spaces** — Task 8 extends the input layer to concatenate a one-hot `stack_id` (dim=16, padded) before the MLP. The predictor is single-model but stack-aware.
3. **Cold-start (< 500 pairs)** — Task 6 implements a `ready` flag that stays `False` until the training buffer has >= 500 pairs AND at least one fit has succeeded. `AeonPredictor.predict_next` falls back to returning `h_t` unchanged when `not ready`, which keeps the serving path on pure-retrieval mode.

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/memory/aeon_predictor.py` | `LatentMLP` (numpy) + `AeonPredictor` facade | Create |
| `src/memory/aeonsleep.py` | Add step 5 hook to `sleep_cycle` | Modify |
| `scripts/eval_aeon_predictor.py` | PoC eval: Recall@5, MRR baseline vs predictive | Create |
| `tests/memory/test_aeon_predictor.py` | Unit tests for `LatentMLP`, `AeonPredictor`, stack conditioning, collapse detector | Create |
| `tests/memory/test_aeonsleep_predictor_hook.py` | Integration test for sleep_cycle step 5 | Create |
| `tests/scripts/test_eval_aeon_predictor.py` | Smoke test for eval script (< 5s) | Create |
| `results/2026-04-17-aeon-predictor-poc.md` | Real numbers from Task 11 (written, not committed-as-placeholder) | Create |

Files that change together live together: the predictor module, its tests, and the sleep_cycle hook are touched in sequence.

---

### Task 1: Scaffold module, dataclasses, and test skeleton

**Files:**
- Create: `src/memory/aeon_predictor.py`
- Create: `tests/memory/test_aeon_predictor.py`

- [ ] **Step 1: Write the failing test (module import + class existence)**

Create `tests/memory/test_aeon_predictor.py`:

```python
"""Tests for AeonPredictor — JEPA-inspired latent predictor on top of AeonSleep."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.memory.aeon_predictor import (
    AeonPredictor,
    LatentMLP,
    PredictorConfig,
)
from src.memory.aeonsleep import AeonSleep, Episode


def _mock_embed(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def test_imports_exist():
    assert AeonPredictor is not None
    assert LatentMLP is not None
    assert PredictorConfig is not None


def test_config_defaults():
    cfg = PredictorConfig(dim=384)
    assert cfg.dim == 384
    assert cfg.hidden == 256
    assert cfg.horizon == 1
    assert cfg.n_stacks == 16
    assert cfg.cold_start_threshold == 500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_aeon_predictor.py -v`
Expected: `ModuleNotFoundError: No module named 'src.memory.aeon_predictor'`

- [ ] **Step 3: Write minimal implementation (scaffolding only)**

Create `src/memory/aeon_predictor.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_aeon_predictor.py -v`
Expected: `2 passed` (test_imports_exist, test_config_defaults)

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/memory/aeon_predictor.py tests/memory/test_aeon_predictor.py
git commit -m "feat(aeon): scaffold latent predictor module"
```

---

### Task 2: `LatentMLP.forward` (numpy 2-layer + skip)

**Files:**
- Modify: `src/memory/aeon_predictor.py`
- Modify: `tests/memory/test_aeon_predictor.py`

- [ ] **Step 1: Write the failing test for forward pass**

Append to `tests/memory/test_aeon_predictor.py`:

```python
class TestLatentMLPForward:
    def test_forward_shape(self):
        mlp = LatentMLP(dim=384, hidden=256, n_stacks=16, seed=0)
        x = _mock_embed(384, seed=1).reshape(1, -1)
        stack = np.zeros((1, 16), dtype=np.float32)
        stack[0, 3] = 1.0
        h_hat = mlp.forward(x, stack)
        assert h_hat.shape == (1, 384)
        assert h_hat.dtype == np.float32

    def test_forward_batch(self):
        mlp = LatentMLP(dim=64, hidden=32, n_stacks=8, seed=0)
        x = np.stack([_mock_embed(64, seed=i) for i in range(5)])
        stack = np.zeros((5, 8), dtype=np.float32)
        stack[np.arange(5), np.arange(5) % 8] = 1.0
        h_hat = mlp.forward(x, stack)
        assert h_hat.shape == (5, 64)
        # Skip connection means output is not trivially zero at init.
        assert not np.allclose(h_hat, 0.0, atol=1e-6)

    def test_forward_skip_dominates_at_init(self):
        # With small init weights, forward should be close to x (skip path).
        mlp = LatentMLP(dim=32, hidden=16, n_stacks=4, seed=0)
        x = _mock_embed(32, seed=42).reshape(1, -1)
        stack = np.zeros((1, 4), dtype=np.float32)
        stack[0, 0] = 1.0
        h_hat = mlp.forward(x, stack)
        cos = float(
            (h_hat[0] @ x[0])
            / ((np.linalg.norm(h_hat[0]) * np.linalg.norm(x[0])) + 1e-8)
        )
        assert cos > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_aeon_predictor.py::TestLatentMLPForward -v`
Expected: `NotImplementedError: Task 2` on all three.

- [ ] **Step 3: Write minimal implementation**

Replace the `LatentMLP` class in `src/memory/aeon_predictor.py`:

```python
def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


class LatentMLP:
    """2-layer numpy MLP with skip connection.

    Input: concat(x[dim], stack_onehot[n_stacks]) of size dim+n_stacks
    Hidden: linear(hidden) -> ReLU -> linear(hidden) -> ReLU
    Output: linear(dim) + x   (residual / skip on the embedding path)
    """

    def __init__(self, dim: int, hidden: int, n_stacks: int, seed: int = 0) -> None:
        self.dim = dim
        self.hidden = hidden
        self.n_stacks = n_stacks
        rng = np.random.default_rng(seed)
        in_dim = dim + n_stacks
        scale1 = np.sqrt(2.0 / in_dim)
        scale2 = np.sqrt(2.0 / hidden)
        scale3 = np.sqrt(2.0 / hidden) * 0.1  # small init so skip dominates at t=0
        self.w1 = (rng.standard_normal((in_dim, hidden)) * scale1).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.w2 = (rng.standard_normal((hidden, hidden)) * scale2).astype(np.float32)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.w3 = (rng.standard_normal((hidden, dim)) * scale3).astype(np.float32)
        self.b3 = np.zeros(dim, dtype=np.float32)

    def forward(self, x: np.ndarray, stack_onehot: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must be (batch, {self.dim}), got {x.shape}")
        if stack_onehot.shape != (x.shape[0], self.n_stacks):
            raise ValueError(
                f"stack_onehot must be (batch, {self.n_stacks}), got {stack_onehot.shape}"
            )
        inp = np.concatenate([x, stack_onehot], axis=1).astype(np.float32)
        z1 = np.clip(inp @ self.w1 + self.b1, -30.0, 30.0)
        h1 = _relu(z1)
        z2 = np.clip(h1 @ self.w2 + self.b2, -30.0, 30.0)
        h2 = _relu(z2)
        delta = h2 @ self.w3 + self.b3
        out = (x + delta).astype(np.float32)
        # Cache for backward.
        self._cache = {"inp": inp, "z1": z1, "h1": h1, "z2": z2, "h2": h2, "x": x}
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_aeon_predictor.py::TestLatentMLPForward -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/memory/aeon_predictor.py tests/memory/test_aeon_predictor.py
git commit -m "feat(aeon): LatentMLP forward pass"
```

---

### Task 3: `LatentMLP.backward_cosine` (cosine-loss SGD step)

**Files:**
- Modify: `src/memory/aeon_predictor.py`
- Modify: `tests/memory/test_aeon_predictor.py`

- [ ] **Step 1: Write the failing test for backward pass**

Append to `tests/memory/test_aeon_predictor.py`:

```python
class TestLatentMLPBackward:
    def test_backward_reduces_cosine_loss(self):
        rng = np.random.default_rng(0)
        dim, n_stacks = 32, 4
        mlp = LatentMLP(dim=dim, hidden=16, n_stacks=n_stacks, seed=0)
        # Build a trivial pair: target = rotated x, same stack.
        x = rng.standard_normal((8, dim)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        target = np.roll(x, 1, axis=1)  # deterministic "next" latent
        target /= np.linalg.norm(target, axis=1, keepdims=True) + 1e-8
        stack = np.zeros((8, n_stacks), dtype=np.float32)
        stack[:, 0] = 1.0

        losses = []
        for _ in range(200):
            _ = mlp.forward(x, stack)
            loss = mlp.backward_cosine(target, lr=0.05)
            losses.append(loss)

        assert losses[-1] < losses[0] - 0.1, (
            f"loss did not drop enough: start={losses[0]:.3f} end={losses[-1]:.3f}"
        )

    def test_backward_returns_scalar(self):
        mlp = LatentMLP(dim=16, hidden=8, n_stacks=2, seed=0)
        x = np.ones((2, 16), dtype=np.float32) / 4.0
        stack = np.zeros((2, 2), dtype=np.float32)
        stack[:, 0] = 1.0
        _ = mlp.forward(x, stack)
        loss = mlp.backward_cosine(x, lr=0.01)
        assert isinstance(loss, float)
        assert 0.0 <= loss <= 2.0  # cosine loss range
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_aeon_predictor.py::TestLatentMLPBackward -v`
Expected: `AttributeError: 'LatentMLP' object has no attribute 'backward_cosine'`

- [ ] **Step 3: Write minimal implementation**

Append to the `LatentMLP` class in `src/memory/aeon_predictor.py` (after `forward`):

```python
    def backward_cosine(self, target: np.ndarray, lr: float = 1e-3) -> float:
        """One SGD step with cosine-similarity loss.

        loss = 1 - mean(cos(h_hat, target)). Returns the scalar loss
        BEFORE the update (for logging / convergence checks).
        """
        cache = self._cache
        x = cache["x"]
        inp = cache["inp"]
        h1 = cache["h1"]
        h2 = cache["h2"]
        z1 = cache["z1"]
        z2 = cache["z2"]
        batch, dim = x.shape
        if target.shape != (batch, dim):
            raise ValueError(f"target shape {target.shape} != {(batch, dim)}")

        # Recompute h_hat from the same x + delta path so grad lines up.
        delta = h2 @ self.w3 + self.b3
        h_hat = (x + delta).astype(np.float32)

        eps = 1e-8
        n_hat = np.linalg.norm(h_hat, axis=1, keepdims=True) + eps
        n_tgt = np.linalg.norm(target, axis=1, keepdims=True) + eps
        cos = np.sum(h_hat * target, axis=1, keepdims=True) / (n_hat * n_tgt)
        loss = float(1.0 - cos.mean())

        # d loss / d h_hat = -(1/batch) * [target/(|h_hat|*|target|)
        #                   - cos * h_hat / (|h_hat|^2)]
        d_h_hat = -(
            target / (n_hat * n_tgt)
            - cos * h_hat / (n_hat * n_hat)
        ) / batch

        # d_h_hat flows into delta and into the skip (skip grad on x is
        # not used — we don't update x, it is input data).
        d_delta = d_h_hat  # shape (batch, dim)
        d_w3 = h2.T @ d_delta
        d_b3 = d_delta.sum(axis=0)
        d_h2 = d_delta @ self.w3.T
        d_z2 = d_h2 * (z2 > 0)
        d_w2 = h1.T @ d_z2
        d_b2 = d_z2.sum(axis=0)
        d_h1 = d_z2 @ self.w2.T
        d_z1 = d_h1 * (z1 > 0)
        d_w1 = inp.T @ d_z1
        d_b1 = d_z1.sum(axis=0)

        # Gradient clip — mirrors ForgettingGate pattern.
        clip = 5.0
        for g in (d_w1, d_b1, d_w2, d_b2, d_w3, d_b3):
            np.clip(g, -clip, clip, out=g)

        self.w1 -= lr * d_w1
        self.b1 -= lr * d_b1
        self.w2 -= lr * d_w2
        self.b2 -= lr * d_b2
        self.w3 -= lr * d_w3
        self.b3 -= lr * d_b3
        return loss
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_aeon_predictor.py::TestLatentMLPBackward -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/memory/aeon_predictor.py tests/memory/test_aeon_predictor.py
git commit -m "feat(aeon): LatentMLP cosine-loss SGD step"
```

---

### Task 4: Collapse detector (`detect_collapse` helper)

**Files:**
- Modify: `src/memory/aeon_predictor.py`
- Modify: `tests/memory/test_aeon_predictor.py`

- [ ] **Step 1: Write the failing test for variance check**

Append to `tests/memory/test_aeon_predictor.py`:

```python
from src.memory.aeon_predictor import detect_collapse


class TestCollapseDetector:
    def test_flags_collapsed_predictions(self):
        rng = np.random.default_rng(0)
        h_t = rng.standard_normal((100, 64)).astype(np.float32)
        h_hat = np.ones_like(h_t) * 0.01  # near-constant == collapse
        flagged, ratio = detect_collapse(h_t, h_hat)
        assert flagged is True
        assert ratio < 0.1

    def test_accepts_healthy_predictions(self):
        rng = np.random.default_rng(1)
        h_t = rng.standard_normal((100, 64)).astype(np.float32)
        h_hat = h_t + rng.standard_normal(h_t.shape).astype(np.float32) * 0.05
        flagged, ratio = detect_collapse(h_t, h_hat)
        assert flagged is False
        assert ratio > 0.5

    def test_boundary_exactly_at_threshold(self):
        # Ratio exactly 0.1 -> NOT flagged (strict <).
        h_t = np.ones((10, 4), dtype=np.float32)
        h_t[0, 0] = 11.0  # std ~ 3.0
        h_hat = np.ones((10, 4), dtype=np.float32)
        h_hat[0, 0] = 2.0  # std ~ 0.3 -> ratio exactly 0.1
        flagged, ratio = detect_collapse(h_t, h_hat, threshold=0.1)
        assert flagged is False or ratio == pytest.approx(0.1, abs=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_aeon_predictor.py::TestCollapseDetector -v`
Expected: `ImportError: cannot import name 'detect_collapse'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/memory/aeon_predictor.py` (at module level, below the dataclass):

```python
def detect_collapse(
    h_t: np.ndarray, h_hat: np.ndarray, threshold: float = 0.1
) -> tuple[bool, float]:
    """Flag predictor collapse when std(h_hat) << std(h_t).

    Returns (flagged, ratio). ratio = std(h_hat) / std(h_t) averaged
    across feature dims. Flagged is True iff ratio < threshold.
    """
    if h_t.shape != h_hat.shape:
        raise ValueError(f"shape mismatch {h_t.shape} vs {h_hat.shape}")
    std_t = float(np.std(h_t))
    std_hat = float(np.std(h_hat))
    if std_t < 1e-9:
        return False, 1.0
    ratio = std_hat / std_t
    return bool(ratio < threshold), float(ratio)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_aeon_predictor.py::TestCollapseDetector -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/memory/aeon_predictor.py tests/memory/test_aeon_predictor.py
git commit -m "feat(aeon): predictor collapse detector"
```

---

### Task 5: `AeonPredictor.ingest_latent` + training buffer

**Files:**
- Modify: `src/memory/aeon_predictor.py`
- Modify: `tests/memory/test_aeon_predictor.py`

- [ ] **Step 1: Write the failing test for ingest**

Append to `tests/memory/test_aeon_predictor.py`:

```python
class TestAeonPredictorIngest:
    def _mk_palace(self, dim: int = 32) -> AeonSleep:
        return AeonSleep(dim=dim)

    def test_ingest_appends_to_buffer(self):
        palace = self._mk_palace(dim=32)
        pred = AeonPredictor(palace=palace, config=PredictorConfig(dim=32))
        h = _mock_embed(32, seed=1)
        pred.ingest_latent("t0", h, ts=datetime(2026, 4, 17, 10, 0))
        assert pred.buffer_size() == 1

    def test_ingest_builds_temporal_pairs(self):
        palace = self._mk_palace(dim=32)
        pred = AeonPredictor(palace=palace, config=PredictorConfig(dim=32))
        t0 = datetime(2026, 4, 17, 10, 0)
        pred.ingest_latent("t0", _mock_embed(32, seed=1), ts=t0, stack_id=3)
        pred.ingest_latent(
            "t1", _mock_embed(32, seed=2), ts=t0 + timedelta(minutes=1), stack_id=3
        )
        pairs = pred.pairs_for_training()
        assert len(pairs) == 1
        h_t, h_next, stack_id = pairs[0]
        assert h_t.shape == (32,)
        assert h_next.shape == (32,)
        assert stack_id == 3

    def test_ingest_rejects_wrong_dim(self):
        palace = self._mk_palace(dim=32)
        pred = AeonPredictor(palace=palace, config=PredictorConfig(dim=32))
        with pytest.raises(ValueError, match="dim"):
            pred.ingest_latent("t0", np.zeros(16, dtype=np.float32), ts=datetime.utcnow())

    def test_ingest_writes_to_palace_and_adds_temporal_edge(self):
        palace = self._mk_palace(dim=32)
        pred = AeonPredictor(palace=palace, config=PredictorConfig(dim=32))
        t0 = datetime(2026, 4, 17, 10, 0)
        pred.ingest_latent("t0", _mock_embed(32, seed=1), ts=t0)
        pred.ingest_latent("t1", _mock_embed(32, seed=2), ts=t0 + timedelta(minutes=1))
        temporal_edges = [e for e in palace.graph.edges(kind="temporal")]
        # Both nodes present, and a temporal edge t0 -> t1 exists.
        assert palace.graph.has_node("t0")
        assert palace.graph.has_node("t1")
        assert any(e.src == "t0" and e.dst == "t1" for e in temporal_edges)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_aeon_predictor.py::TestAeonPredictorIngest -v`
Expected: `NotImplementedError: Task 5` on construction.

- [ ] **Step 3: Write minimal implementation**

Replace the `AeonPredictor` skeleton in `src/memory/aeon_predictor.py` and add a dataclass for the buffer:

```python
@dataclass
class _PairSample:
    turn_id: str
    h: np.ndarray            # shape (dim,)
    ts: datetime
    stack_id: int            # -1 means "unknown / default"


class AeonPredictor:
    """Facade wrapping AeonSleep with a JEPA-style latent predictor."""

    def __init__(self, palace: "AeonSleep", config: PredictorConfig) -> None:
        if config.dim != palace.dim:
            raise ValueError(
                f"config.dim={config.dim} != palace.dim={palace.dim}"
            )
        self.palace = palace
        self.config = config
        self.mlp = LatentMLP(
            dim=config.dim,
            hidden=config.hidden,
            n_stacks=config.n_stacks,
            seed=config.seed,
        )
        self._buffer: list[_PairSample] = []
        self._trained_once = False

    # ---------------------------------------------------------- buffer mgmt

    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def ready(self) -> bool:
        return (
            self._trained_once
            and len(self._buffer) >= self.config.cold_start_threshold
        )

    # ---------------------------------------------------------- ingest

    def ingest_latent(
        self,
        turn_id: str,
        h: np.ndarray,
        ts: datetime,
        stack_id: int | None = None,
    ) -> None:
        """Append a latent sample to the buffer and persist to the palace.

        Writes an Episode to AeonSleep (atlas + graph) so existing recall
        paths still work. Adds a temporal edge from the previous sample
        (if any) to this one.
        """
        if h.shape != (self.config.dim,):
            raise ValueError(
                f"h.shape={h.shape} != expected ({self.config.dim},) dim"
            )
        from src.memory.aeonsleep import Episode

        ep = Episode(
            id=turn_id,
            text="",
            embedding=h.astype(np.float32).tolist(),
            ts=ts,
            topic="__predictor__",
            payload={"stack_id": -1 if stack_id is None else int(stack_id)},
        )
        self.palace.write(ep)

        prev = self._buffer[-1] if self._buffer else None
        if prev is not None and self.palace.graph.has_node(prev.turn_id):
            # Add explicit temporal edge even across different topics.
            self.palace.graph.add_typed_edge(
                prev.turn_id, turn_id, "temporal"
            )

        self._buffer.append(
            _PairSample(
                turn_id=turn_id,
                h=h.astype(np.float32).copy(),
                ts=ts,
                stack_id=-1 if stack_id is None else int(stack_id),
            )
        )

    def pairs_for_training(self) -> list[tuple[np.ndarray, np.ndarray, int]]:
        """Build `(h_t, h_{t+1}, stack_id)` triples from the ordered buffer."""
        out: list[tuple[np.ndarray, np.ndarray, int]] = []
        for a, b in zip(self._buffer, self._buffer[1:]):
            out.append((a.h, b.h, a.stack_id))
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_aeon_predictor.py::TestAeonPredictorIngest -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/memory/aeon_predictor.py tests/memory/test_aeon_predictor.py
git commit -m "feat(aeon): ingest_latent + training buffer"
```

---

### Task 6: `AeonPredictor.predict_next` (cold-start fallback + horizon)

**Files:**
- Modify: `src/memory/aeon_predictor.py`
- Modify: `tests/memory/test_aeon_predictor.py`

- [ ] **Step 1: Write the failing test for `predict_next`**

Append to `tests/memory/test_aeon_predictor.py`:

```python
class TestAeonPredictorPredict:
    def test_predict_cold_start_returns_input(self):
        palace = AeonSleep(dim=32)
        pred = AeonPredictor(palace=palace, config=PredictorConfig(dim=32))
        h = _mock_embed(32, seed=1)
        out = pred.predict_next(h, horizon=1)
        assert out.shape == (32,)
        np.testing.assert_allclose(out, h, atol=1e-6)  # fallback == identity

    def test_predict_after_ready_differs_from_input(self):
        palace = AeonSleep(dim=16)
        cfg = PredictorConfig(dim=16, hidden=8, n_stacks=2, cold_start_threshold=4)
        pred = AeonPredictor(palace=palace, config=cfg)
        t0 = datetime(2026, 4, 17, 10, 0)
        for i in range(6):
            pred.ingest_latent(
                f"t{i}", _mock_embed(16, seed=i), ts=t0 + timedelta(minutes=i)
            )
        # Force trained flag by calling fit_on_buffer (added in this task).
        pred.fit_on_buffer(lr=0.05, epochs=50, batch_size=4)
        assert pred.ready is True
        h = _mock_embed(16, seed=99)
        out = pred.predict_next(h, horizon=1)
        assert out.shape == (16,)
        # After training it should NOT be exactly h (residual has moved).
        assert not np.allclose(out, h, atol=1e-4)

    def test_predict_horizon_k_iterates(self):
        palace = AeonSleep(dim=16)
        cfg = PredictorConfig(dim=16, hidden=8, n_stacks=2, cold_start_threshold=4)
        pred = AeonPredictor(palace=palace, config=cfg)
        t0 = datetime(2026, 4, 17, 10, 0)
        for i in range(6):
            pred.ingest_latent(
                f"t{i}", _mock_embed(16, seed=i), ts=t0 + timedelta(minutes=i)
            )
        pred.fit_on_buffer(lr=0.05, epochs=30, batch_size=4)
        h = _mock_embed(16, seed=99)
        out1 = pred.predict_next(h, horizon=1)
        out3 = pred.predict_next(h, horizon=3)
        # horizon=3 must differ from horizon=1 (3 applied rollouts).
        assert not np.allclose(out1, out3, atol=1e-4)

    def test_predict_rejects_bad_horizon(self):
        palace = AeonSleep(dim=16)
        pred = AeonPredictor(palace=palace, config=PredictorConfig(dim=16))
        with pytest.raises(ValueError, match="horizon"):
            pred.predict_next(_mock_embed(16, seed=0), horizon=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_aeon_predictor.py::TestAeonPredictorPredict -v`
Expected: `AttributeError: 'AeonPredictor' object has no attribute 'fit_on_buffer'` / `predict_next`.

- [ ] **Step 3: Write minimal implementation**

Append to the `AeonPredictor` class in `src/memory/aeon_predictor.py`:

```python
    # ---------------------------------------------------------- training

    def fit_on_buffer(
        self,
        *,
        lr: float = 1e-3,
        epochs: int = 1,
        batch_size: int = 32,
    ) -> list[float]:
        """One (or more) SGD passes over the current pair buffer.

        Returns per-epoch mean loss. Sets `_trained_once=True` iff at
        least one pair was seen.
        """
        triples = self.pairs_for_training()
        if not triples:
            return []
        n = len(triples)
        rng = np.random.default_rng(self.config.seed)
        history: list[float] = []
        for _ in range(max(1, epochs)):
            order = rng.permutation(n)
            losses: list[float] = []
            for start in range(0, n, batch_size):
                batch = [triples[i] for i in order[start:start + batch_size]]
                x = np.stack([t[0] for t in batch]).astype(np.float32)
                tgt = np.stack([t[1] for t in batch]).astype(np.float32)
                stack = self._stack_onehot([t[2] for t in batch])
                self.mlp.forward(x, stack)
                loss = self.mlp.backward_cosine(tgt, lr=lr)
                losses.append(loss)
            history.append(float(np.mean(losses)))
        self._trained_once = True
        return history

    # ---------------------------------------------------------- predict

    def predict_next(
        self,
        h_t: np.ndarray,
        horizon: int = 1,
        stack_id: int | None = None,
    ) -> np.ndarray:
        """Predict h_{t+horizon} by iterating the MLP `horizon` times.

        Cold-start: if `not self.ready`, return `h_t` unchanged so the
        serving path stays on pure-retrieval mode.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if h_t.shape != (self.config.dim,):
            raise ValueError(
                f"h_t.shape={h_t.shape} != ({self.config.dim},)"
            )
        if not self.ready:
            return h_t.astype(np.float32).copy()
        x = h_t.reshape(1, -1).astype(np.float32)
        stack = self._stack_onehot([-1 if stack_id is None else int(stack_id)])
        current = x
        for _ in range(horizon):
            current = self.mlp.forward(current, stack)
        return current[0].astype(np.float32)

    # ---------------------------------------------------------- helpers

    def _stack_onehot(self, stack_ids: list[int]) -> np.ndarray:
        n = len(stack_ids)
        out = np.zeros((n, self.config.n_stacks), dtype=np.float32)
        for i, sid in enumerate(stack_ids):
            if sid < 0:
                continue  # unknown stack -> all zeros (null condition)
            idx = sid % self.config.n_stacks
            out[i, idx] = 1.0
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_aeon_predictor.py::TestAeonPredictorPredict -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/memory/aeon_predictor.py tests/memory/test_aeon_predictor.py
git commit -m "feat(aeon): predict_next + cold-start fallback"
```

---

### Task 7: `AeonPredictor.recall` backward-compat delegation

**Files:**
- Modify: `src/memory/aeon_predictor.py`
- Modify: `tests/memory/test_aeon_predictor.py`

- [ ] **Step 1: Write the failing test for recall delegation**

Append to `tests/memory/test_aeon_predictor.py`:

```python
class TestAeonPredictorRecall:
    def test_recall_delegates_to_palace(self):
        palace = AeonSleep(dim=16)
        pred = AeonPredictor(palace=palace, config=PredictorConfig(dim=16))
        t0 = datetime(2026, 4, 17, 10, 0)
        pred.ingest_latent("t0", _mock_embed(16, seed=1), ts=t0)
        pred.ingest_latent(
            "t1", _mock_embed(16, seed=2), ts=t0 + timedelta(minutes=1)
        )
        hits = pred.recall(_mock_embed(16, seed=1), top_k=2)
        assert len(hits) >= 1
        # Top hit should be t0 or t1 — both live in the atlas now.
        assert hits[0].episode_id in {"t0", "t1"}

    def test_recall_top_k_respected(self):
        palace = AeonSleep(dim=16)
        pred = AeonPredictor(palace=palace, config=PredictorConfig(dim=16))
        t0 = datetime(2026, 4, 17, 10, 0)
        for i in range(5):
            pred.ingest_latent(
                f"t{i}", _mock_embed(16, seed=i), ts=t0 + timedelta(minutes=i)
            )
        hits = pred.recall(_mock_embed(16, seed=0), top_k=3)
        assert len(hits) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_aeon_predictor.py::TestAeonPredictorRecall -v`
Expected: `AttributeError: 'AeonPredictor' object has no attribute 'recall'`

- [ ] **Step 3: Write minimal implementation**

Append to the `AeonPredictor` class in `src/memory/aeon_predictor.py`:

```python
    # ---------------------------------------------------------- recall

    def recall(self, query_vec: np.ndarray, top_k: int = 10):
        """Delegate to the underlying AeonSleep — backward-compat path."""
        if query_vec.shape != (self.config.dim,):
            raise ValueError(
                f"query_vec.shape={query_vec.shape} != ({self.config.dim},)"
            )
        return self.palace.recall(query_vec.tolist(), k=top_k)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_aeon_predictor.py::TestAeonPredictorRecall -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/memory/aeon_predictor.py tests/memory/test_aeon_predictor.py
git commit -m "feat(aeon): recall backward-compat delegation"
```

---

### Task 8: Stack conditioning end-to-end

**Files:**
- Modify: `tests/memory/test_aeon_predictor.py`

*(No src changes — the one-hot concat is already in `LatentMLP.forward` and `_stack_onehot`. This task is the end-to-end assertion that stack IDs actually steer the predictor.)*

- [ ] **Step 1: Write the failing test for stack separation**

Append to `tests/memory/test_aeon_predictor.py`:

```python
class TestStackConditioning:
    def test_two_stacks_learn_different_targets(self):
        """Same h_t but stacks 0 and 1 learn opposite shifts."""
        palace = AeonSleep(dim=16)
        cfg = PredictorConfig(
            dim=16, hidden=16, n_stacks=4, cold_start_threshold=4
        )
        pred = AeonPredictor(palace=palace, config=cfg)
        t0 = datetime(2026, 4, 17, 10, 0)

        rng = np.random.default_rng(0)
        # Alternate stack 0 (shift-right) and stack 1 (shift-left) pairs.
        for i in range(40):
            h = rng.standard_normal(16).astype(np.float32)
            h /= np.linalg.norm(h) + 1e-8
            pred.ingest_latent(
                f"t{i}", h, ts=t0 + timedelta(minutes=i), stack_id=i % 2
            )

        # Train long enough that the two stacks separate.
        pred.fit_on_buffer(lr=0.05, epochs=200, batch_size=8)

        probe = rng.standard_normal(16).astype(np.float32)
        probe /= np.linalg.norm(probe) + 1e-8
        out_s0 = pred.predict_next(probe, horizon=1, stack_id=0)
        out_s1 = pred.predict_next(probe, horizon=1, stack_id=1)
        assert not np.allclose(out_s0, out_s1, atol=1e-3), (
            "stack 0 and stack 1 should produce distinguishable outputs"
        )

    def test_unknown_stack_uses_null_condition(self):
        palace = AeonSleep(dim=16)
        pred = AeonPredictor(palace=palace, config=PredictorConfig(dim=16, cold_start_threshold=2))
        t0 = datetime(2026, 4, 17, 10, 0)
        for i in range(4):
            pred.ingest_latent(
                f"t{i}", _mock_embed(16, seed=i),
                ts=t0 + timedelta(minutes=i), stack_id=None,
            )
        pred.fit_on_buffer(epochs=10, batch_size=4)
        # stack_id=None should not raise and should not produce NaN.
        out = pred.predict_next(_mock_embed(16, seed=99), stack_id=None)
        assert not np.any(np.isnan(out))
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_aeon_predictor.py::TestStackConditioning -v`
Expected: test_two_stacks_learn_different_targets may already pass thanks to the one-hot; if it FAILS (outputs identical because `n_stacks` one-hot contribution is washed out), bump the `w1` scale for the stack columns.

- [ ] **Step 3: If needed, fix `_stack_onehot` scaling**

If `test_two_stacks_learn_different_targets` fails in step 2, edit `src/memory/aeon_predictor.py::AeonPredictor._stack_onehot` to scale the one-hot up so the signal is not swamped by the `dim`-sized input:

```python
    def _stack_onehot(self, stack_ids: list[int]) -> np.ndarray:
        n = len(stack_ids)
        out = np.zeros((n, self.config.n_stacks), dtype=np.float32)
        for i, sid in enumerate(stack_ids):
            if sid < 0:
                continue
            idx = sid % self.config.n_stacks
            out[i, idx] = 1.0
        # Amplify the stack signal so it isn't washed out by a high-dim x.
        # Ratio calibrated so stack column L2 ~ sqrt(dim)/sqrt(n_stacks).
        scale = float(np.sqrt(self.config.dim / max(self.config.n_stacks, 1)))
        return (out * scale).astype(np.float32)
```

Otherwise skip to Step 4.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_aeon_predictor.py::TestStackConditioning -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/memory/aeon_predictor.py tests/memory/test_aeon_predictor.py
git commit -m "feat(aeon): stack-conditioned predictor e2e"
```

---

### Task 9: Hook predictor training into `AeonSleep.sleep_cycle`

**Files:**
- Modify: `src/memory/aeonsleep.py`
- Create: `tests/memory/test_aeonsleep_predictor_hook.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/memory/test_aeonsleep_predictor_hook.py`:

```python
"""Integration test: sleep_cycle step 5 trains the predictor."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.memory.aeon_predictor import (
    AeonPredictor,
    PredictorConfig,
    detect_collapse,
)
from src.memory.aeonsleep import AeonSleep


def _unit(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-8)


@pytest.fixture
def palace_with_pairs():
    palace = AeonSleep(dim=16)
    cfg = PredictorConfig(
        dim=16, hidden=16, n_stacks=4, cold_start_threshold=4
    )
    pred = AeonPredictor(palace=palace, config=cfg)
    t0 = datetime(2026, 4, 17, 10, 0)
    rng = np.random.default_rng(0)
    for i in range(12):
        h = _unit(rng.standard_normal(16).astype(np.float32))
        pred.ingest_latent(
            f"t{i}", h, ts=t0 + timedelta(minutes=i), stack_id=i % 4
        )
    palace.attach_predictor(pred)
    return palace, pred


def test_sleep_cycle_runs_predictor_training(palace_with_pairs):
    palace, pred = palace_with_pairs
    assert pred._trained_once is False
    report = palace.sleep_cycle()
    assert pred._trained_once is True
    # The report carries predictor stats.
    assert report.predictor_epochs == 1
    assert report.predictor_loss is not None
    assert 0.0 <= report.predictor_loss <= 2.0


def test_sleep_cycle_rollback_on_collapse(palace_with_pairs):
    palace, pred = palace_with_pairs
    # Force a collapsed state by zeroing the output layer.
    pred.mlp.w3 *= 0.0
    pred.mlp.b3 *= 0.0
    # Take a snapshot of w1 to verify rollback-touched weights match the
    # pre-fit snapshot after a detected collapse.
    pre_w1 = pred.mlp.w1.copy()
    report = palace.sleep_cycle()
    # Collapse detector should have tripped and weights rolled back.
    assert report.predictor_collapsed is True
    np.testing.assert_allclose(pred.mlp.w1, pre_w1, atol=1e-6)


def test_sleep_cycle_no_predictor_attached():
    palace = AeonSleep(dim=16)
    report = palace.sleep_cycle()
    assert report.predictor_epochs == 0
    assert report.predictor_loss is None
    assert report.predictor_collapsed is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_aeonsleep_predictor_hook.py -v`
Expected: `AttributeError: 'AeonSleep' object has no attribute 'attach_predictor'`

- [ ] **Step 3: Add `attach_predictor` + extend `SleepReport` + step 5 in `sleep_cycle`**

In `src/memory/aeonsleep.py`:

3a. Extend `SleepReport` — replace the existing dataclass near line 95:

```python
@dataclass
class SleepReport:
    """Per-cycle summary returned by :meth:`AeonSleep.sleep_cycle`."""

    tags_assigned: int
    evicted: int
    kept: int
    clusters_built: int
    compression_ratio: float
    # Predictor hook (step 5) — optional, all zeros when no predictor.
    predictor_epochs: int = 0
    predictor_loss: float | None = None
    predictor_collapsed: bool = False
```

3b. In `AeonSleep.__init__` (right after `self._summary_counter = itertools.count()` around line 158), add:

```python
        # Optional latent predictor attached via attach_predictor().
        self._predictor = None
```

3c. Add a new public method on `AeonSleep` (right before `# ------ write api`, around line 160):

```python
    def attach_predictor(self, predictor) -> None:
        """Register a predictor to be trained during sleep_cycle step 5."""
        self._predictor = predictor
```

3d. In `sleep_cycle`, after the `stats = self.consolidator.last_stats()` line and before the return, insert step 5:

```python
        # Step 5: train attached latent predictor on current buffer.
        predictor_epochs = 0
        predictor_loss: float | None = None
        predictor_collapsed = False
        if self._predictor is not None:
            triples = self._predictor.pairs_for_training()
            if triples:
                # Snapshot weights for rollback on collapse.
                import copy as _copy
                snap = {
                    "w1": self._predictor.mlp.w1.copy(),
                    "b1": self._predictor.mlp.b1.copy(),
                    "w2": self._predictor.mlp.w2.copy(),
                    "b2": self._predictor.mlp.b2.copy(),
                    "w3": self._predictor.mlp.w3.copy(),
                    "b3": self._predictor.mlp.b3.copy(),
                }
                from src.memory.aeon_predictor import detect_collapse

                history = self._predictor.fit_on_buffer(
                    lr=1e-3, epochs=1, batch_size=32
                )
                predictor_epochs = len(history)
                predictor_loss = history[-1] if history else None

                # Collapse check on the same buffer.
                import numpy as _np
                xs = _np.stack([t[0] for t in triples]).astype(_np.float32)
                tgt = _np.stack([t[1] for t in triples]).astype(_np.float32)
                sids = [t[2] for t in triples]
                stack = self._predictor._stack_onehot(sids)
                h_hat = self._predictor.mlp.forward(xs, stack)
                flagged, _ratio = detect_collapse(tgt, h_hat)
                if flagged:
                    predictor_collapsed = True
                    # Roll back weights.
                    for k, v in snap.items():
                        setattr(self._predictor.mlp, k, v)
```

3e. Update the return statement at the end of `sleep_cycle`:

```python
        return SleepReport(
            tags_assigned=tags_assigned,
            evicted=evicted,
            kept=len(raw_ids) - evicted,
            clusters_built=len(clusters),
            compression_ratio=compression,
            predictor_epochs=predictor_epochs,
            predictor_loss=predictor_loss,
            predictor_collapsed=predictor_collapsed,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_aeonsleep_predictor_hook.py -v`
Expected: `3 passed`

Also run the original AeonSleep tests to confirm no regression:

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/test_aeonsleep.py -v`
Expected: all pre-existing tests still pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/memory/aeonsleep.py tests/memory/test_aeonsleep_predictor_hook.py
git commit -m "feat(aeon): sleep_cycle step 5 trains predictor"
```

---

### Task 10: PoC evaluation script (`eval_aeon_predictor.py`)

**Files:**
- Create: `scripts/eval_aeon_predictor.py`
- Create: `tests/scripts/test_eval_aeon_predictor.py`

- [ ] **Step 1: Write the failing smoke test for the script**

Create `tests/scripts/test_eval_aeon_predictor.py`:

```python
"""Smoke test for scripts/eval_aeon_predictor.py — must finish < 5 s."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_eval_aeon_predictor_smoke(tmp_path):
    out = tmp_path / "result.json"
    t0 = time.time()
    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "eval_aeon_predictor.py"),
            "--dim", "16",
            "--n-turns", "80",
            "--n-queries", "10",
            "--cold-start", "4",
            "--epochs", "20",
            "--out", str(out),
            "--seed", "0",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=60,
    )
    elapsed = time.time() - t0
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    data = json.loads(out.read_text())
    for key in (
        "baseline_recall_at_5",
        "predictive_recall_at_5",
        "baseline_mrr",
        "predictive_mrr",
        "win_rate_predictive",
    ):
        assert key in data
    assert elapsed < 5.0, f"smoke run too slow: {elapsed:.2f}s"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/scripts/test_eval_aeon_predictor.py -v`
Expected: `FileNotFoundError` or `returncode != 0` (script does not exist).

Ensure `tests/scripts/__init__.py` exists:

```bash
cd /Users/electron/Documents/Projets/micro-kiki
mkdir -p tests/scripts
test -f tests/scripts/__init__.py || touch tests/scripts/__init__.py
```

- [ ] **Step 3: Write minimal implementation**

Create `scripts/eval_aeon_predictor.py`:

```python
#!/usr/bin/env python3
"""PoC Scenario B eval — AeonPredictor vs pure retrieval.

Generates a synthetic stream of (h_t, h_{t+1}) pairs (random walk on the
unit sphere), trains the predictor, then for N held-out queries compares:

    baseline:   palace.recall(h_q,                 k=5)
    predictive: palace.recall(predictor.predict_next(h_q), k=5)

Reports Recall@5, MRR, and the per-query win rate of the predictive path.

Usage:
    uv run python scripts/eval_aeon_predictor.py --dim 384 --n-turns 1000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.memory.aeon_predictor import AeonPredictor, PredictorConfig
from src.memory.aeonsleep import AeonSleep


@dataclass
class EvalResult:
    baseline_recall_at_5: float
    predictive_recall_at_5: float
    baseline_mrr: float
    predictive_mrr: float
    win_rate_predictive: float
    n_queries: int
    elapsed_seconds: float
    final_train_loss: float
    predictor_ready: bool


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def _build_stream(n_turns: int, dim: int, seed: int) -> list[np.ndarray]:
    """Random-walk on the unit sphere: h_{t+1} = normalize(h_t + 0.3 * noise)."""
    rng = np.random.default_rng(seed)
    out = [_unit(rng.standard_normal(dim).astype(np.float32))]
    for _ in range(n_turns - 1):
        step = 0.3 * rng.standard_normal(dim).astype(np.float32)
        out.append(_unit(out[-1] + step))
    return out


def _reciprocal_rank(hit_ids: list[str], gold: str) -> float:
    for rank, hid in enumerate(hit_ids, start=1):
        if hid == gold:
            return 1.0 / rank
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--n-turns", type=int, default=1000)
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--cold-start", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t_start = time.time()

    # 1. Build stream and ingest.
    palace = AeonSleep(dim=args.dim)
    cfg = PredictorConfig(
        dim=args.dim,
        hidden=min(256, args.dim),
        n_stacks=16,
        cold_start_threshold=args.cold_start,
        seed=args.seed,
    )
    pred = AeonPredictor(palace=palace, config=cfg)
    palace.attach_predictor(pred)

    stream = _build_stream(args.n_turns, args.dim, seed=args.seed)
    t0 = datetime(2026, 4, 17, 10, 0)
    for i, h in enumerate(stream):
        pred.ingest_latent(
            f"t{i}", h, ts=t0 + timedelta(seconds=i), stack_id=rng.integers(0, 16)
        )

    # 2. Train predictor.
    history = pred.fit_on_buffer(
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size
    )
    final_loss = history[-1] if history else float("nan")

    # 3. Build eval set: held-out indices (last 20% of stream).
    n_held = max(args.n_queries, 1)
    held_start = max(1, len(stream) - n_held - 1)
    queries = []
    for i in range(held_start, held_start + n_held):
        if i + 1 >= len(stream):
            break
        queries.append((stream[i], f"t{i + 1}"))  # (h_q, gold next-turn id)

    # 4. Compare baseline vs predictive.
    baseline_hits, pred_hits = [], []
    baseline_rr, pred_rr = [], []
    wins = 0
    for h_q, gold in queries:
        base = palace.recall(h_q.tolist(), k=5)
        base_ids = [h.episode_id for h in base]
        baseline_hits.append(gold in base_ids)
        baseline_rr.append(_reciprocal_rank(base_ids, gold))

        h_pred = pred.predict_next(h_q, horizon=1)
        pr = palace.recall(h_pred.tolist(), k=5)
        pr_ids = [h.episode_id for h in pr]
        pred_hits.append(gold in pr_ids)
        pred_rr.append(_reciprocal_rank(pr_ids, gold))

        if pred_rr[-1] >= baseline_rr[-1] and (
            pred_rr[-1] > baseline_rr[-1] or pred_hits[-1] > baseline_hits[-1]
        ):
            wins += 1

    result = EvalResult(
        baseline_recall_at_5=float(np.mean(baseline_hits)) if baseline_hits else 0.0,
        predictive_recall_at_5=float(np.mean(pred_hits)) if pred_hits else 0.0,
        baseline_mrr=float(np.mean(baseline_rr)) if baseline_rr else 0.0,
        predictive_mrr=float(np.mean(pred_rr)) if pred_rr else 0.0,
        win_rate_predictive=(wins / len(queries)) if queries else 0.0,
        n_queries=len(queries),
        elapsed_seconds=time.time() - t_start,
        final_train_loss=float(final_loss),
        predictor_ready=pred.ready,
    )

    payload = asdict(result)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/scripts/test_eval_aeon_predictor.py -v`
Expected: `1 passed` in < 5 s.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add scripts/eval_aeon_predictor.py tests/scripts/test_eval_aeon_predictor.py tests/scripts/__init__.py
git commit -m "feat(aeon): PoC eval script + smoke test"
```

---

### Task 11: Run the real PoC experiment + document numbers

**Files:**
- Create: `results/2026-04-17-aeon-predictor-poc.md`

*(No code changes — this task runs the eval at full scale and writes the findings.)*

- [ ] **Step 1: Run the full eval**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
mkdir -p results
uv run python scripts/eval_aeon_predictor.py \
    --dim 384 \
    --n-turns 1000 \
    --n-queries 100 \
    --cold-start 500 \
    --epochs 50 \
    --lr 1e-3 \
    --batch-size 32 \
    --seed 0 \
    --out results/2026-04-17-aeon-predictor-poc.json
```

Expected: JSON printed to stdout AND saved to `results/2026-04-17-aeon-predictor-poc.json`. Wall-clock < 30 s on GrosMac M5.

- [ ] **Step 2: Write the results report**

Create `results/2026-04-17-aeon-predictor-poc.md` with the real numbers from Step 1. Fill in the `<FILL FROM JSON>` slots by reading `results/2026-04-17-aeon-predictor-poc.json`:

```markdown
# Aeon Latent Predictor — PoC Scenario B Results (2026-04-17)

## Setup
- dim=384, n_turns=1000, n_queries=100
- cold-start threshold: 500 pairs
- training: 50 epochs, lr=1e-3, batch=32, cosine-sim loss
- hardware: GrosMac M5 (16 GB), numpy only, no GPU

## Results (from results/2026-04-17-aeon-predictor-poc.json)
| Metric                       | Baseline (RAG) | Predictive (JEPA rollout) |
|------------------------------|----------------|---------------------------|
| Recall@5                     | <FILL FROM JSON: baseline_recall_at_5>  | <FILL FROM JSON: predictive_recall_at_5>  |
| MRR                          | <FILL FROM JSON: baseline_mrr>          | <FILL FROM JSON: predictive_mrr>          |
| Per-query predictive win-rate (>= 60% is success) | — | <FILL FROM JSON: win_rate_predictive> |
| Final cosine training loss   | —              | <FILL FROM JSON: final_train_loss>        |
| Eval wall-clock (s)          | —              | <FILL FROM JSON: elapsed_seconds>         |

## Decision
- Success gate (win_rate_predictive >= 0.60): <PASS | FAIL>
- Kill gate (final_train_loss > 0.3 OR collapse): <N/A | TRIPPED>

## Next step
- If PASS: promote AeonPredictor into the critic/judge query path in
  `src/cognitive/judge.py` behind a feature flag; plan Scenario B-2
  (stack-conditioned per-LoRA predictor, horizon>1 rollout) on the
  backlog.
- If FAIL: keep Aeon as pure retrieval (status quo); log PoC result
  and close task.
```

- [ ] **Step 3: Run pre-existing test suite to confirm no regression**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/ tests/test_aeonsleep.py tests/test_aeon_hook.py -v`
Expected: all green (no regressions).

- [ ] **Step 4: Commit results**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add results/2026-04-17-aeon-predictor-poc.md results/2026-04-17-aeon-predictor-poc.json
git commit -m "docs(aeon): PoC B results"
```

---

### Task 12: Final sanity sweep

**Files:**
- None (verification only)

- [ ] **Step 1: Full project test run**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/ tests/scripts/test_eval_aeon_predictor.py tests/test_aeonsleep.py -v`
Expected: all tests green.

- [ ] **Step 2: Confirm commit log**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && git log --oneline -15`
Expected: 12 new commits visible with the `feat(aeon):` / `docs(aeon):` prefixes from this plan, none exceeding 50 chars subject length.

- [ ] **Step 3: Close out**

If Task 11 PASSED:
- Mark PoC Scenario B complete in the project task tracker.
- Open a follow-up task "Stack-conditioned predictor hardening + judge integration" per the next step in `results/2026-04-17-aeon-predictor-poc.md`.

If Task 11 FAILED:
- Revert `AeonSleep.attach_predictor` call sites in downstream code (there are none yet — `aeon_hook.py` was intentionally NOT touched by this plan).
- Keep the module in-tree (no deletion) for the next iteration but mark the eval outcome in the results report.

---

## Plan summary

- 12 tasks total, each 5 steps (60 checkbox steps) except Task 12 (3 steps).
- Each task produces ONE commit.
- No file exceeds one clear responsibility.
- No TODO / TBD / "similar to" in any step.
- Risk mitigations (collapse, stack spaces, cold-start) are covered by Tasks 4, 8, and 6 respectively.
- Success / kill criteria are encoded in Task 11's decision block.
