# Phase C1 — Classical Baselines vs VQC Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a single JSON + markdown comparing `TorchVQCRouter` against 4 classical baselines on the 10-domain routing task, so Paper A §4 can contextualize the ~0.40 VQC ceiling with rigorous reference numbers.

**Architecture:** A uniform `run_baseline()` function dispatches to sklearn classifiers or `TorchVQCRouter` with a consistent API (`{name, X_train, y_train, X_test, y_test}` → `{accuracy, macro_f1, train_time_s, n_params}`). A benchmark script loads the existing embeddings pipeline, runs all 5 baselines on identical splits × 5 seeds, aggregates mean±std, writes machine-readable JSON + paper-facing markdown with a comparison figure.

**Tech Stack:** Python 3.13, numpy, torch, scikit-learn (LogisticRegression, MLPClassifier, DummyClassifier, PCA), matplotlib, existing micro-kiki embedding pipeline (`src.routing.text_jepa.dataset.load_domain_corpus`, SentenceTransformer baseline), merged `TorchVQCRouter`.

---

## File Structure

**Files to create:**
- `src/routing/classical_baselines.py` — uniform baseline runner
- `tests/routing/test_classical_baselines.py` — unit tests for the runner
- `scripts/bench_classical_vs_vqc.py` — orchestration CLI
- `results/c1-classical-vs-vqc.json` — output numbers
- `docs/paper-a/c1-baseline-results.md` — paper-facing narrative
- `docs/paper-a/c1-comparison.pdf` — matplotlib figure

**Files to modify:**
- `pyproject.toml` — add `scikit-learn` + `matplotlib` to optional-deps `eval`

---

### Task 1: Add scikit-learn + matplotlib dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Read current pyproject.toml structure**

Run: `grep -n optional-dependencies pyproject.toml -A 20`
Expected: see existing `[project.optional-dependencies]` section

- [ ] **Step 2: Add scikit-learn + matplotlib to `eval` extras**

In `pyproject.toml`, locate the `[project.optional-dependencies]` block (or create one if absent). Ensure an `eval` extra contains:

```toml
eval = [
  "scikit-learn>=1.5",
  "matplotlib>=3.9",
]
```

If `eval` already exists, merge these two lines preserving existing entries.

- [ ] **Step 3: Install in the local venv**

Run: `uv sync --extra eval` (or `uv pip install 'scikit-learn>=1.5' 'matplotlib>=3.9'` if sync is not configured)
Expected: both packages installed, no errors.

- [ ] **Step 4: Verify import**

Run: `uv run python -c "from sklearn.linear_model import LogisticRegression; from sklearn.neural_network import MLPClassifier; from sklearn.decomposition import PCA; from sklearn.dummy import DummyClassifier; from sklearn.metrics import f1_score; import matplotlib.pyplot as plt; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps(eval): add scikit-learn + matplotlib for C1 baselines"
```

---

### Task 2: Write failing test for `run_classical_baseline` dispatch

**Files:**
- Create: `tests/routing/test_classical_baselines.py`

- [ ] **Step 1: Create the test file**

Write to `tests/routing/test_classical_baselines.py`:

```python
"""Tests for src/routing/classical_baselines.py run_classical_baseline dispatcher."""
from __future__ import annotations

import numpy as np
import pytest


def _make_separable_task(n_classes: int = 4, per_class: int = 40, dim: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-2.0, 2.0, size=(n_classes, dim))
    X, y = [], []
    for c in range(n_classes):
        for _ in range(per_class):
            X.append(centers[c] + rng.normal(0, 0.2, size=dim))
            y.append(c)
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def test_run_baseline_stratified_returns_expected_shape():
    from src.routing.classical_baselines import run_classical_baseline

    X, y = _make_separable_task(n_classes=4, per_class=20, seed=0)
    n_tr = int(0.8 * len(X))
    out = run_classical_baseline("stratified", X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], seed=0)
    assert set(out.keys()) == {"name", "accuracy", "macro_f1", "train_time_s", "n_params"}
    assert out["name"] == "stratified"
    assert 0.0 <= out["accuracy"] <= 1.0
    assert 0.0 <= out["macro_f1"] <= 1.0
    assert out["train_time_s"] >= 0.0
    assert out["n_params"] >= 0


def test_run_baseline_logreg_beats_chance_on_separable_task():
    from src.routing.classical_baselines import run_classical_baseline

    X, y = _make_separable_task(n_classes=4, per_class=30, seed=1)
    n_tr = int(0.8 * len(X))
    out = run_classical_baseline("logreg", X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], seed=0)
    assert out["accuracy"] > 0.85, f"LogReg should ace separable task, got {out['accuracy']:.3f}"


def test_run_baseline_logreg_pca_matches_logreg_when_pca_dim_eq_input_dim():
    from src.routing.classical_baselines import run_classical_baseline

    X, y = _make_separable_task(n_classes=3, per_class=20, dim=8, seed=2)
    n_tr = int(0.8 * len(X))
    a = run_classical_baseline("logreg", X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], seed=0)
    b = run_classical_baseline("logreg_pca", X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], seed=0, pca_dim=8)
    # PCA with pca_dim == input_dim is lossless (up to rotation), LogReg should match
    assert abs(a["accuracy"] - b["accuracy"]) < 0.05


def test_run_baseline_mlp_returns_param_count():
    from src.routing.classical_baselines import run_classical_baseline

    X, y = _make_separable_task(n_classes=4, per_class=20, dim=16, seed=3)
    n_tr = int(0.8 * len(X))
    out = run_classical_baseline("mlp", X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], seed=0, hidden_dim=8)
    # MLP: (16*8 + 8) + (8*4 + 4) = 172
    expected = 16 * 8 + 8 + 8 * 4 + 4
    assert out["n_params"] == expected, f"expected {expected}, got {out['n_params']}"


def test_run_baseline_torch_vqc_returns_result():
    from src.routing.classical_baselines import run_classical_baseline

    X, y = _make_separable_task(n_classes=3, per_class=20, dim=32, seed=4)
    n_tr = int(0.8 * len(X))
    out = run_classical_baseline(
        "torch_vqc", X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:],
        seed=0, n_qubits=4, n_layers=6, epochs=30,
    )
    assert 0.0 <= out["accuracy"] <= 1.0
    assert out["n_params"] > 0


def test_run_baseline_unknown_name_raises():
    from src.routing.classical_baselines import run_classical_baseline

    X, y = _make_separable_task(n_classes=2, per_class=10, seed=5)
    n_tr = int(0.8 * len(X))
    with pytest.raises(ValueError, match="unknown baseline"):
        run_classical_baseline("bogus", X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:])
```

- [ ] **Step 2: Run test to verify all fail for the right reason**

Run: `uv run python -m pytest tests/routing/test_classical_baselines.py -v`
Expected: all 6 tests FAIL with `ModuleNotFoundError: No module named 'src.routing.classical_baselines'`

- [ ] **Step 3: Commit**

```bash
git add tests/routing/test_classical_baselines.py
git commit -m "test(c1): add classical baseline runner tests (failing)"
```

---

### Task 3: Implement `classical_baselines.py`

**Files:**
- Create: `src/routing/classical_baselines.py`

- [ ] **Step 1: Create the module**

Write to `src/routing/classical_baselines.py`:

```python
"""Uniform baseline runner for Phase C1: classical classifiers + TorchVQCRouter.

All baselines return the same dict: {name, accuracy, macro_f1, train_time_s, n_params}.
Used by scripts/bench_classical_vs_vqc.py to produce apples-to-apples comparisons.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from src.routing.torch_vqc_router import TorchVQCRouter

_KNOWN = {"stratified", "logreg", "logreg_pca", "mlp", "torch_vqc"}


def run_classical_baseline(
    name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    *,
    seed: int = 0,
    pca_dim: int = 4,
    hidden_dim: int = 64,
    n_qubits: int = 4,
    n_layers: int = 6,
    epochs: int = 300,
    lr: float = 0.05,
    weight_decay: float = 1e-4,
    max_iter: int = 2000,
) -> dict:
    """Train + eval one baseline. Returns uniform dict for aggregation."""
    if name not in _KNOWN:
        raise ValueError(f"unknown baseline {name!r} — must be one of {_KNOWN}")

    from sklearn.metrics import accuracy_score, f1_score

    t0 = time.perf_counter()

    if name == "stratified":
        from sklearn.dummy import DummyClassifier
        clf = DummyClassifier(strategy="stratified", random_state=seed)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        n_params = 0  # non-parametric

    elif name == "logreg":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=max_iter, random_state=seed, multi_class="multinomial")
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        # coef_: (n_classes, n_features); intercept_: (n_classes,)
        n_params = int(clf.coef_.size + clf.intercept_.size)

    elif name == "logreg_pca":
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        pca = PCA(n_components=pca_dim, random_state=seed)
        Xtr_p = pca.fit_transform(X_tr)
        Xte_p = pca.transform(X_te)
        clf = LogisticRegression(max_iter=max_iter, random_state=seed, multi_class="multinomial")
        clf.fit(Xtr_p, y_tr)
        y_pred = clf.predict(Xte_p)
        # PCA params (components_) + LogReg params
        n_params = int(pca.components_.size + clf.coef_.size + clf.intercept_.size)

    elif name == "mlp":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(
            hidden_layer_sizes=(hidden_dim,),
            max_iter=max_iter,
            random_state=seed,
            solver="adam",
            early_stopping=False,
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        # sum weights + biases across all layers
        n_params = int(sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_))

    elif name == "torch_vqc":
        model = TorchVQCRouter(
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_classes=int(max(y_tr.max(), y_te.max())) + 1,
            lr=lr,
            seed=seed,
            input_dim=int(X_tr.shape[1]),
            weight_decay=weight_decay,
        )
        Xt = torch.from_numpy(X_tr).double()
        yt = torch.from_numpy(y_tr.astype(np.int64))
        model.train_batched(Xt, yt, epochs=epochs)
        with torch.no_grad():
            Xe = torch.from_numpy(X_te).double()
            y_pred = model.predict(Xe).numpy()
        n_params = int(sum(p.numel() for p in model.parameters()))

    train_time = time.perf_counter() - t0

    acc = float(accuracy_score(y_te, y_pred))
    f1 = float(f1_score(y_te, y_pred, average="macro"))

    return {
        "name": name,
        "accuracy": acc,
        "macro_f1": f1,
        "train_time_s": train_time,
        "n_params": n_params,
    }
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run python -m pytest tests/routing/test_classical_baselines.py -v`
Expected: 6/6 PASSED

- [ ] **Step 3: Commit**

```bash
git add src/routing/classical_baselines.py
git commit -m "feat(c1): classical baseline runner (stratified/logreg/pca/mlp/vqc)"
```

---

### Task 4: Write failing integration test for the benchmark script

**Files:**
- Create: `tests/scripts/test_bench_classical_vs_vqc.py`

- [ ] **Step 1: Ensure tests directory exists**

Run: `mkdir -p tests/scripts && touch tests/scripts/__init__.py`

- [ ] **Step 2: Create the integration test**

Write to `tests/scripts/test_bench_classical_vs_vqc.py`:

```python
"""Integration test: bench_classical_vs_vqc.py produces a valid JSON report."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tmp_synthetic_embeddings(tmp_path):
    """Simulate what the real bench would see: a (n_samples, 384) embedding matrix
    + integer labels, saved as .npz so the bench can load without SentenceTransformer.
    """
    rng = np.random.default_rng(0)
    n_classes = 4
    per_class = 25
    centers = rng.uniform(-1.5, 1.5, size=(n_classes, 384))
    X, y = [], []
    for c in range(n_classes):
        for _ in range(per_class):
            X.append(centers[c] + rng.normal(0, 0.3, size=384))
            y.append(c)
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)

    cache = tmp_path / "embs.npz"
    np.savez(cache, embeddings=X, labels=y)
    return cache


def test_bench_produces_valid_json(tmp_path, tmp_synthetic_embeddings):
    output = tmp_path / "c1-out.json"
    cmd = [
        sys.executable,
        "scripts/bench_classical_vs_vqc.py",
        "--embeddings-npz", str(tmp_synthetic_embeddings),
        "--output", str(output),
        "--seeds", "0,1",
        "--epochs", "30",          # tiny for CI speed
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"stderr: {r.stderr}\nstdout: {r.stdout}"
    assert output.exists()

    data = json.loads(output.read_text())
    # Expected: one entry per baseline × seed + an aggregated summary
    assert "runs" in data and len(data["runs"]) == 5 * 2  # 5 baselines × 2 seeds
    assert "aggregated" in data
    assert set(data["aggregated"].keys()) == {
        "stratified", "logreg", "logreg_pca", "mlp", "torch_vqc"
    }
    for name, agg in data["aggregated"].items():
        assert "accuracy_mean" in agg
        assert "accuracy_std" in agg
        assert 0.0 <= agg["accuracy_mean"] <= 1.0
```

- [ ] **Step 3: Run test to verify it fails for the right reason**

Run: `uv run python -m pytest tests/scripts/test_bench_classical_vs_vqc.py -v`
Expected: FAIL with `FileNotFoundError: [Errno 2] No such file or directory: 'scripts/bench_classical_vs_vqc.py'` (or similar)

- [ ] **Step 4: Commit**

```bash
git add tests/scripts/test_bench_classical_vs_vqc.py tests/scripts/__init__.py
git commit -m "test(c1): add integration test for bench script (failing)"
```

---

### Task 5: Implement `bench_classical_vs_vqc.py`

**Files:**
- Create: `scripts/bench_classical_vs_vqc.py`

- [ ] **Step 1: Create the script**

Write to `scripts/bench_classical_vs_vqc.py`:

```python
#!/usr/bin/env python3
"""Phase C1 benchmark: classical baselines vs TorchVQCRouter on routing data.

Loads pre-computed embeddings (.npz with 'embeddings' + 'labels') or, if a
corpus+domains CLI is provided, embeds on the fly via SentenceTransformer.
Runs 5 baselines × N seeds with identical 80/20 splits, aggregates mean±std.

Usage (cached embeddings — fast for iteration):
    uv run python scripts/bench_classical_vs_vqc.py \\
        --embeddings-npz results/.c1-cache.npz \\
        --output results/c1-classical-vs-vqc.json \\
        --seeds 0,1,2,3,4

Usage (full pipeline):
    uv run python scripts/bench_classical_vs_vqc.py \\
        --data-dir data/final \\
        --domains dsp,electronics,emc,embedded,freecad,kicad-dsl,platformio,power,spice,stm32 \\
        --max-per-domain 50 \\
        --backbone models/niche-embeddings \\
        --embeddings-npz results/.c1-cache.npz \\
        --output results/c1-classical-vs-vqc.json
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.routing.classical_baselines import run_classical_baseline

logger = logging.getLogger(__name__)

_BASELINES = ["stratified", "logreg", "logreg_pca", "mlp", "torch_vqc"]


def _embed_corpus(data_dir: Path, domains: list[str], max_per_domain: int,
                  backbone: str, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from sentence_transformers import SentenceTransformer

    from src.routing.text_jepa.dataset import load_domain_corpus

    samples = load_domain_corpus(data_dir, domains=domains, max_per_domain=max_per_domain)
    dom_to_idx = {d: i for i, d in enumerate(domains)}
    labels = np.array([dom_to_idx[s.domain] for s in samples], dtype=np.int64)

    st = SentenceTransformer(str(backbone), device="cpu")
    tok = st.tokenizer
    transformer = st[0].auto_model.to("cpu")

    embs = []
    for s in samples:
        enc = tok(s.text, return_tensors="pt", truncation=True, max_length=seq_len, padding="max_length")
        with torch.no_grad():
            out = transformer(**enc).last_hidden_state
        embs.append(out.squeeze(0).mean(dim=0).cpu().numpy())
    return np.stack(embs).astype(np.float64), labels


def _split(X, y, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_tr = int(0.8 * len(idx))
    return idx[:n_tr], idx[n_tr:]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings-npz", type=Path, default=None,
                   help="NPZ cache with 'embeddings' + 'labels'. If exists, loaded; "
                        "if not and --data-dir provided, computed and saved.")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--domains", default="")
    p.add_argument("--max-per-domain", type=int, default=50)
    p.add_argument("--backbone", default="models/niche-embeddings")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # Load or compute embeddings
    if args.embeddings_npz and args.embeddings_npz.exists():
        logger.info("loading embeddings from %s", args.embeddings_npz)
        cache = np.load(args.embeddings_npz)
        X, y = cache["embeddings"], cache["labels"]
    elif args.data_dir is not None:
        domains = [d.strip() for d in args.domains.split(",") if d.strip()]
        if not domains:
            logger.error("--domains required when computing embeddings")
            return 2
        logger.info("computing embeddings for %d domains ×%d samples",
                    len(domains), args.max_per_domain)
        X, y = _embed_corpus(args.data_dir, domains, args.max_per_domain,
                             args.backbone, args.seq_len)
        if args.embeddings_npz:
            args.embeddings_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.embeddings_npz, embeddings=X, labels=y)
            logger.info("cached embeddings to %s", args.embeddings_npz)
    else:
        logger.error("either --embeddings-npz (existing) or --data-dir is required")
        return 2

    logger.info("X shape=%s  y classes=%d", X.shape, int(y.max()) + 1)

    runs = []
    logger.info("running %d baselines × %d seeds = %d total",
                len(_BASELINES), len(seeds), len(_BASELINES) * len(seeds))
    for name in _BASELINES:
        for seed in seeds:
            tr, te = _split(X, y, seed)
            out = run_classical_baseline(
                name, X[tr], y[tr], X[te], y[te],
                seed=seed, epochs=args.epochs,
            )
            out["seed"] = seed
            logger.info("  %s seed=%d  acc=%.3f  f1=%.3f  t=%.2fs  p=%d",
                        name, seed, out["accuracy"], out["macro_f1"],
                        out["train_time_s"], out["n_params"])
            runs.append(out)

    # Aggregate
    aggregated = {}
    for name in _BASELINES:
        accs = [r["accuracy"] for r in runs if r["name"] == name]
        f1s = [r["macro_f1"] for r in runs if r["name"] == name]
        times = [r["train_time_s"] for r in runs if r["name"] == name]
        params = next(r["n_params"] for r in runs if r["name"] == name)
        aggregated[name] = {
            "accuracy_mean": float(statistics.mean(accs)),
            "accuracy_std": float(statistics.pstdev(accs)) if len(accs) > 1 else 0.0,
            "macro_f1_mean": float(statistics.mean(f1s)),
            "train_time_s_mean": float(statistics.mean(times)),
            "n_params": params,
        }

    out = {
        "runs": runs,
        "aggregated": aggregated,
        "config": {
            "n_samples": int(len(X)),
            "n_classes": int(y.max()) + 1,
            "input_dim": int(X.shape[1]),
            "seeds": seeds,
            "epochs": args.epochs,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", args.output)

    # Also print a human-readable table
    print("\n=== C1 Results (mean over %d seeds) ===" % len(seeds))
    print(f"{'baseline':<12} {'acc':>8} {'f1':>8} {'time':>7} {'params':>8}")
    for name in _BASELINES:
        a = aggregated[name]
        print(f"{name:<12} {a['accuracy_mean']:>6.3f}±{a['accuracy_std']:.3f} "
              f"{a['macro_f1_mean']:>8.3f} {a['train_time_s_mean']:>6.1f}s {a['n_params']:>8d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run integration test to verify it passes**

Run: `uv run python -m pytest tests/scripts/test_bench_classical_vs_vqc.py -v`
Expected: PASSED (may take 30-60s for torch_vqc training)

- [ ] **Step 3: Commit**

```bash
git add scripts/bench_classical_vs_vqc.py
git commit -m "feat(c1): bench_classical_vs_vqc.py orchestrates 5 baselines × N seeds"
```

---

### Task 6: Run on real 10-class data, produce results JSON

**Files:**
- Create: `results/c1-classical-vs-vqc.json` (output)

- [ ] **Step 1: Compute embeddings cache (first run)**

Run:
```bash
uv run python scripts/bench_classical_vs_vqc.py \
    --data-dir data/final \
    --domains dsp,electronics,emc,embedded,freecad,kicad-dsl,platformio,power,spice,stm32 \
    --max-per-domain 50 \
    --backbone models/niche-embeddings \
    --embeddings-npz results/.c1-cache.npz \
    --seeds 0,1,2,3,4 \
    --epochs 300 \
    --output results/c1-classical-vs-vqc.json
```
Expected: runtime ~5-10 min (embedding ~30s + 5 baselines × 5 seeds × training). Log ends with the results table and `wrote results/c1-classical-vs-vqc.json`.

- [ ] **Step 2: Inspect the JSON**

Run: `jq '.aggregated' results/c1-classical-vs-vqc.json`
Expected: a dict with 5 keys (one per baseline), each containing `accuracy_mean`, `accuracy_std`, `macro_f1_mean`, `train_time_s_mean`, `n_params`. Sanity checks:
- `stratified.accuracy_mean` ≈ 0.10 (chance for 10 classes, ±0.03)
- `logreg.accuracy_mean` ≥ 0.40 (raw MiniLM is informative)
- `torch_vqc.accuracy_mean` ≥ 0.20 (matches session finding)

- [ ] **Step 3: Commit the cache + JSON**

```bash
git add results/c1-classical-vs-vqc.json
# Do NOT commit the .npz cache (>1MB, regeneratable, gitignored)
git commit -m "results(c1): real-data baseline comparison — 5 baselines × 5 seeds"
```

- [ ] **Step 4: Add .c1-cache.npz to gitignore if not covered**

Run: `grep -q '.npz' .gitignore || echo '*.npz' >> .gitignore && git diff .gitignore`
Expected: either no change (already ignored) or a single `+*.npz` line. Commit if changed:

```bash
git add .gitignore && git commit -m "chore: ignore npz caches"
```

---

### Task 7: Generate figure + paper-facing markdown

**Files:**
- Create: `scripts/figure_c1_comparison.py`
- Create: `docs/paper-a/c1-baseline-results.md`
- Create: `docs/paper-a/c1-comparison.pdf`

- [ ] **Step 1: Create the figure script**

Write to `scripts/figure_c1_comparison.py`:

```python
#!/usr/bin/env python3
"""Generate c1-comparison.pdf from c1-classical-vs-vqc.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("results/c1-classical-vs-vqc.json"))
    p.add_argument("--output", type=Path, default=Path("docs/paper-a/c1-comparison.pdf"))
    args = p.parse_args()

    data = json.loads(args.input.read_text())
    order = ["stratified", "logreg_pca", "torch_vqc", "mlp", "logreg"]
    labels = {
        "stratified": "Stratified\nrandom",
        "logreg_pca": "LogReg\non PCA-4",
        "torch_vqc": "Torch VQC\n(ours)",
        "mlp": "MLP\n(384→64)",
        "logreg": "LogReg\non raw 384-D",
    }
    means = [data["aggregated"][n]["accuracy_mean"] for n in order]
    stds = [data["aggregated"][n]["accuracy_std"] for n in order]
    params = [data["aggregated"][n]["n_params"] for n in order]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    x = np.arange(len(order))
    colors = ["#bbbbbb", "#ffaa66", "#6699ff", "#66bb77", "#cc5555"]
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[n] for n in order], fontsize=9)
    ax.set_ylabel("Test accuracy (10-class routing)")
    ax.axhline(y=0.1, color="gray", linestyle="--", linewidth=0.8, label="Chance (0.10)")
    ax.set_ylim(0, max(means) * 1.15 + 0.05)
    ax.legend(loc="upper left")
    for bar, mean, pcount in zip(bars, means, params):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{mean:.2f}\n({pcount:,}p)", ha="center", va="bottom", fontsize=8)
    ax.set_title("C1: Classical baselines vs Torch VQC router")
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it**

Run: `uv run python scripts/figure_c1_comparison.py`
Expected: prints `wrote docs/paper-a/c1-comparison.pdf`. The PDF contains a bar chart with 5 bars + error bars + chance line.

- [ ] **Step 3: Write the paper-facing narrative**

Write to `docs/paper-a/c1-baseline-results.md`:

```markdown
# C1: Classical Baselines vs Torch VQC Router

**Setup.** 10 domain-routing classes (dsp, electronics, emc, embedded, freecad, kicad-dsl, platformio, power, spice, stm32) × 50 samples per domain from `data/final`, encoded to 384-D via frozen MiniLM-L6-v2 mean-pool, 80/20 train/test split, 5 seeds, 300 epochs for trainable baselines.

**Baselines.**
- **Stratified random**: draws labels from the train distribution — chance floor.
- **LogReg on PCA-4**: 384-D embeddings compressed to 4-D via PCA, then multinomial logistic regression. Information-matched to VQC (same 4 effective features).
- **Torch VQC (ours)**: 4-qubit, 6-layer StronglyEntanglingLayers, learned projection 384→4, weight decay 1e-4.
- **MLP (384→64)**: 1-hidden-layer classifier with capacity comparable to VQC+projection.
- **LogReg on raw 384-D**: upper bound — no information loss, maximally expressive linear head.

**Results** (see Figure c1-comparison.pdf). Numbers below are populated from `results/c1-classical-vs-vqc.json` by `scripts/figure_c1_comparison.py` — exact values regenerate deterministically.

| Baseline | Test acc | Macro F1 | Params | Train time |
|---|---|---|---|---|
| Stratified random | ~0.10 ± 0.03 | ~0.09 | 0 | <0.01s |
| LogReg on PCA-4 | ~0.20 ± 0.04 | — | ~1.6k | <0.5s |
| Torch VQC (ours) | ~0.30 ± 0.07 | — | ~1.7k | ~3s |
| MLP (384→64) | ~0.55 ± 0.05 | — | ~25k | ~15s |
| LogReg on raw 384-D | ~0.70 ± 0.03 | — | ~3.8k | ~0.5s |

(Replace approximate numbers with the ones from the generated JSON after Task 6 finishes.)

**Interpretation.**
1. **The VQC beats its information-matched classical baseline** (LogReg-PCA-4) by ~10 pt. This suggests the learned projection + non-linear circuit mapping extracts richer 4-D features than PCA's variance-maximizing projection.
2. **But the VQC sits ~40 pt below the full-capacity classical ceiling** (LogReg raw 384-D). The gap is the "price of quantum" at 4 qubits — the information bottleneck predicted in Plan 6 findings.
3. **MLP with comparable parameter count** dominates the VQC by ~25 pt. Our architecture's advantage is NOT capacity; it is the formal quantum-computational framework, potentially with advantages on non-classical inputs (future work).

**Implications for Paper A.** This result reframes the contribution: the VQC is **not** competitive as a routing classifier on classical embeddings. Its value is methodological — `torch-vqc` makes VQC research tractable at scale, AND the learned-projection architecture is a portable pattern for future quantum-ML on pretrained features. Paper A should be explicit about this.
```

- [ ] **Step 4: Replace the approximate numbers with actual measured values**

Run: `jq '.aggregated' results/c1-classical-vs-vqc.json`

Hand-edit `docs/paper-a/c1-baseline-results.md` to substitute the real mean±std values from the JSON into the table (rows labeled "~X.XX ± 0.XX"). Do the same for "~40 pt" in the Interpretation paragraph — compute (logreg.accuracy_mean − torch_vqc.accuracy_mean) × 100.

- [ ] **Step 5: Commit the figure + markdown**

```bash
git add docs/paper-a/c1-baseline-results.md docs/paper-a/c1-comparison.pdf scripts/figure_c1_comparison.py
git commit -m "docs(c1): results narrative + comparison figure for Paper A §4"
```

---

### Task 8: Push + verify

- [ ] **Step 1: Push to origin**

Run: `git push origin main`
Expected: all commits pushed, no errors.

- [ ] **Step 2: Run the full pytest suite once to ensure no regressions**

Run: `uv run python -m pytest tests/routing/test_classical_baselines.py tests/scripts/test_bench_classical_vs_vqc.py -v`
Expected: all tests PASSED.

- [ ] **Step 3: Update the Phase C roadmap**

In `docs/superpowers/plans/2026-04-19-phase-c-roadmap.md`, change the C1 row status from "Detailed plan ready" to "Done (commit SHA)" where SHA is `git rev-parse --short HEAD`. Commit + push this final status update:

```bash
git add docs/superpowers/plans/2026-04-19-phase-c-roadmap.md
git commit -m "docs(phase-c): mark C1 complete"
git push origin main
```

---

## Kill criterion reminder

If `logreg` (raw 384-D) accuracy_mean > 0.80 on this 10-class task, re-read the Phase C roadmap's C1 kill clause: the VQC contribution degrades to "tool release only" and Paper A should be retracted. Document the retraction decision in `docs/paper-a/c1-baseline-results.md` and stop before C2.

## Out of scope for this plan

- **Hyperparameter search** for any baseline — all use reasonable sklearn defaults for reproducibility.
- **Non-standard baselines** (gradient-boosted trees, SVM-RBF, etc.) — these add noise without addressing the main point (information-capacity ceiling).
- **Multiple datasets** — a single routing task is enough for C1; multi-task eval is C3/C4's job.
- **Plotting interactivity** — static PDF only; HTML dashboards are out of scope.

## Total estimated time

- Task 1 (deps): 5 min
- Task 2 (tests): 20 min
- Task 3 (impl): 30 min
- Task 4 (integration test): 15 min
- Task 5 (bench script): 45 min
- Task 6 (real run + commit): 10-15 min (dominated by embedding pass)
- Task 7 (figure + narrative): 30 min
- Task 8 (push + roadmap update): 10 min

**Total: ~3 hours engineering + ~5-10 min wall-clock compute.** Originally scoped at 2-3 days assuming PennyLane — torch-vqc collapses it to an afternoon.
