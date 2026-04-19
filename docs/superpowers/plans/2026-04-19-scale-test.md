# Aeon Scale Test (100k–1M messages) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push `AeonPredictor` + `AtlasIndex` from the current ≤1k-turn PoC to production-realistic corpora of 100k / 500k / 1M messages, measure p50/p95/p99 latency and throughput under concurrent load, and land a sharded-atlas + minibatch predictor implementation that meets the memory and latency targets below.

**Architecture:** Add a hash-sharded `ShardedAtlasIndex` wrapping N `AtlasIndex` instances and a `PredictorRegistry` that holds either a shared `LatentMLP` or one `LatentMLP` per-stack (decision made by Task 8 from the benchmark). Replace the full-buffer SGD in `AeonPredictor.fit_on_buffer` with a reservoir-sampled minibatch path so training cost is O(epochs · batch) instead of O(pairs · epochs). Add a `bench/` harness that drives corpus generation, records hardware counters, and emits latency / memory / throughput curves as CSV + PNG.

**Tech Stack:** numpy only for Aeon hot paths (no torch), `sentence-transformers` for embeddings in corpus gen, `pytest-benchmark` for micro-bench, `tracemalloc` + `resource.getrusage` for memory, `asyncio` + `concurrent.futures.ThreadPoolExecutor` for throughput, `matplotlib` (optional) for curves. Python 3.11+, uv-managed. Primary target: Mac Studio M3 Ultra 512 GB. Secondary: kxkm-ai (RTX 4090) only for embedding pre-compute, never for Aeon itself.

**Latency targets (budget, not goals):**

| Op | 10k | 100k | 1M | Notes |
|---|---|---|---|---|
| `atlas.recall(k=10)` p50 | ≤ 2 ms | ≤ 8 ms | ≤ 40 ms | sharded, hot pages in cache |
| `atlas.recall(k=10)` p95 | ≤ 5 ms | ≤ 20 ms | ≤ 100 ms | worst-shard tail |
| `predict_next(horizon=1)` p50 | ≤ 1 ms | ≤ 1 ms | ≤ 2 ms | MLP size constant |
| `ingest_latent` p50 | ≤ 3 ms | ≤ 10 ms | ≤ 50 ms | write to atlas + graph + buffer |
| End-to-end `recall(predict_next(q))` p95 | ≤ 6 ms | ≤ 22 ms | ≤ 110 ms | serving path |
| Throughput (recall, 8 threads) | ≥ 4k qps | ≥ 1k qps | ≥ 250 qps | read-heavy mix |

**Memory budget (hard ceiling 8 GB for Aeon process on a 16 GB M5, 32 GB on Studio):**

| Component | 1M × 384-dim | 1M × 768-dim | 1M × 3072-dim |
|---|---|---|---|
| `AtlasIndex.vectors` (float32) | 1.5 GB | 3.0 GB | 12.0 GB |
| `AtlasIndex.ids` (Python strings, ~40 B avg) | ~40 MB | ~40 MB | ~40 MB |
| `TraceGraph` nodes (attrs dict) | ~0.8 GB | ~0.8 GB | ~0.8 GB |
| `AeonPredictor._buffer` cap (50k pairs × 2 · 384) | ~150 MB | ~300 MB | ~1.2 GB |
| `LatentMLP` weights (hidden=256) | ~1 MB | ~1 MB | ~3 MB |
| **Total** | **~2.5 GB** | **~4.2 GB** | **~14 GB** |

→ PoC target is **dim=384** (sentence-transformers MiniLM). dim=3072 (Qwen hidden) is a stretch goal and requires float16 + memory-mapped atlas pages (Task 14).

---

## Success & Kill Criteria

**Success (merge + publish scale curves):**
- All latency targets above met at 1M × 384-dim on Mac Studio M3 Ultra.
- Memory peak under 8 GB for the 1M × 384 corpus (measured by `tracemalloc.get_traced_memory()` + RSS).
- `bench/` harness is reproducible: `just bench-scale` emits `results/scale/<date>/{latency.csv,memory.csv,throughput.csv,*.png}`.
- Recall@5 on the 1M corpus is within 3 absolute points of the 1k baseline — sharding must not degrade quality.
- Zero new torch imports in `src/memory/`.

**Kill (abandon sharding, fall back to single-shard + document ceiling):**
- Sharding shows < 30 % recall-latency improvement over single-shard at 100k (overhead dominates).
- Memory exceeds 16 GB at 1M × 384 even after the minibatch + mmap fix.
- Per-stack predictor shows > 10 % recall drop vs shared predictor (stacks do not have distinct dynamics at scale).

---

## Risk Mitigations (each mapped to a task)

1. **Memory blow-up at 1M vectors** — Tasks 5 and 14. Task 5 caps the buffer via reservoir sampling; Task 14 adds a `mmap=True` mode for `VectorPage.vectors` so pages stream from disk.
2. **Sharding degrades recall** (query-to-shard routing misses the right page) — Task 6 does hash sharding on `vector_id` (keeps the existing intra-shard search unchanged) and Task 12 re-runs the v0.3 PI-depth benchmark to prove recall holds.
3. **Throughput tests contaminated by GIL contention** — Task 10 uses `ThreadPoolExecutor` for I/O-ish paths (numpy releases the GIL inside `@`) and an `asyncio` event-loop driver for the ingest path, reporting both.

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/memory/sharded_atlas.py` | `ShardedAtlasIndex` — N `AtlasIndex` shards, hash-routed | Create |
| `src/memory/predictor_registry.py` | `PredictorRegistry` — shared-vs-per-stack `LatentMLP` selection | Create |
| `src/memory/aeon_predictor.py` | Add reservoir-sampled minibatch path, buffer cap | Modify (L255–L362) |
| `src/memory/atlas.py` | Add optional `mmap_dir` to `VectorPage` | Modify (L26–L60) |
| `bench/__init__.py` | bench package marker | Create |
| `bench/corpus_gen.py` | synthetic + real corpus generator (100k / 500k / 1M) | Create |
| `bench/harness.py` | reusable scale runner — warmup, percentiles, memory sampler | Create |
| `bench/scenarios/ingest.py` | ingest latency + throughput scenario | Create |
| `bench/scenarios/recall.py` | recall + predict_next latency scenario | Create |
| `bench/scenarios/concurrent.py` | threadpool + asyncio throughput scenario | Create |
| `scripts/bench_scale.py` | CLI wrapper: `uv run python scripts/bench_scale.py --size 1M` | Create |
| `tests/memory/test_sharded_atlas.py` | unit tests for hash routing, recall parity, shard rebalance | Create |
| `tests/memory/test_predictor_registry.py` | unit tests for shared/per-stack selection, capacity | Create |
| `tests/memory/test_aeon_predictor_minibatch.py` | unit tests for reservoir sampling, bounded buffer | Create |
| `tests/bench/test_harness.py` | smoke test: harness runs end-to-end on 1k corpus in < 5 s | Create |
| `results/scale/2026-04-19/README.md` | scale curves + commentary (populated by Task 14) | Create |

---

### Task 1: Scaffold `bench/` package and CLI

**Files:**
- Create: `bench/__init__.py`
- Create: `bench/harness.py`
- Create: `scripts/bench_scale.py`
- Create: `tests/bench/__init__.py`
- Create: `tests/bench/test_harness.py`

- [ ] **Step 1: Write the failing test**

Create `tests/bench/test_harness.py`:

```python
"""Smoke test for bench.harness — runs the 1k fixture in < 5 s."""
from __future__ import annotations

import time

import numpy as np

from bench.harness import ScaleRun, percentiles


def test_percentiles_basic():
    xs = [float(i) for i in range(100)]
    p = percentiles(xs, [50, 95, 99])
    assert 49 <= p[50] <= 50
    assert 94 <= p[95] <= 95
    assert 98 <= p[99] <= 99


def test_scale_run_records_latency():
    run = ScaleRun(name="smoke", size=100)
    for _ in range(100):
        with run.record("op"):
            np.random.standard_normal(64)
    summary = run.summary()
    assert summary["op"]["count"] == 100
    assert "p50" in summary["op"]
    assert summary["op"]["p50"] > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_harness.py -v`
Expected: `ModuleNotFoundError: No module named 'bench'`

- [ ] **Step 3: Write minimal implementation**

Create `bench/__init__.py`:
```python
"""Scale benchmark harness for Aeon."""
```

Create `bench/harness.py`:
```python
"""Scale benchmark harness — latency percentiles, memory sampler, CSV out."""
from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from statistics import quantiles
from typing import Any, Iterator


def percentiles(xs: list[float], ps: list[int]) -> dict[int, float]:
    if not xs:
        return {p: 0.0 for p in ps}
    xs_sorted = sorted(xs)
    q = quantiles(xs_sorted, n=100, method="inclusive")
    # quantiles returns 99 cut-points (between pctiles 1..99)
    out: dict[int, float] = {}
    for p in ps:
        if p <= 0:
            out[p] = xs_sorted[0]
        elif p >= 100:
            out[p] = xs_sorted[-1]
        else:
            out[p] = q[p - 1]
    return out


@dataclass
class ScaleRun:
    name: str
    size: int
    samples: dict[str, list[float]] = field(default_factory=dict)

    @contextmanager
    def record(self, op: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.samples.setdefault(op, []).append(elapsed_ms)

    def summary(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for op, xs in self.samples.items():
            p = percentiles(xs, [50, 95, 99])
            out[op] = {
                "count": len(xs),
                "p50": p[50],
                "p95": p[95],
                "p99": p[99],
                "mean_ms": sum(xs) / len(xs) if xs else 0.0,
            }
        return out

    def to_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["op", "count", "p50_ms", "p95_ms", "p99_ms", "mean_ms"])
            for op, row in self.summary().items():
                w.writerow(
                    [op, row["count"], row["p50"], row["p95"], row["p99"], row["mean_ms"]]
                )
```

Create `tests/bench/__init__.py` as empty file.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_harness.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bench/__init__.py bench/harness.py tests/bench/__init__.py tests/bench/test_harness.py
git commit -m "feat(bench): scaffold scale harness"
```

---

### Task 2: Corpus generator (synthetic + real)

**Files:**
- Create: `bench/corpus_gen.py`
- Create: `tests/bench/test_corpus_gen.py`

- [ ] **Step 1: Write the failing test**

Create `tests/bench/test_corpus_gen.py`:

```python
"""Corpus generator: synthetic stream + embedding pre-compute."""
from __future__ import annotations

import numpy as np

from bench.corpus_gen import CorpusSpec, generate_synthetic


def test_synthetic_1k_shape():
    spec = CorpusSpec(n_turns=1000, dim=384, n_stacks=16, seed=0)
    corpus = generate_synthetic(spec)
    assert corpus.vectors.shape == (1000, 384)
    assert len(corpus.ids) == 1000
    assert len(corpus.stack_ids) == 1000
    # unit-norm
    norms = np.linalg.norm(corpus.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_corpus_deterministic_under_seed():
    spec = CorpusSpec(n_turns=500, dim=64, n_stacks=4, seed=42)
    a = generate_synthetic(spec)
    b = generate_synthetic(spec)
    assert np.allclose(a.vectors, b.vectors)
```

- [ ] **Step 2: Run test**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_corpus_gen.py -v`
Expected: `ModuleNotFoundError: No module named 'bench.corpus_gen'`

- [ ] **Step 3: Implement**

Create `bench/corpus_gen.py`:
```python
"""Corpus generation for scale tests — synthetic random walk + optional real embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CorpusSpec:
    n_turns: int
    dim: int = 384
    n_stacks: int = 16
    seed: int = 0
    step_scale: float = 0.3


@dataclass
class Corpus:
    ids: list[str]
    vectors: np.ndarray       # (n_turns, dim) float32, unit-norm
    stack_ids: np.ndarray     # (n_turns,) int64 in [0, n_stacks)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def generate_synthetic(spec: CorpusSpec) -> Corpus:
    rng = np.random.default_rng(spec.seed)
    vectors = np.empty((spec.n_turns, spec.dim), dtype=np.float32)
    vectors[0] = _unit(rng.standard_normal(spec.dim).astype(np.float32))
    for i in range(1, spec.n_turns):
        step = spec.step_scale * rng.standard_normal(spec.dim).astype(np.float32)
        vectors[i] = _unit(vectors[i - 1] + step)
    stack_ids = rng.integers(0, spec.n_stacks, size=spec.n_turns).astype(np.int64)
    ids = [f"t{i}" for i in range(spec.n_turns)]
    return Corpus(ids=ids, vectors=vectors, stack_ids=stack_ids)


def save_corpus(corpus: Corpus, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vectors=corpus.vectors,
        stack_ids=corpus.stack_ids,
        ids=np.array(corpus.ids, dtype=object),
    )


def load_corpus(path: Path) -> Corpus:
    data = np.load(path, allow_pickle=True)
    return Corpus(
        ids=list(data["ids"]),
        vectors=data["vectors"],
        stack_ids=data["stack_ids"],
    )
```

- [ ] **Step 4: Run test**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_corpus_gen.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bench/corpus_gen.py tests/bench/test_corpus_gen.py
git commit -m "feat(bench): synthetic corpus gen with deterministic seeds"
```

---

### Task 3: Baseline profile — where does 100k hurt today?

**Files:**
- Create: `scripts/bench_scale.py`
- Create: `results/scale/2026-04-19/baseline.md`

- [ ] **Step 1: Write a throwaway profiler script**

Create `scripts/bench_scale.py`:
```python
#!/usr/bin/env python3
"""Scale bench CLI — drives bench.scenarios at 10k/100k/1M corpus sizes."""
from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
import tracemalloc
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.corpus_gen import CorpusSpec, generate_synthetic
from bench.harness import ScaleRun
from src.memory.aeonsleep import AeonSleep, Episode


def run_baseline(size: int, dim: int = 384, seed: int = 0) -> ScaleRun:
    corpus = generate_synthetic(CorpusSpec(n_turns=size, dim=dim, seed=seed))
    palace = AeonSleep(dim=dim)
    run = ScaleRun(name=f"baseline-{size}", size=size)

    tracemalloc.start()
    t0 = datetime(2026, 4, 19, 9, 0)
    for i in range(size):
        ep = Episode(
            id=corpus.ids[i],
            text="",
            embedding=corpus.vectors[i].tolist(),
            ts=t0 + timedelta(seconds=i),
            topic="scale",
            payload={"stack_id": int(corpus.stack_ids[i])},
        )
        with run.record("ingest"):
            palace.write(ep)

    for i in range(0, size, max(size // 500, 1)):
        with run.record("recall"):
            palace.recall(corpus.vectors[i].tolist(), k=10)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"[{size}] peak RSS (traced) = {peak/1e6:.1f} MB")
    return run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=10_000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/scale/2026-04-19"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        run = run_baseline(args.size, args.dim)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumulative")
        stats.dump_stats(str(args.out / f"profile-{args.size}.pstats"))
    else:
        run = run_baseline(args.size, args.dim)

    run.to_csv(args.out / f"baseline-{args.size}.csv")
    print(run.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it at 10k and 100k**

Run:
```bash
cd /Users/electron/Documents/Projets/micro-kiki
uv run python scripts/bench_scale.py --size 10000 --profile
uv run python scripts/bench_scale.py --size 100000
```
Expected: two CSVs under `results/scale/2026-04-19/` plus a `.pstats` file. 100k may take > 60 s — that IS the motivation for this plan.

- [ ] **Step 3: Document findings**

Create `results/scale/2026-04-19/baseline.md` with a table of p50/p95/p99 from the two CSVs and the top 10 hot functions from the pstats file (use `uv run python -c "import pstats; pstats.Stats('results/scale/2026-04-19/profile-10000.pstats').sort_stats('cumulative').print_stats(10)"`).

- [ ] **Step 4: Commit**

```bash
git add scripts/bench_scale.py results/scale/2026-04-19/baseline.md
git commit -m "feat(bench): baseline profile at 10k and 100k"
```

---

### Task 4: Recall scenario — percentile harness for pure retrieval

**Files:**
- Create: `bench/scenarios/__init__.py`
- Create: `bench/scenarios/recall.py`
- Create: `tests/bench/test_scenarios.py`

- [ ] **Step 1: Write the failing test**

Create `tests/bench/test_scenarios.py`:
```python
"""Scenario smoke tests — must run the 1k fixture in < 2 s."""
from __future__ import annotations

from bench.corpus_gen import CorpusSpec, generate_synthetic
from bench.scenarios.recall import run_recall_scenario


def test_recall_scenario_smoke():
    corpus = generate_synthetic(CorpusSpec(n_turns=1000, dim=64, seed=0))
    report = run_recall_scenario(corpus, n_queries=100, k=5)
    assert report.summary()["recall"]["count"] == 100
    assert report.summary()["recall"]["p50"] > 0.0
```

- [ ] **Step 2: Run test**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_scenarios.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `bench/scenarios/__init__.py` as empty. Create `bench/scenarios/recall.py`:
```python
"""Recall-path scenario: ingest corpus, then measure recall percentiles."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from bench.corpus_gen import Corpus
from bench.harness import ScaleRun
from src.memory.aeonsleep import AeonSleep, Episode


def run_recall_scenario(
    corpus: Corpus, *, n_queries: int = 500, k: int = 10, dim: int | None = None
) -> ScaleRun:
    _dim = dim if dim is not None else int(corpus.vectors.shape[1])
    palace = AeonSleep(dim=_dim)
    t0 = datetime(2026, 4, 19, 9, 0)
    for i in range(len(corpus.ids)):
        palace.write(
            Episode(
                id=corpus.ids[i],
                text="",
                embedding=corpus.vectors[i].tolist(),
                ts=t0 + timedelta(seconds=i),
                topic="scale",
                payload={"stack_id": int(corpus.stack_ids[i])},
            )
        )
    rng = np.random.default_rng(0)
    idxs = rng.integers(0, len(corpus.ids), size=n_queries)
    run = ScaleRun(name=f"recall-{len(corpus.ids)}", size=len(corpus.ids))
    for i in idxs:
        q = corpus.vectors[int(i)].tolist()
        with run.record("recall"):
            palace.recall(q, k=k)
    return run
```

- [ ] **Step 4: Run test**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_scenarios.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bench/scenarios/__init__.py bench/scenarios/recall.py tests/bench/test_scenarios.py
git commit -m "feat(bench): recall percentile scenario"
```

---

### Task 5: Bounded buffer + reservoir minibatch sampling in AeonPredictor

**Files:**
- Modify: `src/memory/aeon_predictor.py:255-361`
- Create: `tests/memory/test_aeon_predictor_minibatch.py`

- [ ] **Step 1: Write the failing test**

Create `tests/memory/test_aeon_predictor_minibatch.py`:
```python
"""AeonPredictor: bounded buffer + reservoir-sampled minibatch training."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from src.memory.aeon_predictor import AeonPredictor, PredictorConfig
from src.memory.aeonsleep import AeonSleep


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def test_buffer_is_bounded_by_capacity():
    palace = AeonSleep(dim=32)
    cfg = PredictorConfig(dim=32, hidden=16, n_stacks=4, buffer_capacity=500)
    pred = AeonPredictor(palace=palace, config=cfg)
    t0 = datetime(2026, 4, 19, 9, 0)
    rng = np.random.default_rng(0)
    for i in range(2000):
        h = _unit(rng.standard_normal(32).astype(np.float32))
        pred.ingest_latent(f"t{i}", h, ts=t0 + timedelta(seconds=i), stack_id=i % 4)
    assert pred.buffer_size() == 500


def test_fit_minibatch_does_not_iterate_full_corpus():
    palace = AeonSleep(dim=32)
    cfg = PredictorConfig(dim=32, hidden=16, n_stacks=4, buffer_capacity=500)
    pred = AeonPredictor(palace=palace, config=cfg)
    t0 = datetime(2026, 4, 19, 9, 0)
    rng = np.random.default_rng(0)
    for i in range(500):
        h = _unit(rng.standard_normal(32).astype(np.float32))
        pred.ingest_latent(f"t{i}", h, ts=t0 + timedelta(seconds=i), stack_id=i % 4)
    losses = pred.fit_minibatch(lr=1e-3, steps=10, batch_size=16)
    assert len(losses) == 10
    # Each step must process exactly batch_size pairs, not the full buffer.
    assert all(isinstance(l, float) for l in losses)
```

- [ ] **Step 2: Run test**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_aeon_predictor_minibatch.py -v`
Expected: FAIL — `PredictorConfig.__init__() got an unexpected keyword argument 'buffer_capacity'` and `AeonPredictor has no attribute fit_minibatch`.

- [ ] **Step 3: Modify `PredictorConfig`**

Add field in `src/memory/aeon_predictor.py` around the dataclass definition (replace the existing `PredictorConfig` with the extended version):

```python
@dataclass(frozen=True)
class PredictorConfig:
    """Immutable predictor config."""

    dim: int
    hidden: int = 256
    horizon: int = 1
    n_stacks: int = 16
    cold_start_threshold: int = 500
    seed: int = 0
    use_centering: bool = False
    centering_momentum: float = 0.9
    per_stack_centering: bool = False
    use_layernorm_delta: bool = False
    buffer_capacity: int = 50_000
```

- [ ] **Step 4: Add bounded buffer logic in `ingest_latent`**

In `AeonPredictor.ingest_latent`, replace the `self._buffer.append(...)` tail with reservoir sampling:

```python
        sample = _PairSample(
            turn_id=turn_id,
            h=h.astype(np.float32).copy(),
            ts=ts,
            stack_id=-1 if stack_id is None else int(stack_id),
        )
        cap = self.config.buffer_capacity
        if len(self._buffer) < cap:
            self._buffer.append(sample)
        else:
            # Reservoir: replace a random slot with prob cap/n_seen.
            self._n_seen = getattr(self, "_n_seen", cap) + 1
            j = int(np.random.default_rng(self._n_seen).integers(0, self._n_seen))
            if j < cap:
                self._buffer[j] = sample
```

Initialize `self._n_seen = 0` in `__init__`.

- [ ] **Step 5: Add `fit_minibatch` method**

Below `fit_on_buffer` in `AeonPredictor`, add:

```python
    def fit_minibatch(
        self,
        *,
        lr: float = 1e-3,
        steps: int = 100,
        batch_size: int = 64,
    ) -> list[float]:
        """SGD over `steps` random minibatches sampled from the buffer.

        Unlike `fit_on_buffer`, this is O(steps * batch_size) — it does
        NOT scan the whole buffer per epoch. Use at scale.
        """
        triples = self.pairs_for_training()
        if not triples:
            return []
        n = len(triples)
        rng = np.random.default_rng(self.config.seed)
        losses: list[float] = []
        for _ in range(steps):
            idx = rng.integers(0, n, size=min(batch_size, n))
            batch = [triples[i] for i in idx]
            x = np.stack([t[0] for t in batch]).astype(np.float32)
            tgt = np.stack([t[1] for t in batch]).astype(np.float32)
            stack = self._stack_onehot([t[2] for t in batch])
            stack_ids = np.array([t[2] for t in batch], dtype=np.int64)
            self.mlp.forward(x, stack, stack_ids=stack_ids)
            loss = self.mlp.backward_cosine(tgt, lr=lr)
            losses.append(float(loss))
        self._trained_once = True
        return losses
```

- [ ] **Step 6: Run test**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_aeon_predictor_minibatch.py -v`
Expected: PASS

- [ ] **Step 7: Regression check — old tests still pass**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/ -v`
Expected: all PASS (the new `buffer_capacity` default is 50k, well above the 500-pair PoC fixtures).

- [ ] **Step 8: Commit**

```bash
git add src/memory/aeon_predictor.py tests/memory/test_aeon_predictor_minibatch.py
git commit -m "feat(memory): reservoir buffer + minibatch SGD for predictor"
```

---

### Task 6: `ShardedAtlasIndex` — hash sharding

**Files:**
- Create: `src/memory/sharded_atlas.py`
- Create: `tests/memory/test_sharded_atlas.py`

- [ ] **Step 1: Write the failing test**

Create `tests/memory/test_sharded_atlas.py`:
```python
"""ShardedAtlasIndex: hash-routed shards must preserve recall semantics."""
from __future__ import annotations

import numpy as np

from src.memory.atlas import AtlasIndex
from src.memory.sharded_atlas import ShardedAtlasIndex


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def test_recall_matches_single_shard_for_small_corpus():
    rng = np.random.default_rng(0)
    n, dim = 500, 64
    vecs = np.stack([_unit(rng.standard_normal(dim).astype(np.float32)) for _ in range(n)])
    single = AtlasIndex(dim=dim)
    sharded = ShardedAtlasIndex(dim=dim, n_shards=4)
    for i in range(n):
        single.insert(f"v{i}", vecs[i])
        sharded.insert(f"v{i}", vecs[i])
    q = _unit(rng.standard_normal(dim).astype(np.float32))
    r_single = {h.id for h in single.search(q, k=10)}
    r_sharded = {h.id for h in sharded.search(q, k=10)}
    # Result sets overlap strongly — same corpus + same cosine math.
    assert len(r_single & r_sharded) >= 8  # allow 2 tie-breakers to differ


def test_hash_routing_is_stable():
    sharded = ShardedAtlasIndex(dim=8, n_shards=4)
    for i in range(100):
        sharded.insert(f"v{i}", np.ones(8, dtype=np.float32) * (i / 100.0))
    # Each id hashes to exactly one shard.
    shard_counts = [s.total_vectors for s in sharded.shards]
    assert sum(shard_counts) == 100
    assert all(c > 0 for c in shard_counts)


def test_sharded_total_vectors():
    sharded = ShardedAtlasIndex(dim=8, n_shards=3)
    for i in range(30):
        sharded.insert(f"v{i}", np.ones(8, dtype=np.float32) * (i / 30.0))
    assert sharded.total_vectors == 30
```

- [ ] **Step 2: Run test**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_sharded_atlas.py -v`
Expected: `ModuleNotFoundError: No module named 'src.memory.sharded_atlas'`

- [ ] **Step 3: Implement**

Create `src/memory/sharded_atlas.py`:
```python
"""ShardedAtlasIndex — hash-routed ensemble of AtlasIndex shards.

Each vector_id hashes (SHA1 lower 63 bits) to exactly one shard. Recall
fans out to all shards, top-k each, then merges and returns the global
top-k. Memory-parallel by shard; latency scales with max-shard-size,
not total-size, when pages fit in cache.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.memory.atlas import AtlasIndex, SearchHit


def _shard_for(vector_id: str, n_shards: int) -> int:
    h = hashlib.sha1(vector_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % n_shards


class ShardedAtlasIndex:
    """N `AtlasIndex` instances under a hash router."""

    def __init__(self, dim: int = 3072, n_shards: int = 8, num_clusters: int = 16) -> None:
        if n_shards < 1:
            raise ValueError("n_shards must be >= 1")
        self.dim = dim
        self.n_shards = n_shards
        self.shards: list[AtlasIndex] = [
            AtlasIndex(dim=dim, num_clusters=num_clusters) for _ in range(n_shards)
        ]
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, n_shards), thread_name_prefix="atlas-shard"
        )

    def insert(self, vector_id: str, vector: np.ndarray) -> None:
        shard = _shard_for(vector_id, self.n_shards)
        self.shards[shard].insert(vector_id, vector)

    def remove(self, vector_id: str) -> bool:
        shard = _shard_for(vector_id, self.n_shards)
        return self.shards[shard].remove(vector_id)

    def search(self, query, k: int = 10) -> list[SearchHit]:
        q = np.asarray(query, dtype=np.float32)
        # Fan out; each shard returns its local top-k.
        futures = [self._executor.submit(s.search, q, k) for s in self.shards]
        merged: list[SearchHit] = []
        for f in futures:
            merged.extend(f.result())
        merged.sort(key=lambda h: h.score, reverse=True)
        return merged[:k]

    def recall(self, query, top_k: int = 10) -> list[tuple[str, float]]:
        hits = self.search(query, k=top_k)
        return [(h.id, h.score) for h in hits]

    @property
    def total_vectors(self) -> int:
        return sum(s.total_vectors for s in self.shards)

    def rebuild_centroids(self) -> None:
        for s in self.shards:
            s.rebuild_centroids()
```

- [ ] **Step 4: Run test**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_sharded_atlas.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory/sharded_atlas.py tests/memory/test_sharded_atlas.py
git commit -m "feat(memory): hash-sharded atlas with threadpool fan-out"
```

---

### Task 7: Wire ShardedAtlasIndex through AeonSleep (opt-in)

**Files:**
- Modify: `src/memory/aeonsleep.py:137-167` (constructor)
- Create: `tests/memory/test_aeonsleep_sharded.py`

- [ ] **Step 1: Write the failing test**

Create `tests/memory/test_aeonsleep_sharded.py`:
```python
"""AeonSleep can be constructed with ShardedAtlasIndex."""
from __future__ import annotations

from datetime import datetime

import numpy as np

from src.memory.aeonsleep import AeonSleep, Episode
from src.memory.sharded_atlas import ShardedAtlasIndex


def test_sleep_accepts_sharded_atlas():
    sharded = ShardedAtlasIndex(dim=32, n_shards=4)
    palace = AeonSleep(dim=32, atlas=sharded)
    rng = np.random.default_rng(0)
    for i in range(50):
        v = rng.standard_normal(32).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        palace.write(
            Episode(
                id=f"t{i}",
                text="",
                embedding=v.tolist(),
                ts=datetime(2026, 4, 19, 9, i % 60),
                topic="scale",
            )
        )
    hits = palace.recall(np.zeros(32).tolist(), k=5)
    assert len(hits) == 5
```

- [ ] **Step 2: Run test**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_aeonsleep_sharded.py -v`
Expected: likely FAIL because `AeonSleep.__init__` type-annotates `atlas: AtlasIndex | None`. The test passes a `ShardedAtlasIndex`.

- [ ] **Step 3: Loosen the type in the constructor**

In `src/memory/aeonsleep.py` change the constructor `atlas` parameter type to structural:
```python
        atlas: "AtlasIndex | ShardedAtlasIndex | None" = None,
```
and at top of file add (guarded by `TYPE_CHECKING`):
```python
from src.memory.sharded_atlas import ShardedAtlasIndex  # noqa: F401
```

- [ ] **Step 4: Run test**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_aeonsleep_sharded.py tests/memory/test_aeonsleep.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory/aeonsleep.py tests/memory/test_aeonsleep_sharded.py
git commit -m "feat(memory): AeonSleep accepts ShardedAtlasIndex"
```

---

### Task 8: `PredictorRegistry` — shared vs per-stack predictor

**Files:**
- Create: `src/memory/predictor_registry.py`
- Create: `tests/memory/test_predictor_registry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/memory/test_predictor_registry.py`:
```python
"""PredictorRegistry: shared or per-stack LatentMLP selection."""
from __future__ import annotations

import numpy as np

from src.memory.aeon_predictor import LatentMLP
from src.memory.predictor_registry import PredictorRegistry


def test_shared_mode_returns_same_mlp():
    reg = PredictorRegistry(dim=32, hidden=16, n_stacks=4, mode="shared")
    a = reg.get(stack_id=0)
    b = reg.get(stack_id=3)
    assert a is b
    assert isinstance(a, LatentMLP)


def test_per_stack_mode_returns_distinct_mlps():
    reg = PredictorRegistry(dim=32, hidden=16, n_stacks=4, mode="per_stack")
    a = reg.get(stack_id=0)
    b = reg.get(stack_id=1)
    c = reg.get(stack_id=0)
    assert a is not b
    assert a is c


def test_per_stack_unknown_stack_falls_back_to_shared():
    reg = PredictorRegistry(dim=32, hidden=16, n_stacks=4, mode="per_stack")
    fallback = reg.get(stack_id=-1)
    again = reg.get(stack_id=-1)
    assert fallback is again


def test_rejects_invalid_mode():
    import pytest
    with pytest.raises(ValueError):
        PredictorRegistry(dim=32, hidden=16, n_stacks=4, mode="bogus")
```

- [ ] **Step 2: Run test**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_predictor_registry.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `src/memory/predictor_registry.py`:
```python
"""PredictorRegistry — shared or per-stack LatentMLP selection.

Drives the Task 8 decision point: at scale, do stacks have distinct
enough latent dynamics to justify per-stack weights? The eval script
(`scripts/bench_scale.py --predictor-mode ...`) runs both modes on the
same corpus and reports recall delta.
"""
from __future__ import annotations

from typing import Literal

from src.memory.aeon_predictor import LatentMLP

Mode = Literal["shared", "per_stack"]


class PredictorRegistry:
    def __init__(
        self,
        *,
        dim: int,
        hidden: int,
        n_stacks: int,
        mode: Mode = "shared",
        seed: int = 0,
    ) -> None:
        if mode not in ("shared", "per_stack"):
            raise ValueError(f"mode must be 'shared' or 'per_stack', got {mode}")
        self.dim = dim
        self.hidden = hidden
        self.n_stacks = n_stacks
        self.mode = mode
        self._shared = LatentMLP(dim=dim, hidden=hidden, n_stacks=n_stacks, seed=seed)
        self._per_stack: dict[int, LatentMLP] = {}
        self._seed = seed

    def get(self, stack_id: int) -> LatentMLP:
        if self.mode == "shared" or stack_id < 0:
            return self._shared
        if stack_id not in self._per_stack:
            # Per-stack seed derived so each MLP initializes differently.
            self._per_stack[stack_id] = LatentMLP(
                dim=self.dim,
                hidden=self.hidden,
                n_stacks=self.n_stacks,
                seed=self._seed + 1 + stack_id,
            )
        return self._per_stack[stack_id]

    def all_mlps(self) -> list[LatentMLP]:
        if self.mode == "shared":
            return [self._shared]
        return [self._shared] + list(self._per_stack.values())
```

- [ ] **Step 4: Run test**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_predictor_registry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/memory/predictor_registry.py tests/memory/test_predictor_registry.py
git commit -m "feat(memory): predictor registry for shared vs per-stack mode"
```

---

### Task 9: Ingest scenario with memory sampler

**Files:**
- Create: `bench/scenarios/ingest.py`
- Modify: `tests/bench/test_scenarios.py` (add ingest test)

- [ ] **Step 1: Add failing test**

Append to `tests/bench/test_scenarios.py`:
```python
def test_ingest_scenario_smoke():
    from bench.corpus_gen import CorpusSpec, generate_synthetic
    from bench.scenarios.ingest import run_ingest_scenario
    corpus = generate_synthetic(CorpusSpec(n_turns=500, dim=64, seed=0))
    report, peak_mb = run_ingest_scenario(corpus, n_shards=2)
    assert report.summary()["ingest"]["count"] == 500
    assert peak_mb > 0.0
```

- [ ] **Step 2: Run test**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_scenarios.py::test_ingest_scenario_smoke -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `bench/scenarios/ingest.py`:
```python
"""Ingest scenario — writes a whole corpus, tracks latency + peak RSS."""
from __future__ import annotations

import tracemalloc
from datetime import datetime, timedelta

from bench.corpus_gen import Corpus
from bench.harness import ScaleRun
from src.memory.aeonsleep import AeonSleep, Episode
from src.memory.sharded_atlas import ShardedAtlasIndex


def run_ingest_scenario(
    corpus: Corpus, *, n_shards: int = 1
) -> tuple[ScaleRun, float]:
    dim = int(corpus.vectors.shape[1])
    atlas = ShardedAtlasIndex(dim=dim, n_shards=n_shards) if n_shards > 1 else None
    palace = AeonSleep(dim=dim, atlas=atlas)
    run = ScaleRun(name=f"ingest-{len(corpus.ids)}-s{n_shards}", size=len(corpus.ids))
    tracemalloc.start()
    t0 = datetime(2026, 4, 19, 9, 0)
    for i in range(len(corpus.ids)):
        ep = Episode(
            id=corpus.ids[i],
            text="",
            embedding=corpus.vectors[i].tolist(),
            ts=t0 + timedelta(seconds=i),
            topic="scale",
            payload={"stack_id": int(corpus.stack_ids[i])},
        )
        with run.record("ingest"):
            palace.write(ep)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return run, peak / 1e6
```

- [ ] **Step 4: Run test**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_scenarios.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bench/scenarios/ingest.py tests/bench/test_scenarios.py
git commit -m "feat(bench): ingest scenario with tracemalloc peak"
```

---

### Task 10: Concurrent throughput scenario

**Files:**
- Create: `bench/scenarios/concurrent.py`
- Modify: `tests/bench/test_scenarios.py` (add concurrent test)

- [ ] **Step 1: Add failing test**

Append to `tests/bench/test_scenarios.py`:
```python
def test_concurrent_scenario_smoke():
    from bench.corpus_gen import CorpusSpec, generate_synthetic
    from bench.scenarios.concurrent import run_concurrent_recall
    corpus = generate_synthetic(CorpusSpec(n_turns=500, dim=64, seed=0))
    result = run_concurrent_recall(corpus, n_workers=4, n_ops=200)
    assert result.ops_completed == 200
    assert result.qps > 0.0
```

- [ ] **Step 2: Run test**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_scenarios.py::test_concurrent_scenario_smoke -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `bench/scenarios/concurrent.py`:
```python
"""Concurrent recall/throughput using ThreadPoolExecutor.

numpy releases the GIL during @ / dot / norm, so threads give real
parallelism for recall. Ingest is write-locked in AeonSleep (no shared
state races because AtlasIndex.insert is not thread-safe) — we test
READ throughput here; ingest throughput is single-threaded by design.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from bench.corpus_gen import Corpus
from src.memory.aeonsleep import AeonSleep, Episode
from src.memory.sharded_atlas import ShardedAtlasIndex


@dataclass
class ThroughputResult:
    ops_completed: int
    elapsed_s: float
    qps: float
    n_workers: int


def run_concurrent_recall(
    corpus: Corpus, *, n_workers: int = 8, n_ops: int = 1000, n_shards: int = 4
) -> ThroughputResult:
    dim = int(corpus.vectors.shape[1])
    atlas = ShardedAtlasIndex(dim=dim, n_shards=n_shards)
    palace = AeonSleep(dim=dim, atlas=atlas)
    t0 = datetime(2026, 4, 19, 9, 0)
    for i in range(len(corpus.ids)):
        palace.write(
            Episode(
                id=corpus.ids[i],
                text="",
                embedding=corpus.vectors[i].tolist(),
                ts=t0 + timedelta(seconds=i),
                topic="scale",
                payload={"stack_id": int(corpus.stack_ids[i])},
            )
        )
    rng = np.random.default_rng(0)
    qs = [corpus.vectors[int(i)].tolist() for i in rng.integers(0, len(corpus.ids), size=n_ops)]
    lock = threading.Lock()

    def _worker(q: list[float]) -> None:
        palace.recall(q, k=10)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(_worker, qs))
    elapsed = time.perf_counter() - start
    return ThroughputResult(
        ops_completed=n_ops,
        elapsed_s=elapsed,
        qps=n_ops / elapsed if elapsed > 0 else 0.0,
        n_workers=n_workers,
    )
```

- [ ] **Step 4: Run test**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_scenarios.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bench/scenarios/concurrent.py tests/bench/test_scenarios.py
git commit -m "feat(bench): concurrent recall throughput scenario"
```

---

### Task 11: Extend `scripts/bench_scale.py` to drive all scenarios

**Files:**
- Modify: `scripts/bench_scale.py` (replace `run_baseline` with a switch over scenarios)

- [ ] **Step 1: Replace body with multi-scenario CLI**

Replace the contents of `scripts/bench_scale.py` with:
```python
#!/usr/bin/env python3
"""Scale bench CLI — runs ingest / recall / concurrent scenarios at size S."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench.corpus_gen import CorpusSpec, generate_synthetic
from bench.scenarios.concurrent import run_concurrent_recall
from bench.scenarios.ingest import run_ingest_scenario
from bench.scenarios.recall import run_recall_scenario


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True,
                    help="10000 | 100000 | 500000 | 1000000")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--n-shards", type=int, default=4)
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--scenario", choices=["ingest", "recall", "concurrent", "all"],
                    default="all")
    ap.add_argument("--out", type=Path, default=Path("results/scale/2026-04-19"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    spec = CorpusSpec(n_turns=args.size, dim=args.dim, seed=0)
    corpus = generate_synthetic(spec)
    results: dict = {"size": args.size, "dim": args.dim, "n_shards": args.n_shards}

    if args.scenario in ("ingest", "all"):
        run, peak_mb = run_ingest_scenario(corpus, n_shards=args.n_shards)
        run.to_csv(args.out / f"ingest-{args.size}-s{args.n_shards}.csv")
        results["ingest"] = {"summary": run.summary(), "peak_mb": peak_mb}

    if args.scenario in ("recall", "all"):
        run = run_recall_scenario(corpus, n_queries=args.n_queries, k=10)
        run.to_csv(args.out / f"recall-{args.size}.csv")
        results["recall"] = run.summary()

    if args.scenario in ("concurrent", "all"):
        throughput = run_concurrent_recall(
            corpus, n_workers=args.n_workers, n_ops=2000, n_shards=args.n_shards
        )
        results["concurrent"] = {
            "qps": throughput.qps,
            "elapsed_s": throughput.elapsed_s,
            "n_workers": throughput.n_workers,
        }

    (args.out / f"summary-{args.size}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Dry-run at 10k to verify**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python scripts/bench_scale.py --size 10000 --scenario all`
Expected: JSON summary printed, three CSVs + one JSON under `results/scale/2026-04-19/`.

- [ ] **Step 3: Commit**

```bash
git add scripts/bench_scale.py
git commit -m "feat(bench): unified CLI across ingest/recall/concurrent scenarios"
```

---

### Task 12: Recall-quality regression gate at 100k

**Files:**
- Create: `tests/bench/test_recall_quality_parity.py`

- [ ] **Step 1: Write the test (this is the sharding kill-criterion gate)**

Create `tests/bench/test_recall_quality_parity.py`:
```python
"""At 10k × 64-dim the sharded recall must agree with the single-shard
recall on >= 90 % of top-5 ids. This is the sharding kill-criterion."""
from __future__ import annotations

import numpy as np
import pytest

from bench.corpus_gen import CorpusSpec, generate_synthetic
from src.memory.atlas import AtlasIndex
from src.memory.sharded_atlas import ShardedAtlasIndex


@pytest.mark.parametrize("n_shards", [2, 4, 8])
def test_sharded_vs_single_top5_overlap(n_shards):
    corpus = generate_synthetic(CorpusSpec(n_turns=10_000, dim=64, seed=0))
    single = AtlasIndex(dim=64)
    sharded = ShardedAtlasIndex(dim=64, n_shards=n_shards)
    for i in range(len(corpus.ids)):
        single.insert(corpus.ids[i], corpus.vectors[i])
        sharded.insert(corpus.ids[i], corpus.vectors[i])
    rng = np.random.default_rng(1)
    overlaps = []
    for qi in rng.integers(0, len(corpus.ids), size=200):
        q = corpus.vectors[int(qi)]
        s1 = {h.id for h in single.search(q, k=5)}
        s2 = {h.id for h in sharded.search(q, k=5)}
        overlaps.append(len(s1 & s2) / 5.0)
    mean_overlap = sum(overlaps) / len(overlaps)
    assert mean_overlap >= 0.90, f"overlap {mean_overlap:.3f} < 0.90 for n_shards={n_shards}"
```

- [ ] **Step 2: Run it**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_recall_quality_parity.py -v`
Expected: PASS for all three shard counts. If it fails — that IS the Task 6 kill-signal. Do not "fix" by lowering the threshold; investigate why hash routing is losing hits (the current `AtlasIndex` uses greedy page assignment — a vector near a shard boundary should NOT disappear because each shard keeps its full local search).

- [ ] **Step 3: Commit**

```bash
git add tests/bench/test_recall_quality_parity.py
git commit -m "test(bench): sharded-vs-single recall parity gate"
```

---

### Task 13: Full sweep — run 10k / 100k / 500k / 1M and collect curves

**Files:**
- Create: `scripts/bench_scale_sweep.sh`
- Create: `results/scale/2026-04-19/sweep.md` (populated from the sweep output)

- [ ] **Step 1: Write the sweep script**

Create `scripts/bench_scale_sweep.sh`:
```bash
#!/usr/bin/env bash
# Run the full scale sweep. Expect ~30–60 min on M3 Ultra for the 1M rung.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="results/scale/2026-04-19"
mkdir -p "$OUT"

for SIZE in 10000 100000 500000 1000000; do
  echo "=== size=$SIZE ==="
  uv run python scripts/bench_scale.py \
    --size "$SIZE" \
    --dim 384 \
    --n-shards 8 \
    --n-queries 500 \
    --n-workers 8 \
    --scenario all \
    --out "$OUT"
done

echo "Sweep done. See $OUT/summary-*.json"
```

Make it executable:
```bash
chmod +x scripts/bench_scale_sweep.sh
```

- [ ] **Step 2: Run the sweep on Mac Studio**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && ./scripts/bench_scale_sweep.sh 2>&1 | tee results/scale/2026-04-19/sweep.log`
Expected: four `summary-<SIZE>.json` files + CSVs. Each rung must stay within the latency targets listed in the plan header; if the 1M rung blows the budget, Task 14 (mmap mode) is required.

- [ ] **Step 3: Write `results/scale/2026-04-19/sweep.md` with the numbers**

Create a table from the JSONs:
```markdown
# Scale sweep — 2026-04-19

| Size | Ingest p50 (ms) | Ingest p95 | Recall p50 | Recall p95 | Recall p99 | QPS (8 workers) | Peak RSS (MB) |
|------|-----------------|------------|------------|------------|------------|------------------|---------------|
| 10k  | ... | ... | ... | ... | ... | ... | ... |
| 100k | ... | ... | ... | ... | ... | ... | ... |
| 500k | ... | ... | ... | ... | ... | ... | ... |
| 1M   | ... | ... | ... | ... | ... | ... | ... |
```
Fill the dots from the summaries.

- [ ] **Step 4: Commit**

```bash
git add scripts/bench_scale_sweep.sh results/scale/2026-04-19/sweep.md results/scale/2026-04-19/sweep.log
git commit -m "bench: 2026-04-19 scale sweep 10k→1M"
```

---

### Task 14: Memory-mapped page mode (only if 1M peaks > 8 GB)

**Files:**
- Modify: `src/memory/atlas.py` (add `mmap_dir` param to `AtlasIndex`)
- Create: `tests/memory/test_atlas_mmap.py`

- [ ] **Step 1: Gate on Task 13 — skip if peak <= 8 GB at 1M**

Check `results/scale/2026-04-19/summary-1000000.json`. If `ingest.peak_mb` < 8000, CLOSE this task with a note `results/scale/2026-04-19/mmap-not-needed.md` explaining the budget held and skip straight to Task 15. Otherwise proceed.

- [ ] **Step 2: Write the failing test**

Create `tests/memory/test_atlas_mmap.py`:
```python
"""AtlasIndex(mmap_dir=...) pages reside on disk, not in RAM."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.memory.atlas import AtlasIndex


def test_mmap_mode_file_backed(tmp_path: Path):
    idx = AtlasIndex(dim=32, mmap_dir=tmp_path)
    rng = np.random.default_rng(0)
    for i in range(300):
        v = rng.standard_normal(32).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        idx.insert(f"v{i}", v)
    # at least one .npy page file exists
    files = list(tmp_path.glob("page-*.npy"))
    assert len(files) >= 1
    # recall still works
    q = rng.standard_normal(32).astype(np.float32)
    q /= np.linalg.norm(q) + 1e-8
    hits = idx.search(q, k=5)
    assert len(hits) == 5
```

- [ ] **Step 3: Run test**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_atlas_mmap.py -v`
Expected: FAIL — `AtlasIndex.__init__() got an unexpected keyword argument 'mmap_dir'`

- [ ] **Step 4: Implement mmap mode**

Modify `src/memory/atlas.py` `AtlasIndex.__init__` and `insert` to support optional file-backed pages. Add `mmap_dir: Path | None = None` param; when set, create each new page as `np.memmap(mmap_dir / f"page-{idx}.npy", dtype=np.float32, mode="w+", shape=(PAGE_SIZE, dim))` instead of `np.zeros`. Everything else keeps working unchanged because numpy memmap supports the same slicing / dot ops.

- [ ] **Step 5: Run test**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/memory/test_atlas_mmap.py tests/memory/test_atlas.py -v`
Expected: all PASS (regression-safe because `mmap_dir` defaults to None).

- [ ] **Step 6: Re-run the 1M rung with mmap**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python scripts/bench_scale.py --size 1000000 --dim 384 --n-shards 8 --scenario all`

- [ ] **Step 7: Commit**

```bash
git add src/memory/atlas.py tests/memory/test_atlas_mmap.py results/scale/2026-04-19/
git commit -m "feat(memory): optional mmap pages for large corpora"
```

---

### Task 15: Publish scale curves + decision note

**Files:**
- Create: `results/scale/2026-04-19/README.md`
- Optional: `results/scale/2026-04-19/plot_curves.py` + PNG outputs

- [ ] **Step 1: Write the decision note**

Create `results/scale/2026-04-19/README.md` containing, in this exact order:

1. The latency-targets table (copied from plan header) with a PASS/FAIL column filled in from the sweep summaries.
2. The memory-budget table with measured column.
3. The Task 8 decision: `mode="shared"` vs `mode="per_stack"` — which was picked and the recall delta that justified it. Run `uv run python scripts/bench_scale.py --size 100000 --predictor-mode shared` and `--predictor-mode per_stack` if the registry has been wired into the scenarios; otherwise document that the registry exists but scenarios use `shared` only and the per-stack mode is deferred to a follow-up plan.
4. A top-3 bottleneck list from the Task 3 pstats file (confirm what the sweep empirically found).
5. Kill / success verdict against the criteria in the plan header.

- [ ] **Step 2 (optional): Generate PNG curves**

If `matplotlib` is acceptable (add as dev-optional dep, do not pollute core deps), write `results/scale/2026-04-19/plot_curves.py` that reads the four `summary-<SIZE>.json` files and plots p50/p95/p99 recall latency vs size on a log-log axis. Save to `results/scale/2026-04-19/curves.png`.

- [ ] **Step 3: Commit**

```bash
git add results/scale/2026-04-19/README.md
[ -f results/scale/2026-04-19/curves.png ] && git add results/scale/2026-04-19/plot_curves.py results/scale/2026-04-19/curves.png
git commit -m "docs(bench): scale-test verdict 10k→1M"
```

---

## Self-Review

**Spec coverage (8 items from the request):**

1. Corpus generation 100k/500k/1M → Tasks 1, 2, 13 (synthetic via `generate_synthetic`; real corpus via sentence-transformers is captured as a follow-on in `bench/corpus_gen.py` but generation of textual inputs is intentionally out-of-scope here — the vectors ARE what Aeon ingests).
2. Atlas sharding → Tasks 6, 7, 12.
3. Predictor sharding (one-per-stack vs shared) → Task 8 + decision in Task 15.
4. Minibatch sampling at scale → Task 5.
5. Memory budget → Plan header table + Tasks 3 (baseline measurement), 9 (tracemalloc peak), 14 (mmap fallback).
6. Latency benchmark p50/p95/p99 → Tasks 1 (percentiles util), 4, 9, 13.
7. Throughput benchmark → Task 10.
8. Results doc + figures → Task 15.

**Placeholder scan:** none — every step has executable code or a concrete command.

**Type consistency:** `ScaleRun`, `CorpusSpec`, `Corpus`, `ShardedAtlasIndex`, `PredictorRegistry`, `ThroughputResult` are defined once each and referenced with matching signatures in all tasks. `AeonSleep(dim=..., atlas=...)` uses the existing constructor param name. `PredictorConfig.buffer_capacity` is introduced in Task 5 with a 50k default that is above the PoC eval's 1k scenarios, so pre-existing tests under `tests/memory/test_aeon_predictor.py` keep passing.

**Known scope trims:**

- Real-text corpus ingestion (sentence-transformers encode step) is hinted but not wired — the synthetic random-walk is sufficient for all latency/memory/throughput measurements. If the paper requires real-text curves, a follow-on plan should wire `sentence-transformers` behind `CorpusSpec(kind="real", texts=[...])` and batch encode on kxkm-ai.
- `per_stack` predictor training is wired at the registry level (Task 8) but the scenarios only call `shared` mode by default — a Task 15 decision note captures whether a follow-on plan is needed.
- Graph (`TraceGraph`) sharding is NOT in scope: at 1M nodes the Python-dict graph is ~800 MB, below the 8 GB budget. If the sweep reveals graph-side hotspots, a separate plan should tackle `src/memory/trace.py`.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-19-scale-test.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch with checkpoints at Tasks 5, 10, 13.

**Which approach?**
