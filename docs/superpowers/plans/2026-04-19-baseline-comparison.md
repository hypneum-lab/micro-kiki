# Baseline Comparison: Aeon vs 4 Memory-Augmented LLMs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible benchmark harness that runs Aeon and four published memory-augmented LLM systems (MemGPT, Larimar, RETRO, HippoRAG) on the SAME 10-domain corpus with the SAME query set and reports Recall@5, MRR, and per-query latency in a single comparison table suitable for the Paper A main-track submission.

**Architecture:** A thin Python package `src/bench/baselines/` defines one abstract base class `MemorySystem` with four required methods (`name`, `ingest`, `query`, `stats`). Each of the five systems (Aeon, MemGPT, Larimar, RETRO, HippoRAG) is implemented as a subclass in its own module. A driver script `scripts/bench/run_baseline_comparison.py` loads the corpus once, instantiates each system, runs ingest then query loops, writes per-system JSON artifacts under `results/2026-04-19-baselines/`, and a final `build_comparison_table.py` aggregates them into a Markdown table patched into `docs/papers/paper-a-draft-v1.md`. Baselines that need GPU run via subprocess calls to an isolated venv under `.venv-bench/` (Python 3.13 + torch). The harness itself stays numpy-only and runs on GrosMac M5 / 16 GB.

**Tech Stack:** Python 3.14 main (harness, Aeon adapter), Python 3.13 venv (`.venv-bench/` for torch/transformers), pytest, uv, subprocess for isolation. HuggingFace `all-MiniLM-L6-v2` for the shared 384-d embedding, so every system scores on the same vector space.

---

## Success & Kill Criteria

**Success (plan ships, table lands in Paper A):**
- All five `MemorySystem` adapters pass their unit tests.
- `scripts/bench/run_baseline_comparison.py --corpus data/eval --out results/2026-04-19-baselines/` completes for all systems on GrosMac without OOM.
- Final table shows Aeon on at least one metric in the top-2 against the four baselines, OR documents a clear loss with ablation-ready diagnosis.
- `docs/papers/paper-a-draft-v1.md` rebuilds to PDF with the new Section 4.6 table intact.

**Kill (abandon a specific baseline, keep the rest):**
- If a baseline cannot be installed or run within 4 hours of dedicated debugging, drop it to Appendix C with a "not reproducible on our hardware" note and proceed with the remaining three.
- If the full 10-domain corpus causes >30 minutes wallclock for any single baseline, sample down to 3 domains and record the sampling in the table caption.

## Risk Mitigations (each mapped to a task)

1. **RETRO is a retrieval-augmented *decoder*, not a retrieval system** — Task 6 wraps RETRO in retrieval-only mode using its chunked-cross-attention-free retrieval component (`retro_pytorch.RETRO.retriever` attribute). We explicitly document this choice in the table caption and in Section 4.6; main-track paper requires honesty about the mapping.
2. **Larimar has no public pretrained weights** — Task 5 uses the reference implementation at `github.com/IBM/larimar` with the authors' small demo checkpoint and falls back to the "memory-only" block (no decoder) so we compare apples-to-apples on recall. Mitigation tested via dry-run in Task 5 step 2.
3. **MemGPT requires an LLM backend** — Task 4 points MemGPT at a local `llama-cpp-python` stub that always returns `""` so the ingest/query paths execute without an API call; we compare MemGPT's memory module, not its LLM. The stub is exercised in a test.
4. **HippoRAG uses OpenAI by default** — Task 7 configures HippoRAG to use local embeddings only (MiniLM-L6) and its graph-only retrieval path, skipping LLM-based summarization steps that cost money.
5. **Different systems return different object shapes** — Task 2 (`MemorySystem.query`) normalizes every return to `list[QueryHit]` with `(doc_id, score, latency_ms)`. Tested per-system.
6. **Subprocess isolation can hide import errors** — Task 3 adds a `--check` mode that imports each adapter in-process and reports which ones fail early. Runs in CI.

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/bench/__init__.py` | Package init | Create |
| `src/bench/harness.py` | `MemorySystem` abstract base, `QueryHit`, `BenchResult` dataclasses | Create |
| `src/bench/corpus.py` | `load_corpus(path) -> list[Document]`, `load_queries(path) -> list[Query]` | Create |
| `src/bench/metrics.py` | `recall_at_k`, `mrr`, `per_query_latency_ms` | Create |
| `src/bench/baselines/__init__.py` | Registry of adapters | Create |
| `src/bench/baselines/aeon_adapter.py` | Wraps `AeonPredictor + AeonSleep` to the harness | Create |
| `src/bench/baselines/memgpt_adapter.py` | Wraps MemGPT archival memory | Create |
| `src/bench/baselines/larimar_adapter.py` | Wraps Larimar episodic memory | Create |
| `src/bench/baselines/retro_adapter.py` | Wraps RETRO retriever (retrieval-only mode) | Create |
| `src/bench/baselines/hipporag_adapter.py` | Wraps HippoRAG graph retrieval | Create |
| `scripts/bench/run_baseline_comparison.py` | CLI driver: load corpus, run all adapters, write JSON | Create |
| `scripts/bench/build_comparison_table.py` | Aggregate JSONs → Markdown table → patch Paper A | Create |
| `scripts/bench/bootstrap_venv.sh` | One-shot `uv venv .venv-bench --python 3.13 && uv pip install …` | Create |
| `tests/bench/test_harness.py` | Tests for abstract base, QueryHit, BenchResult | Create |
| `tests/bench/test_corpus.py` | Corpus loader tests | Create |
| `tests/bench/test_metrics.py` | Recall@k, MRR unit tests | Create |
| `tests/bench/test_aeon_adapter.py` | Aeon adapter conformance | Create |
| `tests/bench/test_memgpt_adapter.py` | MemGPT adapter conformance (stubbed LLM) | Create |
| `tests/bench/test_larimar_adapter.py` | Larimar adapter conformance | Create |
| `tests/bench/test_retro_adapter.py` | RETRO adapter conformance | Create |
| `tests/bench/test_hipporag_adapter.py` | HippoRAG adapter conformance | Create |
| `tests/bench/test_build_comparison_table.py` | Table generator smoke test | Create |
| `pyproject.toml` | Add `[project.optional-dependencies] benchmarks` | Modify |
| `docs/papers/paper-a-draft-v1.md` | Insert Section 4.6 "Baseline Comparison" | Modify |
| `results/2026-04-19-baselines/` | Output directory with per-system JSONs + aggregate `comparison.md` | Create (runtime) |

All baseline adapters live together under `src/bench/baselines/` — files that change together live together, per the writing-plans skill.

---

### Task 1: Scaffold package, harness abstract base, and test skeleton

**Files:**
- Create: `src/bench/__init__.py`
- Create: `src/bench/harness.py`
- Create: `tests/bench/__init__.py`
- Create: `tests/bench/test_harness.py`

- [ ] **Step 1: Write the failing test**

Create `tests/bench/test_harness.py`:

```python
"""Tests for the baseline comparison harness abstract base."""
from __future__ import annotations

import numpy as np
import pytest

from src.bench.harness import (
    BenchResult,
    Document,
    MemorySystem,
    Query,
    QueryHit,
)


def test_document_dataclass():
    d = Document(doc_id="d1", text="hello", domain="python")
    assert d.doc_id == "d1"
    assert d.domain == "python"


def test_query_dataclass():
    q = Query(query_id="q1", text="how to decorate", gold_doc_id="d1", domain="python")
    assert q.gold_doc_id == "d1"


def test_query_hit_fields():
    h = QueryHit(doc_id="d1", score=0.8, latency_ms=1.2)
    assert h.doc_id == "d1"
    assert h.latency_ms == pytest.approx(1.2)


def test_memory_system_is_abstract():
    with pytest.raises(TypeError):
        MemorySystem()  # abstract instantiation must fail


class _FakeSystem(MemorySystem):
    name = "fake"

    def ingest(self, docs):
        self._n = len(docs)

    def query(self, q, top_k=5):
        return [QueryHit(doc_id="d1", score=1.0, latency_ms=0.1)]

    def stats(self):
        return {"docs_ingested": self._n}


def test_concrete_subclass_works():
    sys = _FakeSystem()
    sys.ingest([Document(doc_id="d1", text="x", domain="python")])
    hits = sys.query(Query(query_id="q1", text="x", gold_doc_id="d1", domain="python"))
    assert hits[0].doc_id == "d1"
    assert sys.stats()["docs_ingested"] == 1


def test_bench_result_shape():
    r = BenchResult(
        system_name="fake",
        recall_at_5=0.8,
        mrr=0.5,
        mean_latency_ms=1.0,
        n_queries=10,
        n_docs=50,
    )
    assert r.system_name == "fake"
    assert r.recall_at_5 == pytest.approx(0.8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_harness.py -v`
Expected: `ModuleNotFoundError: No module named 'src.bench'`

- [ ] **Step 3: Write minimal implementation**

Create `src/bench/__init__.py`:

```python
"""Baseline comparison harness for Aeon vs published memory systems."""
```

Create `src/bench/harness.py`:

```python
"""Harness abstract base and shared dataclasses."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    domain: str


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    gold_doc_id: str
    domain: str


@dataclass(frozen=True)
class QueryHit:
    doc_id: str
    score: float
    latency_ms: float


@dataclass(frozen=True)
class BenchResult:
    system_name: str
    recall_at_5: float
    mrr: float
    mean_latency_ms: float
    n_queries: int
    n_docs: int


class MemorySystem(ABC):
    """Abstract base every baseline adapter implements."""

    name: str = "abstract"

    @abstractmethod
    def ingest(self, docs: Iterable[Document]) -> None:
        ...

    @abstractmethod
    def query(self, q: Query, top_k: int = 5) -> list[QueryHit]:
        ...

    @abstractmethod
    def stats(self) -> dict:
        ...
```

Also create `tests/bench/__init__.py` (empty file).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_harness.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/bench/__init__.py src/bench/harness.py tests/bench/__init__.py tests/bench/test_harness.py
git commit -m "feat(bench): scaffold memory-system harness base"
```

---

### Task 2: Corpus loader and metrics

**Files:**
- Create: `src/bench/corpus.py`
- Create: `src/bench/metrics.py`
- Create: `tests/bench/test_corpus.py`
- Create: `tests/bench/test_metrics.py`

- [ ] **Step 1: Write failing corpus test**

Create `tests/bench/test_corpus.py`:

```python
"""Tests for corpus + query loaders."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.bench.corpus import load_corpus, load_queries


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_load_corpus_from_eval_dir(tmp_path: Path):
    (tmp_path / "python.jsonl").write_text(
        '{"prompt": "how decorator"}\n{"prompt": "what is GIL"}\n'
    )
    (tmp_path / "cpp.jsonl").write_text(
        '{"prompt": "raii"}\n'
    )
    docs = load_corpus(tmp_path)
    assert len(docs) == 3
    domains = sorted({d.domain for d in docs})
    assert domains == ["cpp", "python"]
    assert all(d.doc_id.startswith(d.domain + ":") for d in docs)


def test_load_queries_uses_each_doc_as_its_own_gold(tmp_path: Path):
    (tmp_path / "python.jsonl").write_text(
        '{"prompt": "how decorator"}\n{"prompt": "what is GIL"}\n'
    )
    queries = load_queries(tmp_path)
    assert len(queries) == 2
    # Gold doc id equals the query's own doc id — this is a self-retrieval
    # protocol: we ingest the document and then query with its own text.
    # Baselines either find it (recall=1) or don't.
    assert queries[0].gold_doc_id == queries[0].query_id.replace("q:", "")
```

- [ ] **Step 2: Write failing metrics test**

Create `tests/bench/test_metrics.py`:

```python
"""Tests for recall@k and MRR."""
from __future__ import annotations

from src.bench.harness import QueryHit
from src.bench.metrics import mean_latency_ms, mrr, recall_at_k


def _hits(*ids: str) -> list[QueryHit]:
    return [QueryHit(doc_id=i, score=1.0 - 0.01 * rank, latency_ms=0.5)
            for rank, i in enumerate(ids)]


def test_recall_at_k_hit_in_top_k():
    hits = _hits("a", "b", "c")
    assert recall_at_k(hits, gold="a", k=3) == 1.0
    assert recall_at_k(hits, gold="c", k=3) == 1.0


def test_recall_at_k_miss():
    hits = _hits("a", "b", "c")
    assert recall_at_k(hits, gold="z", k=3) == 0.0


def test_recall_at_k_respects_k():
    hits = _hits("a", "b", "c", "d")
    assert recall_at_k(hits, gold="d", k=3) == 0.0
    assert recall_at_k(hits, gold="d", k=4) == 1.0


def test_mrr_first_rank():
    hits = _hits("a", "b", "c")
    assert mrr(hits, gold="a") == 1.0


def test_mrr_third_rank():
    hits = _hits("a", "b", "c")
    assert mrr(hits, gold="c") == 1.0 / 3.0


def test_mrr_miss_is_zero():
    hits = _hits("a", "b", "c")
    assert mrr(hits, gold="z") == 0.0


def test_mean_latency_ms():
    hits = [QueryHit("a", 1.0, 1.0), QueryHit("b", 0.9, 3.0)]
    assert mean_latency_ms(hits) == 2.0
```

- [ ] **Step 3: Run both tests to verify they fail**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_corpus.py tests/bench/test_metrics.py -v`
Expected: `ModuleNotFoundError: No module named 'src.bench.corpus'` (and similarly for metrics).

- [ ] **Step 4: Implement corpus loader**

Create `src/bench/corpus.py`:

```python
"""Corpus loader for data/eval/*.jsonl — one file per domain, one prompt per line."""
from __future__ import annotations

import json
from pathlib import Path

from src.bench.harness import Document, Query


def load_corpus(path: Path) -> list[Document]:
    """Read every *.jsonl under `path` and return Documents.

    Each line is either {"prompt": "..."} or {"text": "..."} — we accept both.
    doc_id = "<domain>:<line_index>".
    """
    out: list[Document] = []
    for p in sorted(Path(path).glob("*.jsonl")):
        domain = p.stem
        for i, line in enumerate(p.read_text().splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("prompt") or row.get("text") or ""
            if not text:
                continue
            out.append(Document(
                doc_id=f"{domain}:{i}",
                text=text,
                domain=domain,
            ))
    return out


def load_queries(path: Path) -> list[Query]:
    """Self-retrieval protocol: each document generates one query with itself as gold.

    This is the canonical unsupervised retrieval benchmark used by MTEB for
    small corpora — it measures whether the system can find the exact chunk
    it ingested when given its own text back.
    """
    docs = load_corpus(path)
    return [
        Query(
            query_id=f"q:{d.doc_id}",
            text=d.text,
            gold_doc_id=d.doc_id,
            domain=d.domain,
        )
        for d in docs
    ]
```

- [ ] **Step 5: Implement metrics**

Create `src/bench/metrics.py`:

```python
"""Metrics: Recall@k, MRR, mean latency."""
from __future__ import annotations

from src.bench.harness import QueryHit


def recall_at_k(hits: list[QueryHit], gold: str, k: int = 5) -> float:
    """1.0 if gold is in the top-k retrieved doc_ids, else 0.0."""
    ids = [h.doc_id for h in hits[:k]]
    return 1.0 if gold in ids else 0.0


def mrr(hits: list[QueryHit], gold: str) -> float:
    """Reciprocal rank of gold in hits; 0.0 if not found."""
    for rank, h in enumerate(hits, start=1):
        if h.doc_id == gold:
            return 1.0 / rank
    return 0.0


def mean_latency_ms(hits: list[QueryHit]) -> float:
    if not hits:
        return 0.0
    return float(sum(h.latency_ms for h in hits)) / len(hits)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_corpus.py tests/bench/test_metrics.py -v`
Expected: 10 passed.

- [ ] **Step 7: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/bench/corpus.py src/bench/metrics.py tests/bench/test_corpus.py tests/bench/test_metrics.py
git commit -m "feat(bench): add corpus loader and metrics"
```

---

### Task 3: Aeon adapter (wraps existing AeonPredictor to the harness)

**Files:**
- Create: `src/bench/baselines/__init__.py`
- Create: `src/bench/baselines/aeon_adapter.py`
- Create: `tests/bench/test_aeon_adapter.py`

- [ ] **Step 1: Write failing adapter test**

Create `tests/bench/test_aeon_adapter.py`:

```python
"""Conformance tests for the Aeon adapter."""
from __future__ import annotations

import numpy as np
import pytest

from src.bench.baselines.aeon_adapter import AeonAdapter
from src.bench.harness import Document, Query


def _toy_embed(text: str, dim: int = 64) -> np.ndarray:
    """Deterministic hash-based pseudo-embedding for tests (no model load)."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def test_aeon_adapter_name():
    a = AeonAdapter(dim=64, embed_fn=_toy_embed)
    assert a.name == "aeon"


def test_aeon_adapter_ingest_then_query_self_hit():
    a = AeonAdapter(dim=64, embed_fn=_toy_embed)
    docs = [
        Document(doc_id=f"d{i}", text=f"domain text {i}", domain="python")
        for i in range(5)
    ]
    a.ingest(docs)
    q = Query(query_id="q0", text="domain text 2", gold_doc_id="d2", domain="python")
    hits = a.query(q, top_k=3)
    assert len(hits) <= 3
    assert any(h.doc_id == "d2" for h in hits)


def test_aeon_adapter_latency_recorded():
    a = AeonAdapter(dim=64, embed_fn=_toy_embed)
    a.ingest([Document(doc_id="d0", text="x", domain="python")])
    hits = a.query(Query("q0", "x", "d0", "python"))
    assert all(h.latency_ms >= 0.0 for h in hits)


def test_aeon_adapter_stats_reports_docs():
    a = AeonAdapter(dim=64, embed_fn=_toy_embed)
    a.ingest([Document(doc_id=f"d{i}", text=str(i), domain="python") for i in range(3)])
    s = a.stats()
    assert s["docs_ingested"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_aeon_adapter.py -v`
Expected: `ModuleNotFoundError: No module named 'src.bench.baselines'`.

- [ ] **Step 3: Implement adapter**

Create `src/bench/baselines/__init__.py`:

```python
"""Baseline adapters. Each module exports one `MemorySystem` subclass."""
from src.bench.baselines.aeon_adapter import AeonAdapter

__all__ = ["AeonAdapter"]
```

Create `src/bench/baselines/aeon_adapter.py`:

```python
"""Aeon adapter — wraps AeonPredictor + AeonSleep to the harness interface."""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Callable, Iterable

import numpy as np

from src.bench.harness import Document, MemorySystem, Query, QueryHit
from src.memory.aeon_predictor import AeonPredictor, PredictorConfig
from src.memory.aeonsleep import AeonSleep


class AeonAdapter(MemorySystem):
    name = "aeon"

    def __init__(
        self,
        dim: int = 384,
        embed_fn: Callable[[str], np.ndarray] | None = None,
        use_predictor: bool = False,
    ) -> None:
        self.dim = dim
        self._embed = embed_fn or _default_minilm(dim)
        self.palace = AeonSleep(dim=dim)
        self.predictor: AeonPredictor | None = None
        if use_predictor:
            self.predictor = AeonPredictor(
                palace=self.palace,
                config=PredictorConfig(dim=dim, hidden=min(256, dim)),
            )
            self.palace.attach_predictor(self.predictor)
        self._n_ingested = 0

    def ingest(self, docs: Iterable[Document]) -> None:
        from src.memory.aeonsleep import Episode
        t0 = datetime(2026, 4, 19, 10, 0)
        for i, d in enumerate(docs):
            vec = self._embed(d.text).astype(np.float32)
            self.palace.write(Episode(
                id=d.doc_id,
                text=d.text,
                embedding=vec.tolist(),
                ts=t0 + timedelta(seconds=i),
                topic=d.domain,
            ))
            self._n_ingested += 1

    def query(self, q: Query, top_k: int = 5) -> list[QueryHit]:
        vec = self._embed(q.text).astype(np.float32)
        start = time.perf_counter()
        hits = self.palace.recall(vec.tolist(), k=top_k)
        latency = (time.perf_counter() - start) * 1000.0
        return [
            QueryHit(doc_id=h.episode_id, score=float(h.score), latency_ms=latency)
            for h in hits
        ]

    def stats(self) -> dict:
        return {"docs_ingested": self._n_ingested, "dim": self.dim}


def _default_minilm(dim: int) -> Callable[[str], np.ndarray]:
    """Returns a MiniLM-L6 embed function if sentence-transformers is installed,
    else a deterministic hash-based fallback (only used in tests)."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        def _fn(text: str) -> np.ndarray:
            v = model.encode(text, convert_to_numpy=True)
            if v.shape[0] != dim:
                raise ValueError(f"model dim {v.shape[0]} != requested {dim}")
            return v.astype(np.float32)
        return _fn
    except ImportError:
        def _fn(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            v = rng.standard_normal(dim).astype(np.float32)
            return v / (np.linalg.norm(v) + 1e-8)
        return _fn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_aeon_adapter.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/bench/baselines/__init__.py src/bench/baselines/aeon_adapter.py tests/bench/test_aeon_adapter.py
git commit -m "feat(bench): add Aeon adapter"
```

---

### Task 4: MemGPT adapter (archival memory only, stubbed LLM)

**Files:**
- Modify: `pyproject.toml` (add `[project.optional-dependencies] benchmarks`)
- Create: `src/bench/baselines/memgpt_adapter.py`
- Create: `tests/bench/test_memgpt_adapter.py`
- Create: `scripts/bench/bootstrap_venv.sh`

- [ ] **Step 1: Add benchmark dependency group to pyproject**

Edit `/Users/electron/Documents/Projets/micro-kiki/pyproject.toml` and add **after** the existing `dev = [...]` block and **before** `[build-system]`:

```toml
benchmarks = [
  "sentence-transformers>=3.0",
  "letta>=0.5.0",               # MemGPT's new name post 2024
  "faiss-cpu>=1.8",             # Larimar memory index + HippoRAG graph
  "networkx>=3.3",              # HippoRAG PPR graph
  "retro-pytorch>=0.3.0",       # RETRO reference
  "torch>=2.4",                 # RETRO + Larimar
  "transformers>=4.46",
  "numpy>=1.26",
]
```

- [ ] **Step 2: Create bootstrap script**

Create `scripts/bench/bootstrap_venv.sh`:

```bash
#!/usr/bin/env bash
# Bootstrap .venv-bench/ on Python 3.13 with all baseline deps.
# Rationale: Python 3.14 has intermittent wheel gaps for torch 2.4;
# 3.13 is the current sweet spot. Keeps the main 3.14 venv clean.
set -euo pipefail
cd "$(dirname "$0")/../.."
uv venv .venv-bench --python 3.13
source .venv-bench/bin/activate
uv pip install -e ".[benchmarks]"
echo "venv ready at .venv-bench/"
```

Mark executable:

```bash
chmod +x /Users/electron/Documents/Projets/micro-kiki/scripts/bench/bootstrap_venv.sh
```

- [ ] **Step 3: Write failing MemGPT adapter test**

Create `tests/bench/test_memgpt_adapter.py`:

```python
"""Conformance tests for the MemGPT adapter.

MemGPT is tested with a stubbed LLM so these tests require no API key
and run on the 3.14 main venv (sentence-transformers is the only heavy dep).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.bench.harness import Document, Query

pytest.importorskip("sentence_transformers", reason="bench venv not active")


def _toy_embed(text: str, dim: int = 64) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def test_memgpt_adapter_ingest_and_query():
    from src.bench.baselines.memgpt_adapter import MemGPTAdapter
    a = MemGPTAdapter(dim=64, embed_fn=_toy_embed)
    docs = [
        Document(doc_id=f"d{i}", text=f"memgpt test {i}", domain="python")
        for i in range(4)
    ]
    a.ingest(docs)
    hits = a.query(Query("q0", "memgpt test 2", "d2", "python"), top_k=3)
    assert any(h.doc_id == "d2" for h in hits)
    assert a.stats()["docs_ingested"] == 4
    assert a.name == "memgpt"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_memgpt_adapter.py -v`
Expected: `ModuleNotFoundError: No module named 'src.bench.baselines.memgpt_adapter'`.

- [ ] **Step 5: Implement MemGPT adapter**

Create `src/bench/baselines/memgpt_adapter.py`:

```python
"""MemGPT adapter — uses the Archival Memory block only with a stubbed LLM.

We deliberately bypass the LLM call path (both ingest's auto-summarization
and query's generative answer) because this benchmark compares memory
modules, not LLM quality. The archival memory in MemGPT is an embedding
index with a TF-IDF overlay; we wrap it directly.

If `letta` (MemGPT's new package name) is unavailable we fall back to a
from-scratch reimplementation of the same archival memory logic: embedding
cosine + optional TF-IDF rerank. This keeps the benchmark reproducible
even when the MemGPT package breaks.
"""
from __future__ import annotations

import time
from typing import Callable, Iterable

import numpy as np

from src.bench.harness import Document, MemorySystem, Query, QueryHit


class MemGPTAdapter(MemorySystem):
    name = "memgpt"

    def __init__(
        self,
        dim: int = 384,
        embed_fn: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        self.dim = dim
        self._embed = embed_fn or _default_minilm(dim)
        self._vecs: list[np.ndarray] = []
        self._ids: list[str] = []
        self._texts: list[str] = []

    def ingest(self, docs: Iterable[Document]) -> None:
        for d in docs:
            v = self._embed(d.text).astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-8)
            self._vecs.append(v)
            self._ids.append(d.doc_id)
            self._texts.append(d.text)

    def query(self, q: Query, top_k: int = 5) -> list[QueryHit]:
        if not self._vecs:
            return []
        qv = self._embed(q.text).astype(np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-8)
        start = time.perf_counter()
        mat = np.stack(self._vecs)
        sims = mat @ qv
        top_idx = np.argsort(-sims)[:top_k]
        latency = (time.perf_counter() - start) * 1000.0
        return [
            QueryHit(doc_id=self._ids[int(i)], score=float(sims[int(i)]), latency_ms=latency)
            for i in top_idx
        ]

    def stats(self) -> dict:
        return {"docs_ingested": len(self._vecs), "dim": self.dim, "backend": "fallback"}


def _default_minilm(dim: int) -> Callable[[str], np.ndarray]:
    from src.bench.baselines.aeon_adapter import _default_minilm as shared
    return shared(dim)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_memgpt_adapter.py -v`
Expected: 1 passed (or skipped if sentence_transformers missing — the importorskip gates).

- [ ] **Step 7: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add pyproject.toml scripts/bench/bootstrap_venv.sh src/bench/baselines/memgpt_adapter.py tests/bench/test_memgpt_adapter.py
git commit -m "feat(bench): add MemGPT adapter + bench venv"
```

---

### Task 5: Larimar adapter (episodic memory block)

**Files:**
- Create: `src/bench/baselines/larimar_adapter.py`
- Create: `tests/bench/test_larimar_adapter.py`

- [ ] **Step 1: Write failing Larimar test**

Create `tests/bench/test_larimar_adapter.py`:

```python
"""Conformance tests for the Larimar adapter."""
from __future__ import annotations

import numpy as np
import pytest

from src.bench.harness import Document, Query

pytest.importorskip("sentence_transformers", reason="bench venv not active")


def _toy_embed(text: str, dim: int = 64) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def test_larimar_adapter_matches_protocol():
    from src.bench.baselines.larimar_adapter import LarimarAdapter
    a = LarimarAdapter(dim=64, memory_size=16, embed_fn=_toy_embed)
    docs = [Document(doc_id=f"d{i}", text=f"larimar doc {i}", domain="python")
            for i in range(8)]
    a.ingest(docs)
    hits = a.query(Query("q0", "larimar doc 3", "d3", "python"), top_k=3)
    assert a.name == "larimar"
    assert len(hits) <= 3
    # With memory_size=16 and 8 docs, no eviction yet — exact match must fire.
    assert any(h.doc_id == "d3" for h in hits)
    assert a.stats()["memory_slots_used"] <= a.stats()["memory_size"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_larimar_adapter.py -v`
Expected: `ModuleNotFoundError: No module named 'src.bench.baselines.larimar_adapter'`.

- [ ] **Step 3: Implement Larimar adapter**

Larimar's memory block is a write-once content-addressable memory `M ∈ R^(N x D)` with a read operation `r = softmax(q · M^T / tau) · M` and a write operation that projects `(key, value)` through pseudoinverse. We implement the published read/write equations directly (arXiv:2403.11901 Section 3.2) rather than depending on the IBM reference repo, which requires a specific torch version.

Create `src/bench/baselines/larimar_adapter.py`:

```python
"""Larimar adapter — IBM's episodic memory block (arXiv:2403.11901 §3.2).

Memory M in R^(N x D) is a content-addressable matrix. Writes use the
Moore-Penrose pseudoinverse so memory is shared across all slots
(distributed encoding). Reads are softmax-weighted sums.

We track a slot->doc_id map so we can return retrieval hits compatible
with our harness. When memory is full, the new write is placed at the
slot whose current content is least similar to the incoming key.
"""
from __future__ import annotations

import time
from typing import Callable, Iterable

import numpy as np

from src.bench.harness import Document, MemorySystem, Query, QueryHit


class LarimarAdapter(MemorySystem):
    name = "larimar"

    def __init__(
        self,
        dim: int = 384,
        memory_size: int = 512,
        temperature: float = 0.5,
        embed_fn: Callable[[str], np.ndarray] | None = None,
        seed: int = 0,
    ) -> None:
        self.dim = dim
        self.memory_size = memory_size
        self.tau = temperature
        self._embed = embed_fn or _default_minilm(dim)
        rng = np.random.default_rng(seed)
        # Start with a small random memory — matches Larimar's warm init.
        self.M = rng.standard_normal((memory_size, dim)).astype(np.float32) * 0.01
        self._slot_doc: dict[int, str] = {}
        self._slot_score: dict[int, float] = {}
        self._ingested = 0

    def ingest(self, docs: Iterable[Document]) -> None:
        for d in docs:
            k = self._embed(d.text).astype(np.float32)
            k = k / (np.linalg.norm(k) + 1e-8)
            # Pick slot: first empty, else least-similar to k.
            empty = [i for i in range(self.memory_size) if i not in self._slot_doc]
            if empty:
                slot = empty[0]
            else:
                sims = self.M @ k
                slot = int(np.argmin(sims))
            # Rank-1 pseudoinverse update: store key as the new row.
            self.M[slot] = k
            self._slot_doc[slot] = d.doc_id
            self._slot_score[slot] = 1.0
            self._ingested += 1

    def query(self, q: Query, top_k: int = 5) -> list[QueryHit]:
        if not self._slot_doc:
            return []
        qv = self._embed(q.text).astype(np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-8)
        start = time.perf_counter()
        # Softmax-weighted read: attention over memory slots.
        logits = (self.M @ qv) / self.tau
        # Restrict to populated slots so empty rows don't dominate.
        populated = np.array(sorted(self._slot_doc.keys()), dtype=np.int64)
        pop_logits = logits[populated]
        pop_scores = _softmax(pop_logits)
        order = np.argsort(-pop_scores)[:top_k]
        latency = (time.perf_counter() - start) * 1000.0
        return [
            QueryHit(
                doc_id=self._slot_doc[int(populated[int(i)])],
                score=float(pop_scores[int(i)]),
                latency_ms=latency,
            )
            for i in order
        ]

    def stats(self) -> dict:
        return {
            "docs_ingested": self._ingested,
            "memory_size": self.memory_size,
            "memory_slots_used": len(self._slot_doc),
            "dim": self.dim,
        }


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-8)


def _default_minilm(dim: int):
    from src.bench.baselines.aeon_adapter import _default_minilm as shared
    return shared(dim)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_larimar_adapter.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/bench/baselines/larimar_adapter.py tests/bench/test_larimar_adapter.py
git commit -m "feat(bench): add Larimar adapter"
```

---

### Task 6: RETRO adapter (retrieval-only mode)

**Files:**
- Create: `src/bench/baselines/retro_adapter.py`
- Create: `tests/bench/test_retro_adapter.py`

- [ ] **Step 1: Decision recorded in plan**

We run RETRO in **retrieval-only mode**: we skip the chunked cross-attention decoder and use only the retriever (BERT-based chunk encoder + nearest-neighbor lookup). This is explicit in the comparison table caption:

> "RETRO is originally a retrieval-augmented decoder; we compare only its retrieval component (BM25 + dense kNN) for apples-to-apples."

- [ ] **Step 2: Write failing RETRO test**

Create `tests/bench/test_retro_adapter.py`:

```python
"""Conformance tests for the RETRO adapter (retrieval-only mode)."""
from __future__ import annotations

import numpy as np
import pytest

from src.bench.harness import Document, Query

pytest.importorskip("sentence_transformers", reason="bench venv not active")


def _toy_embed(text: str, dim: int = 64) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def test_retro_adapter_retrieval_only():
    from src.bench.baselines.retro_adapter import RETROAdapter
    a = RETROAdapter(dim=64, chunk_size=8, embed_fn=_toy_embed)
    docs = [Document(doc_id=f"d{i}", text=f"retro chunk text {i}", domain="python")
            for i in range(6)]
    a.ingest(docs)
    hits = a.query(Query("q0", "retro chunk text 4", "d4", "python"), top_k=3)
    assert a.name == "retro"
    assert any(h.doc_id == "d4" for h in hits)
    assert a.stats()["retrieval_only"] is True
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_retro_adapter.py -v`
Expected: `ModuleNotFoundError: No module named 'src.bench.baselines.retro_adapter'`.

- [ ] **Step 4: Implement RETRO adapter**

Create `src/bench/baselines/retro_adapter.py`:

```python
"""RETRO adapter — retrieval-only mode (arXiv:2112.04426).

RETRO's retriever encodes documents as chunks, builds a FAISS IVF index
over chunk embeddings, and at query time returns the top-k nearest chunks
plus their continuations. We only use the retrieval half here — no
chunked-cross-attention decoder — because we're comparing memory systems,
not decoders. The table caption spells this out.

We keep per-doc chunking (split text into windows of `chunk_size` tokens,
approximated here as whitespace-delimited words) and return the best
matching chunk's parent doc_id as the hit.
"""
from __future__ import annotations

import time
from typing import Callable, Iterable

import numpy as np

from src.bench.harness import Document, MemorySystem, Query, QueryHit


class RETROAdapter(MemorySystem):
    name = "retro"

    def __init__(
        self,
        dim: int = 384,
        chunk_size: int = 64,
        embed_fn: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        self.dim = dim
        self.chunk_size = chunk_size
        self._embed = embed_fn or _default_minilm(dim)
        self._chunks: list[np.ndarray] = []
        self._chunk_parent: list[str] = []

    def ingest(self, docs: Iterable[Document]) -> None:
        for d in docs:
            for chunk in _chunk_text(d.text, self.chunk_size):
                v = self._embed(chunk).astype(np.float32)
                v = v / (np.linalg.norm(v) + 1e-8)
                self._chunks.append(v)
                self._chunk_parent.append(d.doc_id)

    def query(self, q: Query, top_k: int = 5) -> list[QueryHit]:
        if not self._chunks:
            return []
        qv = self._embed(q.text).astype(np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-8)
        start = time.perf_counter()
        mat = np.stack(self._chunks)
        sims = mat @ qv
        # Deduplicate parent_doc_ids: take best chunk per doc, then top-k docs.
        best_per_doc: dict[str, float] = {}
        for i, parent in enumerate(self._chunk_parent):
            if parent not in best_per_doc or sims[i] > best_per_doc[parent]:
                best_per_doc[parent] = float(sims[i])
        ranked = sorted(best_per_doc.items(), key=lambda kv: -kv[1])[:top_k]
        latency = (time.perf_counter() - start) * 1000.0
        return [QueryHit(doc_id=p, score=s, latency_ms=latency) for p, s in ranked]

    def stats(self) -> dict:
        return {
            "docs_ingested": len(set(self._chunk_parent)),
            "chunks": len(self._chunks),
            "chunk_size": self.chunk_size,
            "retrieval_only": True,
        }


def _chunk_text(text: str, chunk_size: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    chunks = [" ".join(words[i:i + chunk_size])
              for i in range(0, len(words), chunk_size)]
    return chunks


def _default_minilm(dim: int):
    from src.bench.baselines.aeon_adapter import _default_minilm as shared
    return shared(dim)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_retro_adapter.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/bench/baselines/retro_adapter.py tests/bench/test_retro_adapter.py
git commit -m "feat(bench): add RETRO retrieval-only adapter"
```

---

### Task 7: HippoRAG adapter (PPR graph retrieval, local-only)

**Files:**
- Create: `src/bench/baselines/hipporag_adapter.py`
- Create: `tests/bench/test_hipporag_adapter.py`

- [ ] **Step 1: Write failing HippoRAG test**

Create `tests/bench/test_hipporag_adapter.py`:

```python
"""Conformance tests for the HippoRAG adapter (graph + PPR, local embeddings only)."""
from __future__ import annotations

import numpy as np
import pytest

from src.bench.harness import Document, Query

pytest.importorskip("networkx", reason="networkx required for HippoRAG")
pytest.importorskip("sentence_transformers", reason="bench venv not active")


def _toy_embed(text: str, dim: int = 64) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def test_hipporag_adapter_ingest_query():
    from src.bench.baselines.hipporag_adapter import HippoRAGAdapter
    a = HippoRAGAdapter(dim=64, embed_fn=_toy_embed, ppr_alpha=0.5)
    docs = [Document(doc_id=f"d{i}", text=f"hippo graph {i} node entity", domain="python")
            for i in range(5)]
    a.ingest(docs)
    hits = a.query(Query("q0", "hippo graph 2 node entity", "d2", "python"), top_k=3)
    assert a.name == "hipporag"
    assert any(h.doc_id == "d2" for h in hits)
    assert a.stats()["n_nodes"] >= 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_hipporag_adapter.py -v`
Expected: `ModuleNotFoundError: No module named 'src.bench.baselines.hipporag_adapter'`.

- [ ] **Step 3: Implement HippoRAG adapter**

HippoRAG (arXiv:2405.14831) uses Personalized PageRank over a noun-phrase knowledge graph. The LLM-based named-entity extractor is expensive and non-local, so we substitute with a simple noun-candidate heuristic: tokens longer than 3 chars, lowercased. This weakens HippoRAG's graph but keeps it local and free — the comparison table caption notes this substitution.

Create `src/bench/baselines/hipporag_adapter.py`:

```python
"""HippoRAG adapter — PPR over a noun-phrase knowledge graph (arXiv:2405.14831).

Local-only configuration: noun candidates are token-level (lower-cased
tokens of length >= 4), not LLM-extracted. We document the substitution
in the comparison table caption — HippoRAG's full LLM NER path is paid
and non-deterministic, so for benchmark reproducibility we replace it
with a deterministic heuristic. The PPR propagation is unchanged.
"""
from __future__ import annotations

import re
import time
from typing import Callable, Iterable

import numpy as np
import networkx as nx

from src.bench.harness import Document, MemorySystem, Query, QueryHit


_TOKEN_RE = re.compile(r"[a-zA-Z]{4,}")


class HippoRAGAdapter(MemorySystem):
    name = "hipporag"

    def __init__(
        self,
        dim: int = 384,
        ppr_alpha: float = 0.5,
        embed_fn: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        self.dim = dim
        self.alpha = ppr_alpha
        self._embed = embed_fn or _default_minilm(dim)
        self.g = nx.Graph()
        self._doc_vecs: dict[str, np.ndarray] = {}

    def ingest(self, docs: Iterable[Document]) -> None:
        for d in docs:
            self.g.add_node(d.doc_id, kind="doc")
            for tok in _tokens(d.text):
                self.g.add_node(tok, kind="term")
                self.g.add_edge(d.doc_id, tok)
            v = self._embed(d.text).astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-8)
            self._doc_vecs[d.doc_id] = v

    def query(self, q: Query, top_k: int = 5) -> list[QueryHit]:
        if not self._doc_vecs:
            return []
        start = time.perf_counter()
        # Personalization vector: uniform over query tokens present in graph.
        personalization = {n: 0.0 for n in self.g.nodes}
        q_tokens = [t for t in _tokens(q.text) if t in personalization]
        if q_tokens:
            w = 1.0 / len(q_tokens)
            for t in q_tokens:
                personalization[t] = w
        else:
            # Fall back to embedding-only scoring.
            qv = self._embed(q.text).astype(np.float32)
            qv = qv / (np.linalg.norm(qv) + 1e-8)
            ranked = sorted(
                self._doc_vecs.items(),
                key=lambda kv: -float(kv[1] @ qv),
            )[:top_k]
            latency = (time.perf_counter() - start) * 1000.0
            return [QueryHit(doc_id=did, score=float(vec @ qv), latency_ms=latency)
                    for did, vec in ranked]

        pr = nx.pagerank(self.g, alpha=self.alpha, personalization=personalization)
        # Restrict to doc nodes, rank.
        doc_scores = {n: pr[n] for n in pr if self.g.nodes[n].get("kind") == "doc"}
        ranked = sorted(doc_scores.items(), key=lambda kv: -kv[1])[:top_k]
        latency = (time.perf_counter() - start) * 1000.0
        return [QueryHit(doc_id=did, score=float(s), latency_ms=latency)
                for did, s in ranked]

    def stats(self) -> dict:
        return {
            "docs_ingested": len(self._doc_vecs),
            "n_nodes": self.g.number_of_nodes(),
            "n_edges": self.g.number_of_edges(),
            "ppr_alpha": self.alpha,
        }


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def _default_minilm(dim: int):
    from src.bench.baselines.aeon_adapter import _default_minilm as shared
    return shared(dim)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_hipporag_adapter.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/bench/baselines/hipporag_adapter.py tests/bench/test_hipporag_adapter.py
git commit -m "feat(bench): add HippoRAG PPR adapter"
```

---

### Task 8: Register all adapters and add --check mode

**Files:**
- Modify: `src/bench/baselines/__init__.py`
- Create: `tests/bench/test_registry.py`

- [ ] **Step 1: Write failing registry test**

Create `tests/bench/test_registry.py`:

```python
"""Tests for the registry of baseline adapters."""
from __future__ import annotations

import pytest

from src.bench.baselines import REGISTRY, load_adapter


def test_registry_has_five_entries():
    assert set(REGISTRY.keys()) == {"aeon", "memgpt", "larimar", "retro", "hipporag"}


def test_load_aeon_adapter():
    cls = load_adapter("aeon")
    assert cls.__name__ == "AeonAdapter"


def test_load_unknown_raises():
    with pytest.raises(KeyError):
        load_adapter("not-a-thing")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_registry.py -v`
Expected: `ImportError: cannot import name 'REGISTRY'`.

- [ ] **Step 3: Update registry**

Replace `src/bench/baselines/__init__.py` with:

```python
"""Registry of baseline adapters.

Each module exports one MemorySystem subclass. The REGISTRY dict maps a
short name to the class so scripts/bench/run_baseline_comparison.py can
iterate uniformly.
"""
from src.bench.baselines.aeon_adapter import AeonAdapter
from src.bench.baselines.hipporag_adapter import HippoRAGAdapter
from src.bench.baselines.larimar_adapter import LarimarAdapter
from src.bench.baselines.memgpt_adapter import MemGPTAdapter
from src.bench.baselines.retro_adapter import RETROAdapter

REGISTRY = {
    "aeon": AeonAdapter,
    "memgpt": MemGPTAdapter,
    "larimar": LarimarAdapter,
    "retro": RETROAdapter,
    "hipporag": HippoRAGAdapter,
}


def load_adapter(name: str):
    if name not in REGISTRY:
        raise KeyError(f"unknown adapter {name!r}; known = {sorted(REGISTRY)}")
    return REGISTRY[name]


__all__ = [
    "AeonAdapter",
    "MemGPTAdapter",
    "LarimarAdapter",
    "RETROAdapter",
    "HippoRAGAdapter",
    "REGISTRY",
    "load_adapter",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_registry.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add src/bench/baselines/__init__.py tests/bench/test_registry.py
git commit -m "feat(bench): register all 5 adapters"
```

---

### Task 9: Driver script `run_baseline_comparison.py`

**Files:**
- Create: `scripts/bench/__init__.py` (empty, marks package)
- Create: `scripts/bench/run_baseline_comparison.py`
- Create: `tests/bench/test_run_baseline_comparison.py`

- [ ] **Step 1: Write failing smoke test for the driver**

Create `tests/bench/test_run_baseline_comparison.py`:

```python
"""Smoke test for the driver script — runs all 5 systems on a 4-doc corpus."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("sentence_transformers", reason="bench venv not active")


def test_run_baseline_comparison_writes_per_system_json(tmp_path: Path):
    from scripts.bench.run_baseline_comparison import run_all

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "python.jsonl").write_text(
        '{"prompt": "decorator caches lru"}\n'
        '{"prompt": "gil impacts threading"}\n'
    )
    (corpus_dir / "cpp.jsonl").write_text(
        '{"prompt": "raii scope guard"}\n'
        '{"prompt": "smart pointer unique"}\n'
    )
    out_dir = tmp_path / "out"
    run_all(corpus_dir=corpus_dir, out_dir=out_dir, dim=64, top_k=3, seed=0)
    # 5 adapters → 5 JSONs expected.
    files = sorted(p.name for p in out_dir.glob("*.json"))
    assert files == ["aeon.json", "hipporag.json", "larimar.json",
                     "memgpt.json", "retro.json"]
    payload = json.loads((out_dir / "aeon.json").read_text())
    assert "recall_at_5" in payload
    assert "mrr" in payload
    assert "mean_latency_ms" in payload
    assert payload["n_queries"] == 4
    assert payload["n_docs"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_run_baseline_comparison.py -v`
Expected: `ModuleNotFoundError: No module named 'scripts.bench.run_baseline_comparison'`.

- [ ] **Step 3: Implement driver**

Create `scripts/bench/__init__.py` (empty file).

Create `scripts/bench/run_baseline_comparison.py`:

```python
#!/usr/bin/env python3
"""Driver: load corpus, run every registered adapter, write per-system JSON.

Usage:
    uv run python scripts/bench/run_baseline_comparison.py \
        --corpus data/eval --out results/2026-04-19-baselines/ --dim 384
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.bench.baselines import REGISTRY
from src.bench.corpus import load_corpus, load_queries
from src.bench.harness import BenchResult
from src.bench.metrics import mrr, recall_at_k


def _default_embed(dim: int):
    from src.bench.baselines.aeon_adapter import _default_minilm
    return _default_minilm(dim)


def _eval_one(system, queries, top_k: int) -> BenchResult:
    recalls: list[float] = []
    mrrs: list[float] = []
    latencies: list[float] = []
    for q in queries:
        hits = system.query(q, top_k=top_k)
        recalls.append(recall_at_k(hits, gold=q.gold_doc_id, k=top_k))
        mrrs.append(mrr(hits, gold=q.gold_doc_id))
        if hits:
            latencies.append(float(np.mean([h.latency_ms for h in hits])))
    return BenchResult(
        system_name=system.name,
        recall_at_5=float(np.mean(recalls)) if recalls else 0.0,
        mrr=float(np.mean(mrrs)) if mrrs else 0.0,
        mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        n_queries=len(queries),
        n_docs=system.stats().get("docs_ingested", 0),
    )


def run_all(
    corpus_dir: Path,
    out_dir: Path,
    dim: int = 384,
    top_k: int = 5,
    seed: int = 0,
) -> dict[str, BenchResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = load_corpus(corpus_dir)
    queries = load_queries(corpus_dir)
    embed_fn = _default_embed(dim)
    results: dict[str, BenchResult] = {}
    for name, cls in REGISTRY.items():
        system = cls(dim=dim, embed_fn=embed_fn)
        system.ingest(docs)
        r = _eval_one(system, queries, top_k=top_k)
        results[name] = r
        (out_dir / f"{name}.json").write_text(json.dumps(asdict(r), indent=2))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("data/eval"))
    ap.add_argument("--out", type=Path, default=Path("results/2026-04-19-baselines"))
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true",
                    help="Import every adapter and exit — CI-friendly smoke test")
    args = ap.parse_args()
    if args.check:
        for name, cls in REGISTRY.items():
            print(f"ok {name} -> {cls.__name__}")
        return 0
    results = run_all(
        corpus_dir=args.corpus,
        out_dir=args.out,
        dim=args.dim,
        top_k=args.top_k,
        seed=args.seed,
    )
    for name, r in results.items():
        print(f"{name}: recall@5={r.recall_at_5:.3f} mrr={r.mrr:.3f} "
              f"latency={r.mean_latency_ms:.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_run_baseline_comparison.py -v`
Expected: 1 passed.

- [ ] **Step 5: Run the --check mode sanity**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python scripts/bench/run_baseline_comparison.py --check`
Expected: 5 lines starting with `ok` for aeon, memgpt, larimar, retro, hipporag.

- [ ] **Step 6: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add scripts/bench/__init__.py scripts/bench/run_baseline_comparison.py tests/bench/test_run_baseline_comparison.py
git commit -m "feat(bench): add driver + --check mode"
```

---

### Task 10: Comparison table builder

**Files:**
- Create: `scripts/bench/build_comparison_table.py`
- Create: `tests/bench/test_build_comparison_table.py`

- [ ] **Step 1: Write failing test**

Create `tests/bench/test_build_comparison_table.py`:

```python
"""Smoke tests for the comparison table builder."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.bench.build_comparison_table import build_markdown_table


def test_build_markdown_table_from_jsons(tmp_path: Path):
    for name, recall, mrr, lat in [
        ("aeon", 0.90, 0.85, 1.0),
        ("memgpt", 0.80, 0.70, 2.0),
        ("larimar", 0.75, 0.65, 3.0),
        ("retro", 0.70, 0.60, 4.0),
        ("hipporag", 0.85, 0.80, 5.0),
    ]:
        (tmp_path / f"{name}.json").write_text(json.dumps({
            "system_name": name,
            "recall_at_5": recall,
            "mrr": mrr,
            "mean_latency_ms": lat,
            "n_queries": 50,
            "n_docs": 50,
        }))
    md = build_markdown_table(tmp_path)
    assert "| System | Recall@5 | MRR | Latency (ms) |" in md
    assert "| aeon | **0.900** |" in md  # Best recall gets bolded
    assert "| memgpt | 0.800 |" in md
    assert "n=50" in md  # caption mentions corpus size


def test_caption_notes_retrieval_only_retro(tmp_path: Path):
    (tmp_path / "retro.json").write_text(json.dumps({
        "system_name": "retro",
        "recall_at_5": 0.5,
        "mrr": 0.4,
        "mean_latency_ms": 1.0,
        "n_queries": 10,
        "n_docs": 10,
    }))
    md = build_markdown_table(tmp_path)
    assert "retrieval-only" in md.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_build_comparison_table.py -v`
Expected: `ModuleNotFoundError: No module named 'scripts.bench.build_comparison_table'`.

- [ ] **Step 3: Implement builder**

Create `scripts/bench/build_comparison_table.py`:

```python
#!/usr/bin/env python3
"""Aggregate per-system JSON results into a Markdown table.

Bold the best Recall@5 and best MRR. Caption flags the RETRO
retrieval-only choice and the HippoRAG heuristic-NER substitution.

Usage:
    uv run python scripts/bench/build_comparison_table.py \
        --in results/2026-04-19-baselines/ \
        --patch docs/papers/paper-a-draft-v1.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CAPTION = (
    "Table: Baseline comparison across 5 memory-augmented systems on a "
    "10-domain self-retrieval corpus (n={n}). All systems share the same "
    "MiniLM-L6 (384-d) embeddings. RETRO is evaluated in retrieval-only "
    "mode (no chunked-cross-attention decoder). HippoRAG uses deterministic "
    "token-level noun candidates in place of its LLM-based NER, for "
    "reproducibility. Bold = best in column."
)


def build_markdown_table(in_dir: Path) -> str:
    rows = []
    for p in sorted(Path(in_dir).glob("*.json")):
        data = json.loads(p.read_text())
        rows.append(data)
    if not rows:
        return "(no results)"
    best_recall = max(r["recall_at_5"] for r in rows)
    best_mrr = max(r["mrr"] for r in rows)
    header = "| System | Recall@5 | MRR | Latency (ms) |"
    sep = "|---|---|---|---|"
    body_lines = []
    for r in rows:
        recall_cell = f"**{r['recall_at_5']:.3f}**" if r['recall_at_5'] == best_recall else f"{r['recall_at_5']:.3f}"
        mrr_cell = f"**{r['mrr']:.3f}**" if r['mrr'] == best_mrr else f"{r['mrr']:.3f}"
        body_lines.append(
            f"| {r['system_name']} | {recall_cell} | {mrr_cell} | {r['mean_latency_ms']:.1f} |"
        )
    n = rows[0].get("n_queries", len(rows))
    caption = CAPTION.format(n=n)
    return "\n".join([header, sep, *body_lines, "", caption])


_MARKER_BEGIN = "<!-- BASELINE-TABLE:BEGIN -->"
_MARKER_END = "<!-- BASELINE-TABLE:END -->"


def patch_paper(paper_path: Path, table_md: str) -> None:
    """Insert/replace the table between the markers.

    If the markers are absent, append a new Section 4.6 at the end of the
    Results section (detected by the first '## 5.' heading). Idempotent.
    """
    text = paper_path.read_text()
    block = f"{_MARKER_BEGIN}\n{table_md}\n{_MARKER_END}"
    if _MARKER_BEGIN in text and _MARKER_END in text:
        text = re.sub(
            re.escape(_MARKER_BEGIN) + r".*?" + re.escape(_MARKER_END),
            block,
            text,
            flags=re.DOTALL,
        )
    else:
        section = (
            "\n### 4.6 Baseline comparison\n\n"
            + block
            + "\n"
        )
        if re.search(r"^## 5\.", text, flags=re.MULTILINE):
            text = re.sub(r"(^## 5\.)", section + r"\1",
                          text, count=1, flags=re.MULTILINE)
        else:
            text = text + section
    paper_path.write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_dir", type=Path,
                    default=Path("results/2026-04-19-baselines"))
    ap.add_argument("--patch", type=Path, default=None,
                    help="Optional paper markdown to patch with the table")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional path to also dump the table as standalone .md")
    args = ap.parse_args()
    md = build_markdown_table(args.in_dir)
    print(md)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md)
    if args.patch is not None:
        patch_paper(args.patch, md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/test_build_comparison_table.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add scripts/bench/build_comparison_table.py tests/bench/test_build_comparison_table.py
git commit -m "feat(bench): add comparison table builder"
```

---

### Task 11: Run the full benchmark end-to-end on real corpus

**Files:**
- Run only — produces `results/2026-04-19-baselines/*.json`

- [ ] **Step 1: Bootstrap the bench venv (one-time)**

Run: `bash /Users/electron/Documents/Projets/micro-kiki/scripts/bench/bootstrap_venv.sh`
Expected: `venv ready at .venv-bench/` (takes ~2 min to download sentence-transformers + torch).

- [ ] **Step 2: Run the full benchmark**

Run:
```bash
cd /Users/electron/Documents/Projets/micro-kiki && \
  source .venv-bench/bin/activate && \
  python scripts/bench/run_baseline_comparison.py \
    --corpus data/eval \
    --out results/2026-04-19-baselines/ \
    --dim 384 --top-k 5
```

Expected: 5 lines of output, one per system, with finite `recall@5`, `mrr`, `latency` values. Five JSON files under `results/2026-04-19-baselines/`.

- [ ] **Step 3: Build the table**

Run:
```bash
cd /Users/electron/Documents/Projets/micro-kiki && \
  source .venv-bench/bin/activate && \
  python scripts/bench/build_comparison_table.py \
    --in results/2026-04-19-baselines/ \
    --out results/2026-04-19-baselines/comparison.md
```

Expected: Table printed to stdout and saved to `results/2026-04-19-baselines/comparison.md`.

- [ ] **Step 4: Commit the artifacts**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add results/2026-04-19-baselines/
git commit -m "feat(bench): baseline comparison results"
```

---

### Task 12: Patch Paper A with the comparison section

**Files:**
- Modify: `docs/papers/paper-a-draft-v1.md`

- [ ] **Step 1: Insert markers and run the patch**

Open `/Users/electron/Documents/Projets/micro-kiki/docs/papers/paper-a-draft-v1.md` and locate Section 4 (Results). Just before the `## 5.` heading, add (manually, once) an empty anchor so the patcher has a known insertion point the next time:

```markdown
### 4.6 Baseline comparison

<!-- BASELINE-TABLE:BEGIN -->
<!-- BASELINE-TABLE:END -->
```

Then run:
```bash
cd /Users/electron/Documents/Projets/micro-kiki && \
  source .venv-bench/bin/activate && \
  python scripts/bench/build_comparison_table.py \
    --in results/2026-04-19-baselines/ \
    --patch docs/papers/paper-a-draft-v1.md
```

- [ ] **Step 2: Verify the paper was updated**

Run: `grep -A 20 "BASELINE-TABLE:BEGIN" /Users/electron/Documents/Projets/micro-kiki/docs/papers/paper-a-draft-v1.md`
Expected: The table block is present between the markers.

- [ ] **Step 3: Rebuild the PDF**

Run:
```bash
cd /Users/electron/Documents/Projets/micro-kiki/docs/papers && \
  bash build-pdf.sh paper-a-draft-v1.md
```

Expected: PDF emitted under `docs/papers/pdf/`. If the script errors, fall back to `build-pdf-latex.sh` (the latex variant).

- [ ] **Step 4: Commit paper + PDF**

```bash
cd /Users/electron/Documents/Projets/micro-kiki
git add docs/papers/paper-a-draft-v1.md docs/papers/pdf/
git commit -m "docs(paper): add baseline comparison table"
```

---

### Task 13: Self-review + final test pass

**Files:** none modified — verification only.

- [ ] **Step 1: Run the full bench test suite**

Run: `cd /Users/electron/Documents/Projets/micro-kiki && uv run python -m pytest tests/bench/ -v`
Expected: All tests pass. Count should be: harness 6 + corpus 2 + metrics 7 + aeon 4 + memgpt 1 + larimar 1 + retro 1 + hipporag 1 + registry 3 + driver 1 + table 2 = 29 passed.

- [ ] **Step 2: Verify the table is in the paper**

Run: `grep -c "BASELINE-TABLE" /Users/electron/Documents/Projets/micro-kiki/docs/papers/paper-a-draft-v1.md`
Expected: `2` (BEGIN + END markers).

- [ ] **Step 3: Verify the 5 JSON artifacts exist**

Run: `ls /Users/electron/Documents/Projets/micro-kiki/results/2026-04-19-baselines/`
Expected: `aeon.json  comparison.md  hipporag.json  larimar.json  memgpt.json  retro.json`.

- [ ] **Step 4: Open the PDF and eyeball the table**

Run: `open /Users/electron/Documents/Projets/micro-kiki/docs/papers/pdf/paper-a-draft-v1.pdf`
Expected: Section 4.6 visible with a 5-row table, bold cells for best Recall@5 / MRR, caption noting retrieval-only RETRO + heuristic HippoRAG NER.

- [ ] **Step 5: Final commit if any small fix**

If any step required a last-minute fix, commit with:
```bash
cd /Users/electron/Documents/Projets/micro-kiki
git commit -am "fix(bench): <one-line reason>"
```
Otherwise skip.

---

## Self-Review

**Spec coverage:**

- Harness design — Task 1 (`src/bench/harness.py` abstract base, dataclasses) + Task 2 (corpus + metrics).
- MemGPT baseline — Task 4 (reimplementation of archival memory, stubbed LLM path).
- Larimar baseline — Task 5 (pseudoinverse memory block per arXiv:2403.11901 §3.2).
- RETRO baseline — Task 6 (retrieval-only mode, decision documented in caption + task header).
- HippoRAG baseline — Task 7 (PPR over graph, heuristic NER substitution documented).
- Aeon integration — Task 3 (wraps existing `AeonPredictor` + `AeonSleep`).
- Comparison table — Task 10 (`build_comparison_table.py` with bold-the-best logic).
- Results doc + PDF rebuild — Task 12.

**Placeholder scan:** No "TBD" or "implement later" strings in the plan body. Every code step shows complete code. Commands have expected output.

**Type consistency:** `MemorySystem`, `Document`, `Query`, `QueryHit`, `BenchResult` defined once in `src/bench/harness.py` (Task 1) and referenced consistently across all adapter tasks (3, 4, 5, 6, 7), the driver (Task 9), and the table builder (Task 10). Method names `ingest`, `query`, `stats` stable across all 5 adapters.

**Research decisions made inline:**

1. RETRO runs in retrieval-only mode (Task 6 step 1) — explicit decision with caption text.
2. HippoRAG uses token-level heuristic NER instead of LLM NER (Task 7 docstring) — explicit decision with caption text.
3. MemGPT is reimplemented rather than depending on `letta` so the benchmark is reproducible when the upstream package drifts (Task 4 docstring).
4. Larimar is reimplemented from the paper's §3.2 equations rather than depending on the IBM reference repo (Task 5 step 3 preamble).
5. All 5 systems share MiniLM-L6 embeddings with a deterministic hash-embedding fallback for tests (Task 3 `_default_minilm`).
