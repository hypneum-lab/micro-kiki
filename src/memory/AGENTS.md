<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# memory

## Purpose
Aeon memory palace (arxiv 2601.15311) — the dual-index episodic memory that backs both the serving pipeline and the sleep-cycle maintenance. `AtlasIndex` is a page-clustered SIMD-friendly vector store (numpy fallback). `TraceGraph` is a typed directed multi-graph of episodes with causal / temporal / topical / summary-of edges. Two facades sit on top: `AeonPalace` (v0.2, synchronous, dual-index API used by `AeonServingHook`) and `AeonSleep` (v0.3, the unified interface that also wires in `SleepTagger`, `ForgettingGate`, and `Consolidator` for the sleep-cycle benchmark — target ≥ 95 % recall at PI depth 10 after 500 writes + 3 cycles).

## Key Files
| File | Description |
|------|-------------|
| `atlas.py` | `AtlasIndex(dim=3072, num_clusters=16)` — page-clustered vector index, `PAGE_SIZE=256`. `insert`, `recall(query, top_k)`, `search(q, k) → list[SearchHit]`, `rebuild_centroids`, `total_vectors`. Pure numpy dot product, cosine-normalised. `time_search` benchmark helper. |
| `trace.py` | `TraceGraph` — typed directed multi-graph. `EDGE_KINDS=(temporal, causal, topical, summary_of)`, `NODE_KINDS=(raw, summary)`. v0.3 API: `add_node`, `add_typed_edge`, `ancestors`, `descendants`, `time_range`, `remove_node`, `stats`. v0.2 compat API: `Episode`, `CausalityEdge`, `add_episode`, `add_edge`, `walk`, `query_by_time`, `query_by_rule`. |
| `aeon.py` | `AeonPalace(dim=3072, embed_fn=None)` — v0.2 synchronous facade. `write(content, domain, timestamp, links, source, metadata)`, `recall(query, top_k, domain)`, `walk(from_id, max_depth)`, `query_by_time(start, end)`, `compress(older_than, summarize_fn)`. Uses SHA-256-seeded random embed as deterministic test default. |
| `aeonsleep.py` | `AeonSleep(dim, keep_threshold=0.35, ...)` — v0.3 unified facade wiring `AtlasIndex` + `TraceGraph` + `SleepTagger` + `ForgettingGate` + `Consolidator`. Methods: `write(Episode) → Tag`, `recall(query, k) → list[RecallHit]`, `sleep_cycle() → SleepReport`, `query_time(start, end)`, `stats()`. Eviction is guarded by cluster coverage so PI-depth recall stays high. |
| `backends/qdrant_atlas.py` | Qdrant-backed `AtlasIndex` alternative for the Tower deployment. |
| `backends/neo4j_trace.py` | Neo4j-backed `TraceGraph` alternative for the Tower deployment. |

## Subdirectories

`backends/` — optional storage adapters for Qdrant and Neo4j. Used when `AeonPalace`/`AeonSleep` runs against external infra (Tower) instead of the in-process numpy/dict implementation. Import is lazy — nothing here depends on qdrant-client or neo4j at module load.

## For AI Agents

### Working In This Directory
- **Two facades, one data model**: `AeonPalace` (v0.2) and `AeonSleep` (v0.3) share the same `AtlasIndex` + `TraceGraph` pair. Keep compatibility — `src.serving.aeon_hook.AeonServingHook` holds a palace, while `AeonSleep` is reached from sleep-cycle jobs.
- **Eviction invariant** (aeonsleep.py): an episode is dropped only when `P(keep) < keep_threshold` **and** it is already covered by a summary cluster. This double gate is load-bearing for the ≥ 95 % PI-depth recall target — do not relax it.
- **Consolidation preserves backrefs**: every evicted-or-merged raw episode has a `summary_of` edge pointing to its summary node in the Trace graph. Summaries also get pushed into the atlas so recall can hit either kind.
- **Embedding dim**: default 3072 (Qwen3.5-35B hidden size). `AtlasIndex.insert` asserts shape; `AeonSleep.write` validates `len(episode.embedding) == self.dim`.
- **Graph v0.2 compat shim**: `TraceGraph.add_edge(CausalityEdge)` auto-creates missing endpoints and adds a `causal` typed edge. Don't remove — `aeon.py` and the `AeonServingHook` tests depend on it.
- **Determinism in tests**: `AeonPalace._default_embed` is a SHA256-seeded numpy RNG — not for production. Real embeddings must come from the serving layer's tokenizer/model.
- **Neo4j/Qdrant backends are optional**: keep them import-lazy so CI on vanilla Python still loads `src.memory.*` cleanly.

### Testing Requirements
- `tests/memory/test_aeon.py` — palace write/recall/walk/query_by_time/compress.
- `tests/test_atlas.py` — page allocation, recall top-k, centroid rebuild.
- `tests/test_trace.py` — typed edges, ancestors/descendants, time_range.
- `tests/test_aeonsleep.py` — full sleep_cycle, PI-depth recall acceptance (500 episodes / 3 cycles / ≥ 95 % recall).
- `tests/test_consolidation.py`, `tests/test_sleep_tagger.py`, `tests/test_forgetting_gate.py` cover the dependencies.
- `tests/test_aeon_hook.py`, `tests/test_aeon_compress.py` cover serving-side integration.

### Common Patterns
- `@dataclass` for `VectorPage`, `Node`, `Edge`, `Episode` (v0.3 input) with `@dataclass(frozen=True)` for `SearchHit`, `RecallHit`, `SleepReport`, v0.2 `Episode`, and `CausalityEdge`.
- `from __future__ import annotations` universal.
- Pure numpy in the hot path of `AtlasIndex`; no torch at the memory layer.
- Loguru / stdlib `logging.getLogger(__name__)`; no `print()`.
- `hashlib.sha256` for deterministic ids (`AeonPalace.write`).

## Dependencies

### Internal
- `src.cognitive.sleep_tagger`, `src.cognitive.forgetting_gate`, `src.cognitive.consolidation` composed by `AeonSleep`.
- Consumed by `src.serving.aeon_hook.AeonServingHook` (pre/post inference).
- Consumed by `src.search.aeon_indexer.AeonIndexer` (search-result enrichment).
- Consumed by `src.routing.hybrid_pipeline` through the hook.

### External
- `numpy` (atlas, embedding math).
- `qdrant-client` (optional — `backends/qdrant_atlas.py`).
- `neo4j` Python driver (optional — `backends/neo4j_trace.py`).
- stdlib: `hashlib`, `dataclasses`, `datetime`, `collections`, `itertools`, `logging`.

<!-- MANUAL: -->
