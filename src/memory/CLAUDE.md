# src/memory/ — Aeon episodic memory (palace + sleep)

Two facades over one data model: `AeonPalace` (v0.2, sync, used by `AeonServingHook`)
and `AeonSleep` (v0.3, the unified facade that also wires `SleepTagger`,
`ForgettingGate`, `Consolidator`). Both share the same `AtlasIndex` (numpy SIMD
vector store, page-clustered) + `TraceGraph` (typed multi-graph) pair. The cognitive
layer (`src/cognitive/`) consumes this tree; this tree never imports `cognitive`
back. File-by-file index is in `AGENTS.md` — do not duplicate it here.

## Hard invariants

| Invariant | Where |
|---|---|
| Embedding dim = 3072 (Qwen3.6 hidden) — asserted on `insert` / `write` | `atlas.py`, `aeonsleep.py` |
| Eviction double gate: `P(keep) < keep_threshold` AND episode covered by a summary cluster | `aeonsleep.py` |
| `summary_of` backref preserved on every evicted-or-merged raw → summary | `aeonsleep.sleep_cycle` |
| `EDGE_KINDS = (temporal, causal, topical, summary_of)` and `NODE_KINDS = (raw, summary)` are closed sets | `trace.py` |
| `AtlasIndex.PAGE_SIZE = 256`, cosine-normalised dot product, pure numpy hot path | `atlas.py` |
| Acceptance: ≥ 95 % recall at PI depth 10 after 500 writes + 3 sleep cycles | `tests/test_aeonsleep.py` |

## Coupling

| Direction | Contract |
|---|---|
| → `src/cognitive/` | `AeonSleep` composes `SleepTagger`, `ForgettingGate`, `Consolidator` via DI in `__init__`; never imports them lazily mid-call. |
| ← `src/serving/aeon_hook.py` | Holds an `AeonPalace`; pre/post-inference write+recall. Async wrapper around sync `write` lives there, not here. |
| ← `src/search/aeon_indexer.py` | Calls `aeon_client.store_trace` (awaitable) — that adapter wraps the sync palace. |
| ↔ `src/eval/forgetting.py` | Stack-level forgetting; **distinct** from per-episode `cognitive/forgetting_gate.py`. Do not conflate. |

## backends/ (Tower-only adapters)

`qdrant_atlas.py` and `neo4j_trace.py` are drop-in alternatives for external infra.
**Lazy import** discipline: nothing in `src.memory.*` may import `qdrant_client` or
`neo4j` at module load — vanilla CI must still load the package cleanly. The
in-process numpy/dict implementations are the source of truth for tests.

## Anti-patterns

- Relaxing the eviction double gate "because P(keep) is already low" — breaks the
  ≥ 95 % PI-depth recall target; the summary-coverage check is load-bearing.
- Using `AeonPalace._default_embed` (SHA256-seeded RNG) anywhere except tests — it
  is a determinism shim, not a real encoder. Production embeddings come from the
  serving tokenizer.
- Removing the v0.2 `add_edge(CausalityEdge)` compat shim on `TraceGraph` — both
  `aeon.py` and `tests/test_aeon_hook.py` depend on auto-creating endpoints.
- Adding a feature to `forgetting_gate` features tuple (`age, access_count, conflict,
  emb_norm`) — that lives in `cognitive/`, but if you change shape you must retrain
  weights and bump the gate version; flag it in the PR description.
- Importing torch into the atlas hot path — keep numpy-only so CI on the GrosMac
  R&D box (no MLX, no Metal) stays green.
- Eagerly importing `qdrant_client` / `neo4j` at module top in `backends/` — defer
  to function scope.
