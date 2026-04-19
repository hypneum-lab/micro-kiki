<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# tests/memory

## Purpose
Tests Phase VIII Aeon Memory Palace — the two-layer episodic memory used by the serving hook and the cognitive stack. `AtlasIndex` is the SIMD-style vector layer (cosine recall), `TraceGraph` is the causal/temporal episode graph, and `AeonPalace` composes both behind a single write/recall/walk/compress API.

## Key Files
| File | Description |
|------|-------------|
| `test_aeon.py` | `TestAtlasIndex` — insert+recall roundtrip (cosine similarity > 0.99 for identical vector), empty-index recall returns `[]`, top-k ordering puts the nearer vector first. `TestTraceGraph` — `add_episode` + `add_edge(CausalityEdge)` + `walk("e1", max_depth=2)` returns 2 nodes, `query_by_time` filters by timestamp window, `query_by_rule(domain=...)` filters by domain. `TestAeonPalace` — `write` + `recall` basic path, `write(..., links=[e1])` lets `walk(e1, max_depth=2)` reach the new episode, `compress(older_than=...)` merges/prunes old episodes, and `stats` returns `{vectors, episodes}` counts. |

## For AI Agents

### Working In This Directory
Run the suite: `uv run python -m pytest tests/memory/ -q`.
- No `@pytest.mark.integration` tests — native Python backends only. The Qdrant / Neo4j-backed variants live elsewhere (and would be integration-marked).
- Tests are synchronous (no `pytest.mark.asyncio`).
- `AeonPalace(dim=3072)` is the production embedding dim used in this test file.

### Testing Requirements
- `AtlasIndex.recall` invariants: same vector recalled with score > 0.99 (cosine); empty index returns `[]`; top-k ordering respects similarity descending.
- `TraceGraph.walk` must traverse causality edges up to `max_depth`; `query_by_time` must respect the `[start, end]` window; `query_by_rule` must filter on `domain`.
- `AeonPalace.compress(older_than=...)` must return the count of compressed episodes (at least 1 when a matching old episode exists).
- `AeonPalace.stats` must report both `vectors` and `episodes` counters and they must match the number of `write` calls.

### Common Patterns
- `np.random.randn(dim).astype(np.float32)` for synthetic embeddings.
- `datetime.now()` / `timedelta(days=30)` for time-based queries.
- Positional `Episode(id=..., content=..., domain=..., timestamp=...)` and `CausalityEdge(from_id=..., to_id=..., weight=...)` construction.

## Dependencies

### Internal
- `src.memory.atlas` — `AtlasIndex`
- `src.memory.trace` — `TraceGraph`, `Episode`, `CausalityEdge`
- `src.memory.aeon` — `AeonPalace`

### External
- `pytest`
- `numpy`

<!-- MANUAL: -->
