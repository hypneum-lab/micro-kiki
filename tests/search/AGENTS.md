<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# tests/search

## Purpose
Validates the web-search abstraction layer used by Ralph research and the `web_search` capability: a SQLite-backed query cache, the Exa semantic-search backend, the Semantic-Scholar academic backend, and a local Docs backend that scrapes + indexes documentation URLs. All network I/O is mocked with `httpx.AsyncClient` patches — no live API calls.

## Key Files
| File | Description |
|------|-------------|
| `test_cache.py` | `TestSearchCache` — SQLite store/lookup, TTL expiry (`ttl_seconds=0` + `time.sleep(0.1)` → miss), miss-on-unknown-query, and backend-isolation (same query different backends return distinct payloads). |
| `test_exa_backend.py` | `TestExaBackend` — `search()` parses Exa `POST /search` response into `SearchResult` with `source="exa"`, handles empty `results` list, and re-raises on `raise_for_status` errors. |
| `test_scholar_backend.py` | `TestScholarBackend` — Semantic-Scholar `GET` response → `SearchResult` with `source="scholar"` and metadata (`year`, `citations=42` from `citationCount`). Tests empty-result path. |
| `test_docs_backend.py` | `TestDocsBackend` — scrapes HTML via mocked `httpx.AsyncClient.get`, indexes into SQLite, then retrieves via `search()`. Verifies `name == "docs"` and empty-index search returns `[]`. |

## For AI Agents

### Working In This Directory
Run the suite: `uv run python -m pytest tests/search/ -q`.
- No `@pytest.mark.integration` tests here; every backend is exercised against mocked HTTP responses.
- `tmp_path` isolates every SQLite DB (cache + docs index) — no cleanup required.
- `httpx.AsyncClient.post` / `.get` are patched with `new_callable=AsyncMock, return_value=mock_response` where `mock_response.raise_for_status = MagicMock()`.

### Testing Requirements
- Cache invariants: TTL must expire entries (returns `None` after elapsed time); the composite `(backend, query)` key must isolate namespaces.
- Backend invariants: every backend returns a `list[SearchResult]` with the correct `source` attribute; empty API responses map to `[]`; HTTP errors propagate via `raise_for_status`.
- Scholar metadata mapping: API `citationCount` → `SearchResult.metadata["citations"]`; `year` is preserved.
- Docs backend: HTML must be parsed (text content extracted) and stored; subsequent `search()` must hit the index.

### Common Patterns
- `MagicMock()` for the response object, `AsyncMock` for the HTTP client method, explicit `mock_response.raise_for_status = MagicMock()` to suppress real error-raising.
- `@pytest.mark.asyncio` is explicitly used on every async test in this subtree (even though `asyncio_mode = "auto"` is set globally).
- One behavior per test, AAA structure, no shared mutable state.

## Dependencies

### Internal
- `src.search.base` — `SearchResult`
- `src.search.cache` — `SearchCache`
- `src.search.exa_backend` — `ExaBackend`
- `src.search.scholar_backend` — `ScholarBackend`
- `src.search.docs_backend` — `DocsBackend`

### External
- `pytest`, `pytest-asyncio`
- `httpx` (patched)

<!-- MANUAL: -->
