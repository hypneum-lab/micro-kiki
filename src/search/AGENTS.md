<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# search

## Purpose
Pluggable search backend family used by the Ralph autonomous loop (`src.ralph.research.StoryResearcher`) and by any agent that needs to ground an answer in external sources. Three concrete backends (Exa web, Semantic Scholar papers, local SQLite-FTS5 doc index) sit behind the abstract `SearchBackend`. A SHA256-keyed `SearchCache` (SQLite with per-entry TTL) deduplicates calls, and `AeonIndexer` streams results into Aeon for long-term enrichment so they show up on later recall.

## Key Files
| File | Description |
|------|-------------|
| `base.py` | Abstract `SearchBackend(ABC)` with `name` property and `async search(query, max_results)`. Frozen dataclass `SearchResult(title, url, snippet, source, metadata)` — `source` is one of `"exa"`, `"scholar"`, `"docs"`. |
| `exa_backend.py` | `ExaBackend` — POSTs to `https://api.exa.ai/search` with `x-api-key` header, `type="auto"`, `contents.text=True`. Snippet trimmed to 500 chars. |
| `scholar_backend.py` | `ScholarBackend` — GETs `https://api.semanticscholar.org/graph/v1/paper/search`. `FIELDS = "title,abstract,url,year,citationCount,authors"`. Metadata captures year, citations, author names, paperId. |
| `docs_backend.py` | `DocsBackend` — local SQLite FTS5 virtual table (`tokenize='porter'`). `index_url` fetches via httpx, strips `<script>`/`<style>`/tags, inserts. `search` uses FTS5 `snippet(docs, 2, '<b>', '</b>', '...', 64)` and cleans highlight tags from the returned snippet. |
| `cache.py` | `SearchCache` — SQLite table `cache(key TEXT PK, results TEXT, expires_at REAL)`. `_make_key(backend, query)` is SHA256 over `f"{backend}:{query}"`. Lazy eviction on `lookup` when TTL has elapsed. |
| `aeon_indexer.py` | `AeonIndexer.index_results(query, results, session_id)` — awaits `aeon_client.store_trace({type: "search_result", query, title, url, snippet, source, session_id, metadata})` per result. Does not own an Aeon client — inject one. |

## For AI Agents

### Working In This Directory
- **All backends are async** and must stay that way — `StoryResearcher` awaits both Exa and Scholar in sequence for every ralph story (see `src.ralph.research.research_story`).
- **`SearchResult` is frozen**: treat as a value object. Callers serialise it by hand (`AeonIndexer` spreads fields into a dict) — do not add fields without updating serialisation in `aeon_indexer.py`.
- **FTS5 available everywhere**: `DocsBackend` assumes SQLite was built with FTS5. All CI runners we support have it; if you see `"no such module: fts5"` the runtime is wrong.
- **Cache key format is load-bearing**: `SHA256("{backend}:{query}")` — don't change the separator or hash, older cache rows won't rehydrate.
- **Don't leak API keys**: `ExaBackend` takes `api_key` as a constructor arg; callers read from env. No hardcoded defaults.
- **`AeonIndexer` expects `aeon_client.store_trace` to be awaitable** — the Aeon palace `write` is synchronous, so wrap or adapt before injection.
- HTTP timeouts are per-request (30s for backends, no global client) — don't reuse `httpx.AsyncClient` across requests at this layer.

### Testing Requirements
- `tests/search/test_exa_backend.py`, `test_scholar_backend.py` — mocked httpx responses, field mapping.
- `tests/search/test_docs_backend.py` — FTS5 round-trip on a small indexed doc.
- `tests/search/test_cache.py` — TTL eviction, SHA256 key determinism.
- No dedicated `test_aeon_indexer.py` — indirectly exercised via ralph tests.

### Common Patterns
- `async with httpx.AsyncClient() as client:` for every network call; explicit `timeout=30.0`.
- `@dataclass(frozen=True)` for `SearchResult`.
- `from __future__ import annotations` on every module.
- `source` string is the canonical cross-layer tag — used by `AeonIndexer` to tag traces and by downstream recall to filter by provenance.
- `logging.getLogger(__name__)` where logging is needed; no `print()`.

## Dependencies

### Internal
- Consumed by `src.ralph.research.StoryResearcher` (exa + scholar).
- `AeonIndexer` expects an Aeon client conforming to `src.memory.aeon.AeonPalace` (with an async `store_trace` shim).

### External
- `httpx` — async HTTP for all network backends.
- `sqlite3` (stdlib) — FTS5 for `docs_backend`, TTL cache for `cache`.
- stdlib: `abc`, `hashlib`, `json`, `time`, `re`, `pathlib`, `dataclasses`.

<!-- MANUAL: -->
