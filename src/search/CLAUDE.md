# src/search/ — pluggable async search backends + cache + Aeon bridge

Three concrete `SearchBackend` implementations behind one ABC: Exa (web), Semantic
Scholar (papers), local SQLite-FTS5 (`docs_backend`). `SearchCache` (SHA256-keyed,
SQLite, per-entry TTL) deduplicates calls across backends. `AeonIndexer` streams
results into Aeon so later `recall` surfaces them with provenance. Primary caller
is `src.ralph.research.StoryResearcher`. File-by-file index in `AGENTS.md`.

## Hard invariants

| Invariant | Why |
|---|---|
| All backends are `async`; no sync escape hatch | `StoryResearcher` awaits Exa+Scholar in sequence per ralph story. |
| `SearchResult` is `@dataclass(frozen=True)` — value object | `AeonIndexer` spreads fields into a dict; field add ⇒ update serialisation in same commit. |
| Cache key = `SHA256(f"{backend}:{query}")` — separator and hash are load-bearing | Older cache rows won't rehydrate if you change either. |
| `source ∈ {"exa", "scholar", "docs"}` is the canonical provenance tag | Used by `AeonIndexer` and downstream recall filters. |
| FTS5 is assumed available; `tokenize='porter'` | All supported CI runners have it. `no such module: fts5` ⇒ wrong runtime, not a bug. |
| HTTP timeout 30 s per request, **no shared `httpx.AsyncClient`** at this layer | Per-request `async with` only; reuse belongs upstream. |

## Coupling

| Direction | Contract |
|---|---|
| ← `src/ralph/research.py` | `StoryResearcher.research_story` awaits Exa then Scholar. |
| → `src/memory/` | `AeonIndexer` expects `aeon_client.store_trace` to be **awaitable** — `AeonPalace.write` is sync, so the adapter wraps it. Don't shortcut by calling sync `write` from here. |
| API keys | `ExaBackend(api_key=...)` constructor arg only; no env reads, no defaults. |

## Anti-patterns

- Sharing one `httpx.AsyncClient` across backend calls "for performance" — request
  isolation is intentional; the cache absorbs the duplication cost.
- Changing the cache key separator (`:`) or hash (SHA256) — silently invalidates
  every persisted cache row without a migration path.
- Adding a sync `search()` overload on a backend — `StoryResearcher` and the ralph
  loop are async end-to-end; a sync escape hatch fragments the call graph.
- Hardcoding an Exa or Scholar API key default — keys must come from env via the
  caller; backends never read `os.environ` directly.
- Calling `aeon_palace.write` directly from `AeonIndexer` — bypasses the async
  shim contract and breaks the ralph test mocks.
- Adding a fourth `source` value without auditing every recall-filter call site
  (search → memory → recall provenance is the audit trail).
