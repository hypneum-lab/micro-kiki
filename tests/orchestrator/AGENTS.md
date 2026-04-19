<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# tests/orchestrator

## Purpose
Tests the top-level `OrchestrationEngine` that wires the 5 router capabilities (`web_search`, `self_critique_token`, `self_critique_response`, `self_critique_task`, `deep_eval`) into the generate→critique→search pipeline. Verifies that disabled capabilities are no-ops and that enabled ones inject search context into the prompt before generation.

## Key Files
| File | Description |
|------|-------------|
| `test_engine.py` | `TestOrchestrationEngine.test_simple_query_no_capabilities` — with all 5 capabilities off, `process()` returns the raw `generate_fn` output, empty `search_results`, and `critique_applied is False`. `test_web_search_injects_context` — with `web_search=True`, `engine._search` is patched to return one mocked `SearchResult`, which appears in `result.search_results` and the response reflects generation with sources. |

## For AI Agents

### Working In This Directory
Run the suite: `uv run python -m pytest tests/orchestrator/ -q`.
- No `@pytest.mark.integration` tests; `generate_fn` is an `AsyncMock(return_value=("response", -1.0))` — tuple `(text, log_prob)` to match `BestOfN` expectations.
- `engine._search` is monkeypatched via `patch.object(engine, "_search", AsyncMock(...))` rather than patching the backend directly — coarse-grained mock.
- The `engine` fixture constructs `OrchestrationEngine` with canonical capability thresholds (`web_search=0.15`, `self_critique_token=0.10`, `self_critique_response=0.20`, `self_critique_task=0.35`, `deep_eval=0.25`) and `BestOfN` config (`high=0.8`, `mid=0.5`, `mid_n=3`, `low_n=5`).

### Testing Requirements
- `active_capabilities` is a full dict (all 5 keys present with boolean values) — the engine must not KeyError on any key.
- When `web_search` is active, `result.search_results` must contain the mocked results (length matches); when all capabilities are off, it must be `[]`.
- `router_confidence=0.95` triggers single-shot generation (`BestOfN.select_n → 1`); the test infers this from `response` being the direct `generate_fn` return.
- `result.critique_applied` must be `False` when no self-critique capability is enabled.

### Common Patterns
- Build the full `active_caps` dict as `{k: False for k in [...]}` then flip individual keys to `True` — guarantees no missing keys.
- `patch.object(engine, "_search", mock_search)` inside a `with` block — scoped patching.
- `MagicMock(title=..., url=..., snippet=...)` stand-ins for `SearchResult` objects where exact class identity is not required.

## Dependencies

### Internal
- `src.orchestrator.engine` — `OrchestrationEngine` (composes `BestOfN`, search backends, self-critique modules)

### External
- `pytest`, `pytest-asyncio`

<!-- MANUAL: -->
