<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# orchestrator — Mac Studio ↔ kxkm-ai orchestration layer

## Purpose
Top-level capability router and cross-machine inference bridge for the agentic-capabilities plan (Story 108, Phase XIII of `docs/superpowers/plans/2026-04-15-agentic-capabilities.md`). The `OrchestrationEngine` composes the 5 router-selectable capabilities — web_search, self_critique_token, self_critique_response, self_critique_task, deep_eval — into a single `process()` call, wiring `BestOfN`, `SelfRefine`, and `AgenticLoop` on top of a caller-supplied `generate_fn`. `HttpBridge` is the concrete `generate_fn` used in production: it POSTs to the vLLM OpenAI-compatible endpoint running on `kxkm-ai:8000` and averages token logprobs for the `BestOfN` scorer. Not skeletons — both files are live code backing the end-to-end flow.

## Key Files
| File | Description |
|------|-------------|
| `engine.py` | 105 lines. `OrchestrationResult` (frozen dataclass: response, search_results, critique_applied, iterations) and `OrchestrationEngine`. `__init__` takes `capabilities_config`, `best_of_n_config`, `agentic_max_iterations=5` and constructs a `BestOfN` instance. `process()` is the hot path: optionally runs `_search()` and injects results via `_format_search_context()` into the prompt as a `## Search Results` block, then dispatches on the capability dict — `self_critique_task` wins first and runs an `AgenticLoop` with tools `["search_web", "search_papers"]`; otherwise `self_critique_token` routes through `BestOfN.run(..., router_confidence=...)`; finally `self_critique_response` post-processes through `SelfRefine`. `_search()` is a stub-in-place returning `[]` — real search backends are wired at the capability level, not in this module. |
| `http_bridge.py` | 59 lines. `InferenceResponse` (text, log_prob) and `HttpBridge(vllm_url="http://kxkm-ai:8000", timeout=120.0)`. `generate(prompt, **kwargs)` POSTs to `{vllm_url}/v1/completions` with `model="micro-kiki"`, `max_tokens=2048`, `temperature=0.7`, `logprobs=1`; strips `None` tokens and returns `(text, avg_log_prob)` — the tuple shape expected by `OrchestrationEngine.generate_fn`. `health_check()` GETs `/health` with 5 s timeout and swallows `httpx.RequestError`. |
| `__init__.py` | 1 line — `from __future__ import annotations` only. This is the only true stub in the module; do not delete the others. |

## For AI Agents

### Working In This Directory
- **`engine.py` is production, not a placeholder.** It is the entry point for every orchestrated query. The earlier "skeleton" label in this doc was caused by a reader-side memory cache returning only line 1; the file compiles, imports `BestOfN`/`SelfRefine`/`AgenticLoop`/`SearchResult`, and is covered by tests.
- **Capability-key contract**: `active_capabilities` is a flat `dict[str, bool]` keyed exactly by `web_search`, `self_critique_token`, `self_critique_response`, `self_critique_task`, `deep_eval`. Key drift will silently disable features — the dict uses `.get(key)` with no validation.
- **Dispatch order matters.** `self_critique_task` short-circuits (returns immediately from `AgenticLoop`), skipping `BestOfN` and `SelfRefine`. If you add a new capability, decide up-front whether it is terminal or post-processing.
- **`generate_fn` signature is `(prompt) -> (text, log_prob)`**, not just text. Both the default path and the `BestOfN` scorer depend on the logprob. If you wire a non-vLLM backend, either return `0.0` or a real average-token-logprob — never raise.
- **`_search()` returns `[]` by default**; search backends are intentionally not instantiated here. Wire them via a subclass or a monkeypatch in tests (see `tests/orchestrator/test_engine.py::test_web_search_injects_context`).
- **`HttpBridge` targets vLLM OpenAI-compat**, not raw HF Transformers. Changing the route from `/v1/completions` will break the logprob extraction — vLLM returns `logprobs.token_logprobs: list[float | None]` and the code filters `None` (first token).
- **New `httpx.AsyncClient` per call** — acceptable for the current low-QPS path. If you start benchmarking or hit connection-limit errors, hoist the client to `__init__` and close in an `aclose()` method.

### Testing Requirements
- `tests/orchestrator/test_engine.py` — 2 async tests: `test_simple_query_no_capabilities` (all caps off → raw `generate_fn` output, empty `search_results`, `critique_applied is False`) and `test_web_search_injects_context` (monkeypatches `engine._search` to return one `SearchResult`, asserts it appears in `result.search_results`).
- `tests/test_integration_phase14.py::test_engine_full_flow_mock` — end-to-end with a mock `generate_fn` exercising search + critique chaining.
- Canonical fixture thresholds (mirror these if you add tests): `web_search=0.15`, `self_critique_token=0.10`, `self_critique_response=0.20`, `self_critique_task=0.35`, `deep_eval=0.25`; `BestOfN` config `{high: 0.8, mid: 0.5, mid_n: 3, low_n: 5}`.

### Common Patterns
- Frozen dataclasses for result types (`OrchestrationResult`, `InferenceResponse`).
- Inner `async def gen_text(prompt) -> str` adapters wrap `generate_fn` for callees that only want the text (`AgenticLoop`, `SelfRefine`).
- Explicit `raise_for_status()` on httpx responses; `RequestError` caught only in `health_check`.

## Dependencies

### Internal
- `src.critique.best_of_n.BestOfN`
- `src.critique.self_refine.SelfRefine`
- `src.critique.agentic_loop.AgenticLoop`
- `src.search.base.SearchResult` (typed container for `_search` results)

### External
- `httpx` (async HTTP client)
- Runtime target: vLLM OpenAI-compatible server on `kxkm-ai:8000`

<!-- MANUAL: -->
