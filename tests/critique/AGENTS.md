<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# tests/critique

## Purpose
Covers the self-critique and response-quality modules: the multi-iteration `AgenticLoop` (plan → act → evaluate → retry / abort), the `SelfRefine` critique-and-correct pass, and `BestOfN` response sampling with confidence-gated N. These tests assert the correctness of inner-loop control flow (iteration counts, abort semantics, retry caps) rather than generation quality itself.

## Key Files
| File | Description |
|------|-------------|
| `test_agentic_loop.py` | `TestAgenticLoop` — single-iteration completion (`iterations == 1` when `meets_expectations=true`), `max_iterations=3` cap when evaluator always returns `should_retry=true`, and immediate abort when `next_action == "abort"`. |
| `test_best_of_n.py` | `TestBestOfN` — confidence→N mapping (`high_threshold=0.8 → 1`, `mid_threshold=0.5 → 3`, else → 5), `generate_candidates` yields `ScoredCandidate` list, `select_best` picks maximum `log_prob`, and `run` with high router confidence generates only a single sample. |
| `test_self_refine.py` | `TestSelfRefine` — no correction path when critique JSON has `needs_correction=false` (original response preserved), correction path on `needs_correction=true` (second `generate_fn` call produces rewritten text), and `CritiqueResult` round-trips `factual_errors` + `confidence`. |

## For AI Agents

### Working In This Directory
Run the suite: `uv run python -m pytest tests/critique/ -q`.
- No `@pytest.mark.integration` tests; every LLM call is replaced by a local `async def mock_generate(prompt: str) -> str` closure that branches on prompt content or `call_count`.
- `asyncio_mode = "auto"` is set globally, but tests here still use explicit `@pytest.mark.asyncio` decorators.
- No heavyweight fixtures — objects are constructed inline per test.

### Testing Requirements
- `AgenticLoop` invariants: `result.iterations` equals the number of evaluator passes; `completed` is `True` iff the final evaluator returned `meets_expectations=true`; `next_action == "abort"` stops the loop immediately (do not retry).
- `BestOfN` invariants: `select_n(confidence)` is a pure function of thresholds; `select_best` is deterministic (max `log_prob`); high-confidence path calls `generate_fn` exactly once.
- `SelfRefine` invariants: when critique is clean, `final_response` equals the input response verbatim and `corrected is False`; when critique flags issues, `generate_fn` is called twice (once for critique, once for rewrite).

### Common Patterns
- Closure-captured `call_count` / `nonlocal` counters to script multi-turn mock responses.
- `json.dumps({...})` payloads match the real prompt-response schema (`meets_expectations`, `should_retry`, `next_action`, `factual_errors`, `needs_correction`, `confidence`).
- Branching in `mock_generate` on prompt substring (`"Break this task"`, `"Evaluate"`, `"critical reviewer"`) — keep in sync with real prompt templates.

## Dependencies

### Internal
- `src.critique.agentic_loop` — `AgenticLoop`
- `src.critique.best_of_n` — `BestOfN`, `ScoredCandidate`
- `src.critique.self_refine` — `SelfRefine`, `CritiqueResult`

### External
- `pytest`, `pytest-asyncio`

<!-- MANUAL: -->
