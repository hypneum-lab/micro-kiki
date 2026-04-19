<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# critique

## Purpose
Three-level auto-critique family that sits between inference and the user: **best-of-N** (router-confidence-adaptive sampling, level 1), **self-refine** (single structured critique + correction pass, level 2), and **agentic-loop** (plan → execute → evaluate → retry with a hard iteration cap, level 3). All three drive a caller-supplied async `generate_fn`, parse structured JSON from the model, and return frozen dataclasses so callers can log decisions deterministically. Prompts live in `templates.py` and are the single source of truth for the expected JSON schemas.

## Key Files
| File | Description |
|------|-------------|
| `best_of_n.py` | `BestOfN(high_threshold=0.8, mid_threshold=0.5, mid_n=3, low_n=5)` — `select_n(confidence)` returns 1 / 3 / 5 samples based on router confidence. `generate_candidates` fans out via `asyncio.gather`, `select_best` picks max log-prob. Returns `ScoredCandidate(text, log_prob)`. |
| `self_refine.py` | `SelfRefine(generate_fn)` — single critique+correct round trip. `_get_critique` parses `SELF_REFINE_CRITIQUE` JSON into `CritiqueFeedback(factual_errors, missing_info, clarity_issues, confidence, needs_correction, summary)`. Skips correction when `needs_correction=False`. Returns `CritiqueResult`. |
| `agentic_loop.py` | `AgenticLoop(generate_fn, max_iterations=5)` — plan via `AGENTIC_PLAN` (JSON array of steps), execute each step, evaluate via `AGENTIC_EVALUATE`. Honours `next_action ∈ {proceed, retry, abort}`. Returns `LoopResult(completed, iterations, steps, final_output)`. |
| `templates.py` | `SELF_REFINE_CRITIQUE`, `SELF_REFINE_CORRECTION`, `AGENTIC_PLAN`, `AGENTIC_EVALUATE` prompt templates with exact JSON schemas. Changing a field here is a breaking change for the parsers above. |

## For AI Agents

### Working In This Directory
- **Schemas are contracts**: `SelfRefine` and `AgenticLoop` call `json.loads` with no try/except on the outer path. Changing the JSON shape in `templates.py` without updating the consumer will crash. Parse defensively when adding fields (use `data.get` with defaults, as already done).
- **BestOfN thresholds** (0.8 high, 0.5 mid) are the adaptive contract — `select_n=1` when the router is confident, `select_n=5` in the low-confidence regime. The "router confidence" comes from `HybridPipeline.quantum_confidence` / `MetaRouter` sigmoid output.
- **`agentic_loop.max_iterations=5`** is a hard cap; `AgenticLoop.run` returns `completed=False` when the cap hits. Don't turn this into an unbounded loop — that's `src.ralph.autonomous.AutonomousLoop`'s job, and even it has `max_consecutive_failures`.
- **All `generate_fn` are async** — `Callable[[str], Awaitable[str]]` for self-refine / agentic; `Callable[[str], Awaitable[tuple[str, float]]]` (text + log-prob) for best-of-N.
- **No internal state**: these classes are stateless between `run()` calls — safe to reuse across threads as long as the generate_fn is.

### Testing Requirements
- `tests/critique/test_best_of_n.py` — confidence → n mapping, concurrent sampling, best-selection.
- `tests/critique/test_self_refine.py` — needs-correction=false skip, critique parse, correction prompt structure.
- `tests/critique/test_agentic_loop.py` — plan/execute/evaluate mock chain, proceed/retry/abort branching, iteration cap.

### Common Patterns
- `@dataclass(frozen=True)` for every return type (`ScoredCandidate`, `CritiqueFeedback`, `CritiqueResult`, `StepResult`, `LoopResult`).
- `from __future__ import annotations` everywhere.
- `asyncio.gather` for parallel sampling (best-of-N only — self-refine and agentic are inherently sequential).
- `Callable[..., Awaitable[...]]` type aliases at module top — read these first when modifying signatures.
- `logging.getLogger(__name__)` for info on corrections and retries; no `print()`.

## Dependencies

### Internal
- No imports from other `src/*` modules — the critique layer is deliberately self-contained. Callers (`src.routing.hybrid_pipeline`, `src.ralph.autonomous`) inject their own `generate_fn`.

### External
- stdlib only: `asyncio`, `json`, `logging`, `dataclasses`, `typing.Callable`, `typing.Awaitable`.

<!-- MANUAL: -->
