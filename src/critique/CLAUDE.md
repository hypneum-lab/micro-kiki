# src/critique/ — three-level auto-critique (best-of-N, self-refine, agentic-loop)

Sits between inference and the user. Stateless, stdlib-only, no imports from
other `src/*` modules — callers (`src.routing.hybrid_pipeline`,
`src.ralph.autonomous`) inject the async `generate_fn`. Prompt schemas in
`templates.py` are the single source of truth. File-by-file index in `AGENTS.md`.

## Hard invariants

| Invariant | Where |
|---|---|
| `BestOfN(high=0.8, mid=0.5, mid_n=3, low_n=5)` thresholds — adaptive contract | `best_of_n.py` |
| `AgenticLoop.max_iterations = 5` is a **hard cap**; `completed=False` on miss | `agentic_loop.py` |
| All return types are `@dataclass(frozen=True)` — deterministic decision logs | every module |
| `templates.py` JSON schemas are contracts — parsers `json.loads` defensively but field renames break consumers | `self_refine.py`, `agentic_loop.py` |
| Every `generate_fn` is async (`Callable[[str], Awaitable[str]]`); best-of-N needs `Awaitable[tuple[str, float]]` (text + log-prob) | type aliases at module top |

## Coupling

| Direction | Contract |
|---|---|
| ← `src.routing.hybrid_pipeline` | Passes `quantum_confidence` (sigmoid of `MetaRouter`) as the `confidence` input to `BestOfN.select_n`. |
| ← `src.ralph.autonomous` | Wraps `AgenticLoop` inside its own `AutonomousLoop` (which has `max_consecutive_failures`); critique's hard cap is per-task, autonomous's is per-session — different axes. |
| → cognitive | None. Anti-bias / RBD / DeFrame run **after** the critique winner is picked, in `src.serving.full_pipeline_server`, not here. |

## Anti-patterns

- Removing `max_iterations=5` to "let the agent finish" — that turns this into the
  autonomous loop and bypasses the audit-trail cap.
- Adding instance state across `run()` calls — these classes are stateless on
  purpose; tests reuse one instance with mocked `generate_fn`.
- Catching `json.JSONDecodeError` and silently retrying — schema drift must crash
  loud so the prompt template can be fixed.
- Forking `SELF_REFINE_CRITIQUE` per call site — parametrise via the `bias_type`
  / domain field instead so all callers share one schema.
- Importing from `src.cognitive` or `src.memory` — critique is deliberately
  self-contained stdlib; coupling lives in the caller.
- Using `asyncio.gather` outside `BestOfN.generate_candidates` — self-refine and
  agentic-loop are inherently sequential (each step depends on the previous).
