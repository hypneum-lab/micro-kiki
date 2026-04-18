<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# eval — Stack Evaluation, Forgetting Detection, Cognitive Benches

## Purpose
Gatekeeper layer between "stack trained" and "stack accepted". Runs the forgetting rollback check (angle < 30° AND win-rate drop > 0.03 → rollback), the adaptive LLM-judge stack evaluator, domain-specific GRPO reward functions, and retrospective MAP-paper benches for the v0.2 cognitive layer (Dispatcher + Negotiator). See `CLAUDE.md` in this directory for the adaptive-judge and bias-monitoring rules.

## Key Files
| File | Description |
|------|-------------|
| `forgetting.py` | Canonical forgetting framework. `GradientSubspaceAnalyzer.compute_angle()` uses QR + SVD to get the principal angle in degrees between base and adapted gradient matrices. `ForgettingEvaluator.check_stack()` combines the angle with win-rate delta against `ANGLE_THRESHOLD=30.0` / `WINRATE_DROP_THRESHOLD=0.03`. CLI: `python -m src.eval.forgetting <new_stack> --results-file <pre-computed.json>`. Emits `results/forgetting-<stack>.json`; exits 1 on any rollback. |
| `stack_eval.py` | `StackEvaluator` — async LLM-judge harness. Generates A/B (base vs stack) for each eval prompt, calls `judge_client.generate()` with the `JUDGE_PROMPT` template, parses JSON `{winner, score, reason}`, returns `win_rate_vs_base` + `avg_judge_score` + 5 sample winners. |
| `reward_functions.py` | GRPO reward components, each `(prompt, response, domain) -> [0,1]`: `syntax_valid` (KiCad/SPICE regex), `format_correct` (code blocks, step numbering, unit-aware regex `_COMPONENT_VALUE`), `completeness_reward` (length ramp 50/200/2000/5000 chars), `accuracy_reward` (HTTP call to 480B judge at `localhost:8481`, falls back to 0.5), `combined_reward` (default weights syntax 0.3, format 0.2, completeness 0.1, accuracy 0.4). |
| `map_harness.py` | MAP-paper (Nature Comms 2025) retrospective bench. `MAPCompatibleAgent` Protocol covers 5 modules (conflict_monitor, state_predictor, evaluator, decomposer, coordinator). Synthetic generators seeded deterministically; stdlib-only (no numpy). `MockAgent` provides a reasonable baseline; `MAPHarness.run_all()` emits JSON with schema `map-harness/1.0`. |
| `map_dispatcher_bench.py` | Story-2: `V02DispatcherAgent` models the v0.2 Dispatcher (32→7 YAML matrix) as a MAP conflict monitor via normalised Shannon entropy with a 4-stack activation cap and `chat_floor=0.20` ambiguity boost. Reports `agreement_delta_vs_mock`. |
| `map_negotiator_bench.py` | Story-3: `V02NegotiatorAgent` models CAMP arbitration + Catfish dissent (10% flip, deterministic seed 17) + adaptive judge. Tracks `escalation_rate`, `judge_cost_mean` (deep = 7× fast), `spearman_rho`. |
| `__init__.py` | Package marker (v0.3 neuroscience branch). |

## For AI Agents

### Working In This Directory
- **Run forgetting check after EVERY stack**, including retroactive baseline on stacks 02-03 (CLAUDE.md step 14). The `should_rollback` gate requires BOTH conditions — angle-only or wr-only failures do NOT trigger rollback (see `check_forgetting` comment in `forgetting.py`).
- **Adaptive judge discipline**: Qwen3.5-35B (kxkm-ai) first pass always. Escalate to Mistral-Large-Opus (Studio) only when confidence < 0.5. Project `Don't`: *don't use expensive judge when cheap judge is confident*. `map_negotiator_bench` costs deep at 7×.
- **Never eval with training data**: CLAUDE.md anti-pattern. Eval JSONL lives under `data/eval/<stack_id>.jsonl`.
- **Never compare stacks trained at different quantization levels**: CLAUDE.md anti-pattern. All training BF16, inference Q4_K_M — forgetting check must use matching precision both sides.
- **Bias monitoring runs on every response**: KnowBias double-application + RBD runtime detector. Not implemented here yet — these benches validate the v0.2 cognitive spec retrospectively; they do NOT replicate MAP results.
- **Judge accuracy reward fails open to 0.5** — do not raise on judge HTTP errors; GRPO loop must keep running.
- **MAP benches are synthetic** (design-validation signals, not paper replication). Do not infer empirical claims from them.

### Testing Requirements
- Mirror tests in `tests/eval/`. `test_forgetting.py` exercises both `ForgettingReport` (new) and `ForgettingCheckResult` (legacy compat) paths — keep both until all callers migrate.
- Reward-function tests in `tests/eval/test_reward_functions.py` pin the SPICE/KiCad regex and the length-ramp bounds.
- MAP benches ship JSON fixtures under `results/` — don't commit per-run outputs, only schema-versioned samples.

### Common Patterns
- `@dataclass(frozen=True)` for results (`ForgettingReport`, `ForgettingCheckResult`, `BenchResult`).
- Stdlib-only for MAP benches (no numpy) — keeps the cognitive-layer gate fast and dependency-free.
- Judge calls use `httpx.Client(timeout=30)` and extract score via regex first, then fall back to JSON parse.
- `SCHEMA_VERSION` strings on every emitted JSON blob (`map-harness/1.0`, `map-dispatcher-bench/1.0`, `map-negotiator-bench/1.0`).

## Dependencies

### Internal
- `src.eval.map_harness` is imported by both `map_dispatcher_bench` and `map_negotiator_bench`.
- Consumed by `src.ralph.forgetting_auto.ForgettingChecker` (automation wrapper) and scripts under `scripts/run_forgetting.sh`.
- `StackEvaluator` is called from `src.orchestrator` for end-to-end benches.

### External
- `torch` (forgetting only — via `TYPE_CHECKING` to avoid import cost).
- `httpx` (accuracy reward judge HTTP).
- 480B teacher at `http://localhost:8481/v1/chat/completions` (model `Qwen3-Coder-480B-A35B`).

<!-- MANUAL: -->
