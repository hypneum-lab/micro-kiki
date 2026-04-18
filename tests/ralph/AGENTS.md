<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# tests/ralph

## Purpose
Tests the Ralph autonomous implementation loop — the self-driving orchestrator that takes a story, researches it, implements code, runs multi-pass self-review, tests, runs the forgetting-check gate (for training stories), and commits. This suite locks in the review retry loop, the forgetting-rollback blocking semantics (angle < 30° AND win-rate drop > 0.03), and the hard-stop after consecutive failures.

## Key Files
| File | Description |
|------|-------------|
| `test_autonomous.py` | End-to-end tests for `AutonomousLoop.run_story()` and `AutonomousLoop.run()`: no-review bypass, first-pass approval, retry-until-approved (3 passes), `max_review_passes` cap, non-training stories skip forgetting, training stories trigger forgetting with stack-id extraction (`stack-07`), forgetting-rollback blocks commit, and hard-stop after 3 consecutive test failures. |
| `test_forgetting_auto.py` | `TestForgettingChecker`: pass when angle ≥ 30 and win-rate stable, rollback when angle < 30 AND win-rate drop > 0.03, pass when angle low but win-rate stable, and eval JSON persistence via `save_result`. |
| `test_research.py` | `TestStoryResearcher.test_research_produces_markdown` — verifies Exa + Scholar mock results are rendered into a markdown file at `output_dir`; `test_extracts_keywords` checks keyword extraction from story title + description. |
| `test_self_review.py` | `TestCodeReview`: template has bugs/edge_cases/perf/security/style sections, `parse_review` converts JSON to `ReviewResult` with `total_issues` count, and `max_passes` is configurable. |

## For AI Agents

### Working In This Directory
Run the suite: `uv run python -m pytest tests/ralph/ -q`.
- All async tests in `test_autonomous.py` run under `asyncio_mode = "auto"` (bare `async def` — no decorator needed).
- No `@pytest.mark.integration` tests; every dependency is mocked with `AsyncMock` / `MagicMock`.
- `_build_loop` helper in `test_autonomous.py` constructs `AutonomousLoop` with an `evals_dir` inside `tmp_path` and optional `review_fn` / `eval_fn` side_effects lists.
- `eval_fn` signature: `(stack_id: str) -> (angle_degrees, winrate_base, winrate_adapted)`.

### Testing Requirements
- Review-loop invariant: `review_passes` equals the number of times `review_fn` was awaited; implementation is re-run on each non-approved pass.
- Forgetting gate: a story whose title matches `train stack-NN` triggers the gate; rollback iff angle < `angle_threshold` (30°) AND `winrate_base - winrate_adapted` > `winrate_drop_threshold` (0.03). When rollback fires, `outcome.success == False`, `outcome.error == "forgetting rollback triggered"`, and `commit_fn` is NOT awaited.
- Non-training stories: `forgetting_check is None`, `forgetting_angle is None`.
- Hard-stop: `AutonomousLoop.run()` stops after 3 consecutive failed outcomes (e.g. only 3 outcomes returned from 5 input stories).
- `ForgettingChecker.save_result` must persist a JSON file with `stack_id` and `passed` fields.

### Common Patterns
- `AsyncMock(side_effect=[...])` for sequencing multi-pass review responses.
- `AsyncMock(return_value=(angle, winrate_base, winrate_adapted))` for eval mocks.
- `tmp_path` fixture + ephemeral `evals_dir` — no filesystem pollution.
- JSON helper functions `_approved_review()` / `_rejecting_review()` to avoid repeating payload dicts.

## Dependencies

### Internal
- `src.ralph.autonomous` — `AutonomousLoop`, `LoopConfig`
- `src.ralph.forgetting_auto` — `ForgettingChecker`, `ForgettingResult`
- `src.ralph.self_review` — `CodeReview`, `ReviewResult`
- `src.ralph.research` — `StoryResearcher`
- `src.search.base` — `SearchResult` (fixture data)

### External
- `pytest`, `pytest-asyncio`

<!-- MANUAL: -->
