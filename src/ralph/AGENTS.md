<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# ralph

## Purpose
Autonomous story-driven development loop: for every story in the queue, **research** the topic (Exa + Semantic Scholar) → **implement** via an injected codegen callable → **self-review** via structured JSON critique → **test** → **forgetting-check** (only for stack-training stories) → **commit**. Sits one layer above `src.critique.*` (which handles per-response refinement) and is what actually drives the sequential per-domain stack training described in the project CLAUDE.md ("Train stacks sequentially, curriculum order; run forgetting check after EACH stack; rollback if angle < 30° AND win-rate drop > 0.03").

## Key Files
| File | Description |
|------|-------------|
| `autonomous.py` | `AutonomousLoop(researcher, code_review, forgetting_checker, implement_fn, test_fn, commit_fn, review_fn, eval_fn)` — `run_story(story)` orchestrates research → implement → review-retry → test → forgetting → commit. Detects stack-training stories via `_is_training_story` (looks for "train stack" in title/description) and extracts stack id via `_STACK_RE = r"stack-(\d+)"`. Returns `StoryOutcome`. `run(stories)` iterates and hard-stops after `max_consecutive_failures` (default 3). |
| `research.py` | `StoryResearcher(exa_backend, scholar_backend, output_dir=".ralph/research")` — `research_story(story)` extracts ≤10 keywords (arxiv ids stripped), runs 5 Exa + 5 Scholar, writes a markdown dossier at `.ralph/research/<story_id>.md`. Returns the dossier path. |
| `self_review.py` | `CodeReview(max_passes=3)` — structured JSON critique with categories bugs / edge_cases / perf / security / style + approved + summary. `parse_review(raw)` → `ReviewResult`. `total_issues` property sums all five categories. Prompt template `REVIEW_TEMPLATE`. |
| `forgetting_auto.py` | `ForgettingChecker(eval_fn, evals_dir=".ralph/evals", angle_threshold=30.0, winrate_drop_threshold=0.03)` — `evaluate(angle, winrate_base, winrate_adapted) → ForgettingResult`. Rollback requires **both** `angle < 30°` AND `winrate_drop > 0.03` (AND, not OR). `save_result(stack_id, result)` writes `{stack_id}.json` to evals_dir. |

## For AI Agents

### Working In This Directory
- **The forgetting rollback rule is AND, not OR**: `should_rollback = (angle < 30°) AND (winrate_drop > 0.03)`. This matches the project CLAUDE.md verbatim — do not change to OR or relax either threshold without an explicit spec change.
- **Only stack-training stories run the forgetting check**: `_is_training_story` matches the literal substring `"train stack"` in title or description. Non-training stories skip eval entirely.
- **Review-retry caps at 3 passes** (`LoopConfig.max_review_passes`). After 3, loop proceeds with the last code and logs a warning. Don't turn this into an unbounded retry — that conflicts with `max_consecutive_failures` at the outer loop.
- **Hard-stop on 3 consecutive failures**: `AutonomousLoop.run` breaks out of the story queue when `_consecutive_failures >= max_consecutive_failures`. A successful story resets the counter.
- **`dry_run=True`** skips `commit_fn` but still runs research/implement/review/test/forgetting. Use this when iterating on the loop itself.
- **Async everywhere**: `implement_fn`, `test_fn`, `commit_fn`, `review_fn`, `eval_fn` are all `Awaitable`. The researcher backends (Exa/Scholar) are async by contract (see `src/search/AGENTS.md`).
- **`.ralph/` is the artefact directory**: research dossiers in `.ralph/research/`, forgetting evals in `.ralph/evals/`, progress in `.ralph/progress.txt` (path defined in `LoopConfig`).
- **eval_fn returns `(angle, winrate_base, winrate_adapted)`** — a 3-tuple of floats, not a struct. Keep the tuple shape stable; `AutonomousLoop.run_story` unpacks positionally.

### Testing Requirements
- `tests/ralph/test_autonomous.py` — end-to-end loop on a small story list with mocked callables, failure counter, dry-run branch.
- `tests/ralph/test_research.py` — keyword extraction, markdown dossier shape.
- `tests/ralph/test_self_review.py` — review template, JSON parse, total_issues arithmetic.
- `tests/ralph/test_forgetting_auto.py` — AND-rule rollback, `save_result` JSON shape.

### Common Patterns
- `@dataclass(frozen=True)` for configs and results that should not mutate (`LoopConfig`, `ReviewResult`, `ForgettingResult`).
- `@dataclass` (mutable) for `StoryOutcome` because it is aggregated post-hoc.
- `Callable[..., Awaitable[...]]` injection for every side-effecting function — loop never imports the serving or LLM layer directly.
- Markdown artefacts written with a joined `lines` list + `write_text` (research.py pattern).
- Regex compiled at module level (`_STACK_RE`).
- `logging.getLogger(__name__)` — no `print()`.

## Dependencies

### Internal
- `src.ralph.research.StoryResearcher` depends on `src.search.base.SearchBackend` (uses Exa + Scholar concrete backends).
- `AutonomousLoop` composes `StoryResearcher`, `CodeReview`, `ForgettingChecker` via constructor injection.
- Typically driven by `scripts/run_pipeline.sh` / `scripts/run_eval_*.py` at the top level; `implement_fn` is usually a wrapper around `src.serving.*` generation.

### External
- stdlib only: `asyncio`, `json`, `logging`, `re`, `dataclasses`, `pathlib`, `datetime`, `typing.Callable`, `typing.Awaitable`.

<!-- MANUAL: -->
