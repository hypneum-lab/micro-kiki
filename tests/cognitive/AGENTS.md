<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# tests/cognitive

## Purpose
Tests Phase IX (CAMP Negotiator) and Phase X (Anti-bias / KnowBias + RBD) cognitive modules: argument extraction from candidate responses, the adaptive judge with skip / fast / deep backends, the Catfish dissent trigger, the `ReasoningBiasDetector`, and the `AntiBiasOrchestrator` rewrite loop. All LLM calls are mocked.

## Key Files
| File | Description |
|------|-------------|
| `test_negotiator.py` | Five test classes — `TestArgumentExtractor` (heuristic extract without `generate_fn`, LLM-extract with quality_score > 0.5), `TestAdaptiveJudge` (skip backend on ≥0.95 agreement, fast backend on mid agreement), `TestCatfish` (`should_trigger(0.98, 0.4) == True` but not at 0.80 or when quality 0.8), `TestRBD` (clean vs. biased detection), `TestAntiBias` (`check_and_fix` no-rewrite for clean, rewrite path when `biased=true` — second `generate_fn` call returns fair text). |

## For AI Agents

### Working In This Directory
Run the suite: `uv run python -m pytest tests/cognitive/ -q`.
- All tests are `@pytest.mark.asyncio`.
- No `@pytest.mark.integration` tests — mocks use local `async def mock_gen(prompt)` closures returning JSON strings and `AsyncMock` for backend clients.
- `AdaptiveJudge` is constructed with explicit `fast_client=AsyncMock()` whose `generate` returns a JSON verdict.

### Testing Requirements
- `ArgumentExtractor`: must fall back to a heuristic extraction (non-empty `claim`) when no `generate_fn` provided; with `generate_fn`, must parse `{claim, evidence, reasoning}` and compute `quality_score > 0.5`.
- `AdaptiveJudge.backend_used` must be `"skip"` when agreement ≥ 0.95, `"fast"` in the mid range.
- Catfish `should_trigger(agreement, quality)` truth table: `(0.98, 0.4) → True`, `(0.80, 0.4) → False`, `(0.98, 0.8) → False`. When not triggered, `maybe_dissent(...).triggered is False`.
- `ReasoningBiasDetector.detect` must parse `{biased, bias_type, explanation, confidence}` JSON; `AntiBiasOrchestrator.check_and_fix` must only invoke `generate_fn` a second time (rewrite) when `biased=True`.

### Common Patterns
- Nested closures with `nonlocal call_count` to drive multi-turn rewrite flow.
- `AsyncMock.generate.return_value = json.dumps({...})` for fast-judge backend.
- Dataclass construction in arguments (`Argument("claim", "evidence", "reasoning", 0.8)`) — positional, be careful on signature drift.

## Dependencies

### Internal
- `src.cognitive.argument_extractor` — `ArgumentExtractor`, `Argument`
- `src.cognitive.judge` — `AdaptiveJudge`, `JudgeResult`
- `src.cognitive.catfish` — `CatfishModule`
- `src.cognitive.rbd` — `ReasoningBiasDetector`, `BiasDetection`
- `src.cognitive.antibias` — `AntiBiasOrchestrator`

### External
- `pytest`, `pytest-asyncio`

<!-- MANUAL: -->
