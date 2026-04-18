# tests/ — pytest suite

`uv run python -m pytest`. Mirrors `src/` layout (`tests/routing/`, `tests/stacks/`, …). Fixtures in `conftest.py` at each level.

## What this suite guards

- **Router math**: thresholds, top-k truncation, sigmoid-not-softmax, dispatcher YAML → meta-intent mapping.
- **Forgetting gate**: angle math, rollback decision, baseline capture.
- **Dataset hygiene**: dedup correctness, loader determinism, domain-classifier outputs.
- **Config validation**: every stack YAML parses + targets only attention projections.
- **Bias pipeline**: KnowBias pre/post deltas, RBD detector trigger cases.
- **Serving glue**: MLX server, vLLM server, `switchable.py`, aeon hook.
- **Cognitive layer**: negotiator (CAMP + Catfish), consolidation, sleep tagger.

## Stubbing policy (hard rule)

- **Never load a real 35B model in unit tests.** Use tiny stubs (hidden_size ≤ 64) or mock the HF call.
- Real-model tests go behind `@pytest.mark.integration` — skipped by default (see `pyproject.toml` marker).
- MLX ops on CI hosts without Metal: stub `mlx.core` at import time.
- GPU-dependent paths: tag `@pytest.mark.gpu` and skip unless `--run-gpu`.

## Marker conventions

- `integration` — needs real models, SSH to kxkm-ai/Studio, or large data. Default-skipped.
- `gpu` — needs CUDA/Metal. Default-skipped.
- Unmarked tests must run in < 5 s each on a laptop with no GPU.

## Conventions

- AAA (Arrange, Act, Assert); one behavior per test.
- Session-scoped fixtures for any model/tokenizer load (even stubs).
- Parametrize stack lists (don't duplicate the 35 domains in each test).

## Anti-patterns (tests-specific)

- Don't load real models in unit tests — use stubs or mocks.
- Don't share mutable state between tests (no module-level caches; reset `Aeon` backends per test).
- Don't test private/implementation details — assert on public contract.
- Don't write tests that still pass when the code is broken (assertions must fail first, then green).
- Don't call out to kxkm-ai, Studio, HF Hub, or the filesystem `~/KIKI-Mac_tunner/` tree without the `integration` marker.
- Don't put eval quality checks here — those live in `src/eval/` and run on real artifacts.
