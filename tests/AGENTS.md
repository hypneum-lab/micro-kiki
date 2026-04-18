<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# tests

## Purpose
pytest suite for micro-kiki. Directory layout mirrors `src/` (`tests/routing/`, `tests/cognitive/`, `tests/memory/`, etc.). The root-level `test_*.py` files cover cross-cutting concerns: E2E smoke flows, Aeon hook/compress/sleep, antibias pipeline, forgetting framework, MAP harness, dataset scraping/filtering, vLLM + MLX server integration, DiffAttention, spiking formers, and LAS/QTHA/OPLoRA pieces. A single `test_uart_ring.c` is kept alongside for the embedded-C helper shipped from `src/`.

## Key Files
| File | Description |
|------|-------------|
| `conftest.py` | Shared fixtures: `tmp_model_dir` (fake qwen3.5-4b tree), `sample_prompts` (10 cross-domain), `mock_teacher` (sync+async `FakeTeacher` with `call_log`) |
| `CLAUDE.md` | Testing conventions (AAA, no real models in unit tests, `@pytest.mark.gpu`) |
| `test_e2e_smoke.py`, `test_e2e_neuro.py`, `test_e2e_v02_acceptance.py`, `test_e2e_3stacks.py` | End-to-end flows across router + stacks + cognitive + memory |
| `test_forgetting.py`, `test_forgetting_gate.py` | Angle math + rollback decision |
| `test_antibias_pipeline.py`, `test_aeonsleep.py`, `test_aeon_hook.py`, `test_aeon_compress.py`, `test_atlas.py`, `test_consolidation.py`, `test_trace.py` | Cognitive + memory subsystems |
| `test_map_harness.py`, `test_map_dispatcher_bench.py`, `test_map_negotiator_bench.py` | MAP (router) bench harness |
| `test_convert_spikingkiki.py`, `test_diff_attention.py`, `test_spikingformer.py`, `test_lif_neuron.py`, `test_las_*.py`, `test_qtha.py` | v0.3 neuroscience branch tests |
| `test_mlx_server.py`, `test_vllm_server.py`, `test_integration_http.py`, `test_integration_phase14.py` | Serving + HTTP surface |
| `test_generator.py`, `test_dedup.py`, `test_loader.py`, `test_download.py`, `test_quality_filter.py`, `test_scrape_stackexchange.py`, `test_generate_codex.py` | Distillation + dataset pipeline |
| `test_validate_domains.py`, `test_validate_rank_schema.py`, `test_validate_curriculum_order.py`, `test_validate_no_pre_pivot.py` | CI-gate validators mirroring `scripts/validate_*.py` (domain-list drift, rank/alpha schema, curriculum order, pre-pivot identifier sweep) |
| `test_reward_functions.py` | DPO/GRPO reward shaping |
| `test_teacher_client.py` | Qwen3-Coder-480B teacher client |
| `test_dispatcher.py`, `test_stack_eval.py`, `test_moe_lora.py`, `test_oplora.py`, `test_trainer.py`, `test_switchable.py`, `test_phase_*.py`, `test_sleep_tagger.py`, `test_hierarchical_timer.py`, `test_smoke.py` | Unit-level coverage |
| `test_uart_ring.c` | Embedded C test paired with `src/uart_ring.h` |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `cognitive/` | `test_negotiator.py` (CAMP + Catfish arbitration) |
| `memory/` | `test_aeon.py` |
| `routing/` | `test_hybrid_pipeline.py`, `test_model_router.py`, `test_quantum_router.py`, `test_router_11.py`, `test_router_37.py` (37 = 32 domains + 5 capabilities) |
| `orchestrator/`, `critique/`, `compress/`, `search/`, `ralph/` | Per-subsystem tests (see their `src/` counterparts) |
| `scripts/` | Tests for standalone script CLIs: `test_measure_forgetting.py` (OPLoRA angle CLI), `test_eval_aeon_predictor.py`, `test_train_embeddings.py` |

## For AI Agents

### Working In This Directory
- Mirror `src/` layout when adding tests (a new `src/foo/bar.py` gets `tests/foo/test_bar.py`).
- AAA structure; one behavior per test; session-scoped fixtures for anything expensive.
- NEVER load the real Qwen3.5-35B-A3B base in a unit test. Use `tmp_model_dir` for structure, `mock_teacher` for distillation, tiny stubs for anything else.
- Integration tests requiring real models or SSH must be gated with `@pytest.mark.integration` (skipped by default — see `pyproject.toml`).
- Tests asserting GPU behavior use `@pytest.mark.gpu`; CI mocks these.
- `asyncio_mode = "auto"` is on; `async def test_...` just works.
- Don't share mutable state between tests; don't test private methods.

### Testing Requirements
```bash
uv run python -m pytest                         # default — skips integration
uv run python -m pytest tests/cognitive         # subset
uv run python -m pytest -m integration          # opt-in heavy
uv run python -m pytest -k forgetting -x        # fast fail
```

### Common Patterns
- `FakeTeacher` exposes `complete` (sync), `generate` (async), `generate_sync`, and `call_log` for assertions.
- `sample_prompts` spans French chat, code, math, embedded, reasoning — enough breadth to exercise the router.
- `tmp_model_dir` writes a `config.json` stub with `model_type=qwen3` — matches loader expectations without real weights.

## Dependencies

### Internal
Imports from `src/*` throughout. Uses `scripts/` for scrape/generate tests.

### External
`pytest>=8.0`, `pytest-asyncio>=0.24`, `httpx>=0.28.1` (dev/test extras). No GPU libraries required in CI.

<!-- MANUAL: -->
