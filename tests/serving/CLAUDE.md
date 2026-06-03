# tests/serving/ — serving stack tests (sub-suite)

21 files covering `src/serving/` — backend wrappers, FastAPI app, full 7-stage pipeline,
adapter swap. Sub-suite of `tests/`; the parent `tests/CLAUDE.md` rules (stubs over real
models, `integration` / `gpu` markers, AAA, parametrize stack lists) all apply here.

## What this sub-suite specifically guards

| File group | Target invariant |
|---|---|
| `test_app.py`, `test_runtime.py`, `test_schemas.py`, `test_model_aliases.py` | OpenAI-compat surface — `/health`, `/v1/models`, `/v1/chat/completions`, `Metrics`, `FakeMLXRuntime` determinism (sha256(seed)). |
| `test_mlx_client.py`, `test_studio_lora_server.py`, `test_mlx_unpatch.py` | MLX-side adapter swap, `unpatch_all_lora` round-trip, Studio HTTP shim. |
| `test_aeon_hook.py` | `[Memory]` prefix format, post-inference write, fail-open on exception (must not drop the turn). |
| `test_full_pipeline_*` (10 files) | 7-stage skeleton + per-stage tests (parse, queue metrics, recall, adapters, inference, negotiator, response, swap-metric, thinking). Heavy components monkeypatched at import time. |
| `test_integration.py`, `integration/` | Marked `@pytest.mark.integration`, default-skipped. |

## Backend mocking discipline (load-bearing)

- **Never** import MLX or vLLM at module top-level here. The full-pipeline tests
  monkeypatch `_build_*` factories before any heavy import. `studio_lora_server` tests
  patch the `mx` import; `mlx_unpatch` uses `pytest.importorskip("mlx.core")` — it is
  the **only** file in this sub-suite allowed to require MLX, and it stays unmarked
  (skips cleanly on Linux CI).
- vLLM tests live in `tests/test_vllm_server.py` (parent dir) and rely on `subprocess.Popen`
  monkeypatch. Don't move them here without preserving the no-real-vLLM contract.
- Adapter swap tests assert the cache-key tuple skips no-op switches — both at the
  `SwitchableModel` layer and in the subprocess wrappers. Defence-in-depth on the
  4-active-stack ceiling is asserted at both layers.

## Quantization assertions

- A serving test that exercises a Q4_K_M code path must assert the quant level on the
  request payload / process-args — silent drop to FP16 on RTX 4090 will OOM in
  production but pass in mocks if not asserted.
- Cross-quant comparisons (Q4 vs BF16 outputs) belong in `tests/eval/` or as integration
  benches — not here. Parent rule: "don't compare stacks across quantization levels".

## Anti-patterns

- Loading a real Qwen3.6-35B-A3B (or even a real `mlx_lm.server` subprocess) here —
  parent forbids real-model loads in unit tests; this sub-suite's full-pipeline tests
  break loudly if you do (CI timeout + 35 GB download).
- Session-scoped fixtures over a stateful backend (`MLXServer`, `VLLMServer`,
  `SwitchableModel`) — they hold subprocess / adapter cache; use function-scoped
  fixtures and reset between tests.
- Sharing a `FakeMLXRuntime` instance across tests — the metrics counters and seed
  hash chain leak across cases.
- Asserting on private attributes (`_active_stacks`, `_cache_key`) instead of the public
  contract (`apply_stacks`, `list_available`) — parent rule, sharper here because the
  swap cache is implementation-detail.
- Forgetting the `aeon_hook` fail-open path — a test that asserts the hook **must**
  succeed will silently invalidate the production invariant that memory failures degrade
  to the untouched prompt.
