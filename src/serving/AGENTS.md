<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# serving

## Purpose
Inference-server adapters that expose Qwen3.5-35B-A3B (or the legacy 4B base) with LoRA hot-swapping, plus the glue that injects Aeon memory into prompts and persists turns afterwards. Two primary backends are supported: `MLXServer` (Mac Studio M3 Ultra — primary training / serving target) and `VLLMServer` (kxkm-ai RTX 4090, Q4 inference). A `SwitchableModel` PEFT runtime plus three CoreML/ANE stubs (draft model, scorer, router) complete the layer.

## Key Files
| File | Description |
|------|-------------|
| `mlx_server.py` | `MLXServer` — manages `python -m mlx_lm.server --adapter-path <stack>` subprocess. Adapter switch is **not** hot-swap: `switch_adapter` stops and restarts the subprocess (~200ms penalty). `MLXServerConfig` loadable from JSON (`model_path`, `adapter_dir`, `port=8200`, `max_active_adapters=4`). httpx client for `/health`, `/v1/chat/completions`. |
| `vllm_server.py` | `VLLMServer` — spawns `vllm.entrypoints.openai.api_server --enable-lora --max-loras 4 --max-lora-rank 16`. Runtime adapter CRUD via `/v1/load_lora_adapter` and `/v1/unload_lora_adapter` (requires `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` env var). `VLLMServerConfig.model_path` still points at legacy `models/qwen3.5-4b-diffattn/` — update when 35B vLLM path stabilises. |
| `switchable.py` | `SwitchableModel` — PEFT-based in-process hot-swap runtime. `apply_stacks(names)` caches a merged key tuple to skip no-op switches. Hard cap `MAX_ACTIVE_STACKS = 4`. Uses `peft.PeftModel.load_adapter` / `set_adapter`. |
| `aeon_hook.py` | `AeonServingHook` — `pre_inference(prompt, top_k=8)` prepends `[Memory] <content>` lines from `AeonPalace.recall`; `post_inference(prompt, response, domain, turn_id)` writes `Q: ... A: ...` back into the palace. All failures log and degrade to the unaugmented prompt. |
| `ane_draft.py` | `ANEDraftModel` — Qwen3.5-0.8B on Apple Neural Engine via `coremltools.models.MLModel` for speculative decoding. Stub — `predict_next_tokens` returns `[]` pending CoreML export. |
| `ane_router.py` | `ANERouter` — meta-router compiled to CoreML running on ANE (not CPU). Stub — `route` returns `[0.0] * 37` pending export. TODO: reconcile `37` vs the current 35-way `MetaRouter` head when wiring the real CoreML export. |
| `ane_scorer.py` | `ANEScorer` — CoreML reward model on ANE for GRPO/RL scoring. Stub. |

## For AI Agents

### Working In This Directory
- **Mac vs GPU split**: `mlx_server.py` runs on Mac Studio only; `vllm_server.py` runs on kxkm-ai RTX 4090 only. The project rule "Don't train on kxkm-ai (model too large for 24 GB BF16 LoRA)" means kxkm-ai is inference-only via Q4 + vLLM. Keep this split explicit.
- **No MLX hot-swap**: the restart-on-switch behaviour in `MLXServer.switch_adapter` is a documented mlx-lm limitation, not a bug. Do not attempt to patch around it without a confirmed upstream fix.
- **Max 4 active adapters** in both backends (`max_loras=4` vLLM, `max_active_adapters=4` MLX, `MAX_ACTIVE_STACKS=4` SwitchableModel). This matches the project-wide "Don't route > 4 stacks simultaneously" rule.
- **LoRA rank bound**: `VLLMServerConfig.max_lora_rank=16` covers niches (rank 4-16). Foundations (rank 32) will need a config bump when deployed.
- **ANE stubs**: `ane_draft.py`, `ane_router.py`, `ane_scorer.py` are placeholders. Don't wire them into `HybridPipeline` until the `.mlpackage` exports exist in `models/`. The `[0.0] * 37` sentinel in `ane_router.py` is pre-pivot (should be 11 after the 2026-04-16 pivot).
- **AeonServingHook failures must not break inference**: the hook swallows all `Exception` paths and falls back to the untouched prompt. Preserve that invariant — a flaky memory layer should never drop turns.
- **Q4_K_M for inference, BF16 for training** (CLAUDE.md). Don't let any serving code path silently drop below Q4.

### Testing Requirements
- `tests/test_mlx_server.py` — subprocess lifecycle, adapter switch side-effects, health/generate HTTP with mocked httpx.
- `tests/test_vllm_server.py` — start/stop, load/unload adapter, health, generate payload shape.
- `tests/test_switchable.py` — PEFT adapter caching, max-stacks enforcement.
- `tests/test_aeon_hook.py` — `[Memory]` prefix format, post_inference payload, failure paths.

### Common Patterns
- `@dataclass(frozen=True)` for configs (`MLXServerConfig`, `VLLMServerConfig`); `@dataclass` (mutable) for server classes that track subprocess state.
- `subprocess.Popen` + `terminate()` with `wait(timeout=10)` fallback to `kill()` — consistent across both servers.
- `httpx.Client` / `httpx.post` with explicit `timeout=` per call (5s health, 30s admin, 30-120s generate).
- `from __future__ import annotations` everywhere.
- Optional `coremltools` import guarded inside `load()` so CI on non-Mac boxes still imports the module.

## Dependencies

### Internal
- `src.memory.aeon.AeonPalace` consumed by `AeonServingHook` (pre/post-inference memory).
- Consumed by `src.routing.hybrid_pipeline` (via `AeonServingHook` injection) and by scripts under `scripts/run_eval_*.py`.

### External
- `httpx` (both servers, ANE stubs when wired).
- `coremltools` (ANE modules, optional — import inside `load()`).
- `peft.PeftModel` (`SwitchableModel` only).
- `mlx_lm` (external subprocess — not imported).
- `vllm` (external subprocess — not imported).
- stdlib: `subprocess`, `json`, `logging`, `pathlib`, `dataclasses`, `os`, `time`.

<!-- MANUAL: -->
