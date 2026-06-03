# src/serving/ — inference servers + adapter swap + Aeon hook

OpenAI-compat inference layer. Backend-agnostic FastAPI app talks to an `MLXRuntime`
Protocol; concrete backends are MLX (Mac Studio), vLLM (RTX 4090), and `FakeMLXRuntime`
(tests). Adapter activation flows from the router (`src/routing/`) into this tree —
never compute routing decisions here.

## Backend split (hard)

| Host | Backend | Module | Notes |
|---|---|---|---|
| Mac Studio M3 Ultra | MLX-LM (BF16 weights, Q4 KV optional) | `mlx_server.py`, `studio_lora_server.py` | Subprocess restart on adapter swap (~200 ms — mlx-lm limitation) |
| kxkm-ai RTX 4090 24 GB | vLLM (Q4_K_M) | `vllm_server.py` | Runtime LoRA via `/v1/{load,unload}_lora_adapter` (`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`) |
| Any (in-process) | PEFT | `switchable.py` | `apply_stacks` cache-keyed on sorted name tuple |
| CI / dev | Fake | `runtime.FakeMLXRuntime` | No MLX import, deterministic via sha256(seed) |
| ANE (stub) | CoreML | `ane_draft.py`, `ane_router.py`, `ane_scorer.py` | Not wired — see TODO; `ane_router` width sentinel `37` is pre-pivot |

## Module map

| File | Role |
|---|---|
| `runtime.py` | `MLXRuntime` Protocol + `FakeMLXRuntime`. App talks only to the Protocol. |
| `app.py` | FastAPI factory — `/health`, `/v1/models`, `/metrics`, `/v1/chat/completions`. Lifecycle owned by caller, not the app. |
| `schemas.py` | Pydantic request/response shapes (OpenAI-compat). |
| `mlx_server.py` / `studio_lora_server.py` | Studio-side processes (subprocess vs. in-process LoRA shells). |
| `vllm_server.py` | kxkm-ai vLLM subprocess wrapper. |
| `switchable.py` | PEFT in-process hot-swap, `MAX_ACTIVE_STACKS = 4`. |
| `moe_lora_runtime.py` | Legacy MoE-LoRA backend (kept; satisfies Protocol structurally). |
| `aeon_hook.py` | Pre/post inference memory hook — failures degrade to untouched prompt. |
| `kiki_flow_bridge.py` | Bridge to `kiki-flow-research` engine (numerical primitives). |
| `full_pipeline_server.py` | 7-stage cognitive pipeline (router → adapters → recall → infer → negotiator → anti-bias → write). Heavy imports lazy-bound for test mockability. |
| `model_aliases.py` | Model-id → adapter-name resolution for OpenAI-compat. |
| `mlx_unpatch.py` | Restore base layers after LoRA wrapping (LoRALinear/LoRASwitchLinear/DoRA). |

## Quantization rule

Inference: **Q4_K_M only** on RTX 4090 (24 GB ceiling). BF16 paths are Studio-only. Never
ship a Q8 or FP16 inference path on the 4090 — it does not fit, and the quality cliff
referenced in root CLAUDE.md is below Q4, not above. If a backend silently drops to a
different quant level, fail loudly.

## Coupling vs. routing

`src/routing/` decides **which adapters fire** (sigmoid, threshold, top-k ≤ 4).
`src/serving/` decides **which backend runs** (host capability) and **how to activate**
the adapter set (subprocess restart vs. PEFT swap vs. vLLM admin endpoint). Do not move
threshold logic, capability mapping, or sigmoid math into this tree — and do not route
through experimental `quantum_router` / `tn_router` from here.

## Hot-swap discipline

- Subprocess backends (MLX-LM, vLLM): cache the active adapter set; skip the swap if the
  sorted tuple is unchanged. Same rule as `switchable.SwitchableModel._cache_key`.
- The 4-active-stack ceiling is enforced **here**, in addition to routing — defence in
  depth, both layers must reject N > 4.
- Cold-start warm-up: first request after process start eats the load latency; expose it
  via `/metrics` (`mlx_request_seconds` histogram), do not hide it in middleware.

## Anti-patterns

- Calling MLX / vLLM / PEFT directly from `app.py` route handlers — go through the
  `MLXRuntime` Protocol, otherwise the FastAPI app stops being importable on CI.
- Hardcoding the quantization (e.g. `--dtype float16`) in a backend wrapper. Quant level
  is a deploy config, not source.
- Sharing mutable state across requests — `MLXServer`, `VLLMServer`, `SwitchableModel`
  are per-process singletons; per-request state lives in `runtime.RequestContext` or the
  FastAPI dependency layer.
- Silently swallowing an Aeon hook exception **and** dropping the turn — the hook must
  fall back to the untouched prompt and continue (see `aeon_hook.py`).
- Wiring an ANE stub (`ane_*.py`) into `HybridPipeline` before the `.mlpackage` artefact
  exists in `models/` — the `37` sentinel will mis-route.
