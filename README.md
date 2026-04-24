# micro-kiki

**35 domain-expert LoRA adapters + cognitive layer on Qwen3.6-35B-A3B (native MoE, 256 experts, 3B active).**

Deployable artefact of the *dreamOfkiki* research program, part of
[Hypneum Lab](https://github.com/hypneum-lab). Author: Clément Saillant
(L'Electron Rare).

**Status: 35/35 V4 adapters trained and verified healthy.** Router V4 live (top-1 70.7%, top-3 89.5%). Full 7-stage cognitive pipeline deployed (`full_pipeline_server.py`). Stress test: 10/10 queries pass, 8/8 after restart, all domains routed correctly. Post-pivot adapters (35/35) pass the adapter-health validator; pre-pivot MoE-LoRA adapters (stacks-v3-r16) were archived as dead weights after an `lora_B = 0` audit — see `docs/research/2026-04-19-prepivot-moe-lora-audit.md`.

Sequential per-domain training via MLX on Mac Studio M3 Ultra 512 GB. Q4_K_M inference on Mac Studio (MLX) or kxkm-ai (RTX 4090 24 GB). Router is 31 sigmoid outputs (35 domains merged to 31 — electronics-hw, kicad, spice consolidated). Metal OOM on M3 Ultra is prevented by raising the MLX cache limit: `mx.set_memory_limit(460 * 1024**3)` + `mx.set_cache_limit(32 * 1024**3)` before training (see `CLAUDE.md` hard invariants). Default `set_cache_limit` is too small and triggers GPU Hang on long runs.

> **Training, datasets, and the `mlx-lm` fork live in the sibling repo [`KIKI-Mac_tunner`](https://github.com/L-electron-Rare/KIKI-Mac_tunner).** This repo holds the runtime: routing, cognitive layer, serving, eval, and the per-domain configs that drive the tuner.

---

## Architecture — 7-Stage Cognitive Pipeline

```
Domain query
    │
    ▼
┌─ Stage 1 ─┐
│  Router V4 │  mpnet 768d → MLP(768→512→31) → sigmoid
│  top-1 70.7%, top-3 89.5%          ≤ 4 active adapters
└────────────┘
    │
    ▼
┌─ Stage 2 ─┐
│ LoRA Swap  │  O(10ms) unpatch — not reload
└────────────┘
    │
    ▼
┌─ Stage 3 ─┐
│ Aeon Recall│  Atlas (SIMD vector) + Trace (neuro-symbolic graph)
│            │  ~62ms latency
└────────────┘
    │
    ▼
┌─ Stage 4 ─┐
│MLX Infer   │  Qwen3.6-35B Q4 + KV cache 8-bit + strip_thinking
│            │  3.6-5.2s per query
└────────────┘
    │
    ▼
┌─ Stage 5 ─┐
│ Negotiator │  CAMP arbitration + Catfish dissent
└────────────┘
    │
    ▼
┌─ Stage 6 ─┐
│ Anti-bias  │  KnowBias double-application + RBD + DeFrame
└────────────┘
    │
    ▼
┌─ Stage 7 ─┐
│ Aeon Write │  Persist episode (~32ms)
└────────────┘
    │
    ▼
Response
```

## Pipeline endpoints

The full pipeline is served by `full_pipeline_server.py` (FastAPI, OpenAI-compatible):

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | Full 7-stage cognitive pipeline |
| `POST /v1/route` | Standalone domain classification (router only) |
| `GET /v1/models` | 42 model aliases (base + 35 adapters + pipeline variants) |
| `GET /health` | Health check (adapter status, memory, uptime) |
| `GET /metrics` | Prometheus-format observability (latencies, routing decisions, cache hits) |

## Hard invariants (load-bearing across the whole project)

- **Base** — `Qwen/Qwen3.6-35B-A3B` (Apache 2.0, 262 K context). (Earlier drafts referenced Qwen3.5; superseded 2026-04-18 per real `adapter_config.json`.)
- **Teacher** — `Qwen3-Coder-480B-A35B` MLX 4-bit (1.1 TB local Mac Studio).
- **Adapter surface** — standard LoRA via `mlx_lm lora` on **17 module kinds** per layer: `linear_attn.{in_proj_a,in_proj_b,in_proj_qkv,in_proj_z,out_proj}` (GLA hybrid), `self_attn.{q,k,v,o}_proj`, `mlp.gate` + `mlp.shared_expert_gate` (MoE routers), `mlp.shared_expert.{down,gate,up}_proj`, `mlp.switch_mlp.{down,gate,up}_proj`. (Prior "attention-only, never MoE FFN" rule superseded 2026-04-18 — empirical forgetting test chat-fr↔reasoning mean 79.4° with all modules above 30°.)
- **Rank** — r=16 for all domains, alpha=16 (1:1 ratio per arXiv 2602.04998 "vanilla LoRA r=16 suffices when LR is tuned"; LR optimal ∝ r^(-1/2) per arXiv 2602.06204). 1.03B trainable params (2.96% of 35B).
- **Layers** — 32/40 optimal. 8 layers undertrained; 40 layers overfits (V3 chat-fr 1.304).
- **Learning rate** — 1e-5 (MLX quantized/BF16). Iters: 1000 foundations, 500 coding, 100-200 niches.
- **Metal optimization** — `mx.set_memory_limit(460GB)` + `mx.set_cache_limit(32GB)` required to prevent GPU Hang on M3 Ultra. Peak mem ~107 GB.
- **DoRA** — NOT supported on Qwen3.6 MoE (SwitchLinear incompatible).
- **Training** — MLX only. BF16. Sequential per-domain (never in parallel; stacks interfere). Foundations first, then niches (curriculum order).
- **Forgetting gate** — runs after every stack. Rollback if `cosine(adapter, prev) < 30°` **and** `win-rate drop > 0.03` on cross-domain probes. Canonical operator doc: `docs/training/forgetting-gate.md`.
- **Serving** — Q4_K_M only (quality cliff below). Max **4 active stacks** simultaneously per VRAM / interference budget.
- **Router** — 31 sigmoid outputs (35 domains merged to 31), **not** softmax. Domains co-activate (e.g. STM32 + embedded + DSP).
- **Adapter swap** — O(10ms) via in-place unpatch (not model reload).
- **KV cache** — 8-bit quantized with LRU prefix reuse for inference speedup (~2x).
- **strip_thinking** — default `true` (removes Qwen3 `<think>` CoT tags from output).

## Where to look

| Task | Location |
|---|---|
| Change the Python runtime (router, memory, serving, eval) | `src/` — each subdir has its own `CLAUDE.md` |
| Write or fix a test | `tests/` |
| Add / tune a training recipe, curriculum, or per-domain YAML | `configs/` |
| Generate a dataset, distill, or run an eval / benchmark | `scripts/` |
| Architecture decision logs | `docs/specs/` (most recent dated file wins) |
| Deploy (launchd / systemd / vLLM container) | `deploy/`, `docker/vllm.Dockerfile` |
| Worked examples (KiCad, SPICE, STM32 HAL, …) | `examples/` |

Artifacts (`checkpoints/`, `output/`, `results/`, `models/`, `data/`) contain build outputs — do not edit or add code guidance there.

## Quick start

### Run the full 7-stage pipeline server

```bash
uv sync --all-extras
uv run python full_pipeline_server.py --port 8000
```

Serves the full cognitive pipeline on `http://localhost:8000`. OpenAI-compatible — use any client:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "micro-kiki", "messages": [{"role": "user", "content": "Design a buck converter for 12V to 3.3V at 2A"}]}'
```

### Run the router + cognitive pipeline (local, no inference)

```bash
uv run python scripts/poc_pipeline_v2.py --scenario all
```

Loads trained adapters, initializes Aeon, routes 50 test prompts, logs routing decisions + latencies to `results/poc_latest.json`.

### Config gates (run before pushing config/`src/` changes)

```bash
python scripts/validate_domains.py            # domain list consistency across config mirrors
python scripts/validate_rank_schema.py        # rank = 16 · alpha = 16 (1:1 ratio)
python scripts/validate_curriculum_order.py   # foundations before niches
python scripts/validate_no_pre_pivot.py       # no Qwen3.5-4B leaks in src/
python scripts/validate_adapter_health.py <adapter.safetensors>  # all lora_B non-zero
python -m pytest tests/test_validate_*.py -q  # validator unit tests
```

`.github/workflows/validators.yml` runs these in two parallel CI jobs: `config-invariants` (the four validators + validator tests) and `forgetting-tests` (OPLoRA forgetting-measurement tests with CPU torch).

### Forgetting gate (after each stack trains)

```bash
# 1. Health-check the new adapter (lora_B non-zero everywhere)
python scripts/validate_adapter_health.py output/stacks/stack-NN-<domain>/adapter_model.safetensors

# 2. Measure per-module angle vs. all priors (angle + optional win-rate)
python scripts/measure_forgetting.py \
    --prior-adapter output/stacks/stack-03-cpp/adapter_model.safetensors \
    --new-adapter   output/stacks/stack-04-rust/adapter_model.safetensors \
    --output        results/forgetting-stack04-vs-stack03.json

# 3. Or run the one-shot orchestrator — exits 0/1/2/3 for pass/angle-fail/winrate-fail/health-fail
python scripts/post_train_gate.py <adapter-dir> --prior-dir output/stacks/
```

Operator runbook (dual-server real-adapter flow): `docs/training/e2e-smoke-runbook.md`. E2E smoke: `scripts/smoke_gate_on_studio.py` (last run in `results/smoke-gate.json` — chat-fr ↔ reasoning mean 79.4°, winrate_drop −0.04, gate PASS).

Bulk sweeps:
- `scripts/run_forgetting_sweep.py <adapter-dir>` — pairwise angle matrix (`results/forgetting-matrix.json`, `results/forgetting-matrix-prepivot.json`).
- `scripts/sweep_adapter_health.py <adapter-dir>` — bulk `lora_B` audit (`results/adapter-health-sweep.json`).

### Train a single domain (Mac Studio only)

Training is owned by the sibling repo — this README shows the driver only:

```bash
# From ~/KIKI-Mac_tunner, pointing at a config here
python -m mlx_lm.lora \
  --model Qwen/Qwen3.6-35B-A3B \
  --data ~/micro-kiki/data/merged/kicad-dsl/ \
  --config ~/micro-kiki/configs/lora/kicad-dsl.yaml \
  --output ~/micro-kiki/outputs/stacks/stack-01-kicad-dsl/
```

MLX Metal budget: `mx.set_memory_limit(460)` GB, `mx.set_cache_limit(32)` GB. Peak usage ≈ 107 GB.

### MLX serving (Mac Studio)

```bash
uv run python src/serving/mlx_server.py \
  --model ./outputs/base.safetensors \
  --adapters ./outputs/stacks/ \
  --port 8000 \
  --metal-memory-limit 460GB
```

### vLLM serving (kxkm-ai, RTX 4090)

```bash
uv run python src/serving/vllm_server.py \
  --model Qwen/Qwen3.6-35B-A3B \
  --quantization awq \
  --tensor-parallel-size 1 \
  --port 8001 \
  --gpu-memory-utilization 0.95
```

Supports Q4_K_M base + 2-4 active adapters simultaneously.

### Optional: `nerve-wml` advisor bridge

`MetaRouter.forward()` accepts an optional `query_tokens` argument and,
when the environment variable `NERVE_WML_ENABLED=1` is set, blends
per-domain advice produced by the sibling
[nerve-wml](https://github.com/hypneum-lab/nerve-wml) Nerve Protocol
into the domain slice of the raw logits **pre-sigmoid**:

```python
raw[:, :num_domains] = (1 - α) * raw[:, :num_domains] + α * advice
```

where α is read from `NERVE_WML_ALPHA` (default `0.5`). The advisor is
lazy-loaded once (`_get_nerve_wml_advisor()`), memoised on success, and
never raises into `forward()` — missing install, import error, or a
failing `advise()` call all fall back to the vanilla sigmoid path.

```bash
# Default (no advisor): forward is byte-identical to the pre-bridge
# baseline, zero perf cost, no import attempt.
python -c "from src.routing.router import MetaRouter; ..."

# Enable the advisor (requires nerve-wml installed in the same venv)
NERVE_WML_ENABLED=1 NERVE_WML_ALPHA=0.4 \
  uv run python src/serving/mlx_server.py --model ... --port 8000
```

Contract covered by `tests/routing/test_router_nerve_wml.py` (5 tests:
byte-identical default, graceful missing-advisor, alpha-blend math,
`query_tokens=None` bypass, advisor-raises pass-through).

## Hardware reality

| Role | Machine | Why |
|---|---|---|
| Training | Mac Studio M3 Ultra 512 GB | Only host with enough unified memory for BF16 LoRA on 35B-A3B (peak ~107 GB) |
| Teacher inference | Mac Studio (CPU) | `llama.cpp` on the 1.1 TB `Qwen3-Coder-480B-A35B`, ~5-10 tok/s |
| Production inference | Mac Studio (MLX, primary) / kxkm-ai (RTX 4090 24 GB) | Q4 base + adapters, 3.6-5.2s/query (MLX) or ~30-50 tok/s (vLLM) |
| Cognitive layer | Tower | Qdrant (Atlas) + Neo4j (Trace), ~16 GB RAM |

**Do not train on kxkm-ai** — 35B-A3B BF16 LoRA does not fit in 24 GB. **Do not use QLoRA / BitsAndBytes on 35B-A3B** — known MoE-layer corruption.

## V4 SOTA results

### Router V4 accuracy

| Metric | V3 (MiniLM 384d) | V4 (mpnet 768d) | Improvement |
|---|---|---|---|
| Top-1 accuracy | 46.4% | **70.7%** | +24.3pp |
| Top-3 accuracy | 69.6% | **89.5%** | +19.9pp |

Key changes in V4:
- Embedding: MiniLM 384d → mpnet-base-v2 768d
- Classifier: word boundary matching fixed
- Domain merges: 35→31 (electronics-hw, kicad, spice consolidated)
- Loss: BCE → CrossEntropy + class weights
- Architecture: MLP(768→512→31) → sigmoid

### Training history

| Config | chat-fr val_loss | reasoning val_loss | Notes |
|---|---|---|---|
| V1 (8L r8) | 0.891 | — | First baseline |
| V2 (32L r8) | 0.953 at iter 300 | — | More layers, same rank |
| V3 (40L r32) | 1.304 | — | Overfitting, rank too high |
| **V4 SOTA (32L r16)** | **0.849** | **0.638** | Best ever, -65% vs base 2.417 |

### Pipeline latency breakdown

| Stage | Latency |
|---|---|
| Router (mpnet + MLP) | ~8ms |
| LoRA swap (unpatch) | ~10ms |
| Aeon recall | ~62ms |
| MLX inference (Q4 + KV cache 8-bit) | 3.6-5.2s |
| Negotiator + Anti-bias | <50ms |
| Aeon write | ~32ms |
| **Total pipeline** | **~4-6s** |

### Published models and datasets

| Artifact | URL |
|---|---|
| Dataset (489K, 35 domains) | https://huggingface.co/datasets/clemsail/micro-kiki-v3-dataset |
| Router V4 (mpnet + MLP) | https://huggingface.co/clemsail/micro-kiki-router-v4 |
| Model (4B) | https://huggingface.co/clemsail/micro-kiki-v3 |
| Model (35B, 35 adapters + Opus adapters) | https://huggingface.co/clemsail/micro-kiki-v35b |

### Forgetting check (cross-stack interference)

Cross-stack angle mean 79.4°, all stacks above 30° threshold. No catastrophic forgetting detected across any of the 35 trained adapters.

| Domain | Angle (deg) | Pass |
|---|---|---|
| spice | 82.1 | 1.0 |
| stm32 | 79.4 | 0.78 |
| electronics | 76.3 | 0.69 |
| dsp | 74.8 | 0.69 |

### Stacks vs base

3/10 domains show measurable improvement over base 35B. The base model is already strong on well-represented domains (SPICE, electronics, embedded). The cognitive layer (Aeon memory + Negotiator CAMP) is the real differentiator — 36+ episode recalls per 14-turn dialogue vs 0 for raw LLM.

HumanEval pilot: base 0.80, cpp adapter 0.90 (+10%).

### SNN energy estimate

35B MoE on Loihi-2 (theoretical): **0.032 mJ/tok**. 91.6x ops reduction via LAS conversion. Efficiency score: 27B dense = 2.23, 35B MoE = 0.055 (MoE routing overhead dominates spike cost).

## Current work

V4 training **complete** — all 35/35 adapters trained and verified. Router V4 upgraded (mpnet 768d, 70.7% top-1). Full pipeline server deployed and stress-tested.

**Next steps:**
- Expand eval: broader HumanEval coverage, domain-specific benchmarks (KiCad DSL, SPICE netlist generation, STM32 HAL)
- Push remaining adapters to HuggingFace
- DPO/GRPO alignment (blocked on MLX native support)
- Router V5: explore cross-attention routing and hard-negative mining for sparse domains

## Related repos

| Repo | Relation |
|---|---|
| [**KIKI-Mac_tunner**](https://github.com/L-electron-Rare/KIKI-Mac_tunner) | Sibling — training execution, MLX pipeline, datasets, `mlx-lm` fork |
| [**KIKI-models-tuning**](https://github.com/L-electron-Rare/KIKI-models-tuning) | Upstream — FineFab QLoRA pipeline for Qwen 2.5-32B domain experts (downstream to micro-kiki) |
| [**mascarade**](https://github.com/electron-rare/mascarade) | Runtime consumer — LLM orchestration across 8 providers, loads adapters at inference |
| [**dream-of-kiki**](https://github.com/electron-rare/dream-of-kiki) | Research sibling — dream-based knowledge consolidation, shares profile concepts |
| [**kiki-flow-research**](https://github.com/electron-rare/kiki-flow-research) | Research sibling — Wasserstein flow engine for consolidation, advisory routing callback |

## Directory layout

```
src/
  routing/       Router V4, MetaRouter, domain classifier (mpnet + MLP)
  memory/        Aeon — Atlas (SIMD vector) + Trace (neuro-symbolic graph)
  negotiator/    CAMP arbitration + Catfish dissent
  eval/          Reward functions, forgetting gate (forgetting.py, scorers.py), bias metrics
  serving/       Full pipeline server + MLX server + mlx_client + vLLM server (Q4)
scripts/         70+ entry points (train drivers, eval, distill, benchmarks, validators, gate)
  legacy/        Archived pre-pivot drivers (Qwen3.5-4B era, MoE-LoRA dead-weight adapters)
configs/         YAML recipes — one per domain, lora/ + serving/
data/merged/     Per-domain JSONL train/valid/test
tests/           Router, memory, negotiator, reward, validator tests (no 35B loading)
docs/            specs/ (decisions), research/, plans/, training/ (forgetting-gate, e2e-smoke-runbook)
results/         Eval artefacts — forgetting-matrix.json, adapter-health-sweep.json, smoke-gate.json
deploy/          launchd (Mac Studio) + systemd (kxkm-ai) + docker-compose
.github/         workflows/validators.yml — config-invariants + forgetting-tests jobs
```

## License

Apache License 2.0. See [`LICENSE`](./LICENSE).
