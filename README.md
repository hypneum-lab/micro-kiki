# micro-kiki

**35 domain-expert LoRA adapters + cognitive layer on Qwen 3.5-35B-A3B (native MoE, 256 experts, 3B active).**

Sequential per-domain training via MLX on Mac Studio M3 Ultra 512 GB. Q4_K_M inference on kxkm-ai (RTX 4090 24 GB). Router is 35 sigmoid outputs — domains are not mutually exclusive.

> **Training, datasets, and the `mlx-lm` fork live in the sibling repo [`KIKI-Mac_tunner`](https://github.com/L-electron-Rare/KIKI-Mac_tunner).** This repo holds the runtime: routing, cognitive layer, serving, eval, and the per-domain configs that drive the tuner.

---

## Architecture

```
Domain query
    │
    ▼
[MetaRouter · 35 sigmoid outputs]   ≤ 4 active adapters at a time
    │
    ▼
[Qwen3.5-35B-A3B Q4_K_M + {adapter_1, …, adapter_k}]    MoE routing preserved; LoRA on q/k/v/o only
    │
    ▼
[Aeon memory recall]   Atlas (SIMD vector) + Trace (neuro-symbolic graph)
    │
    ▼
[MLX / vLLM inference]
    │
    ▼
[Negotiator]   CAMP arbitration + Catfish dissent
    │
    ▼
[Anti-bias filter]   KnowBias double-application + RBD + DeFrame
    │
    ▼
[Aeon memory write]   Persist episode
    │
    ▼
Response
```

## Hard invariants (load-bearing across the whole project)

- **Base** — `Qwen/Qwen3.5-35B-A3B` (Apache 2.0, 262 K context).
- **Teacher** — `Qwen3-Coder-480B-A35B` MLX 4-bit (1.1 TB local Mac Studio).
- **Adapter surface** — standard LoRA on q/k/v/o attention projections **only**. Never on MoE FFN: the MoE routing is already trained, touching it collapses expert specialization.
- **Rank budget** — 4–16 for niches, 32 for foundations; `alpha = 2 × rank`; scale 2.0.
- **Training** — MLX only. BF16. Sequential per-domain (never in parallel; stacks interfere). Foundations first, then niches (curriculum order).
- **Forgetting gate** — runs after every stack. Rollback if `cosine(adapter, prev) < 30°` **and** `win-rate drop > 0.03` on cross-domain probes.
- **Serving** — Q4_K_M only (quality cliff below). Max **4 active stacks** simultaneously per VRAM / interference budget.
- **Router** — 35 sigmoid outputs, **not** softmax. Domains co-activate (e.g. STM32 + embedded + DSP).

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

### Run the router + cognitive pipeline (local, no inference)

```bash
uv sync --all-extras
uv run python scripts/poc_pipeline_v2.py --scenario all
```

Loads trained adapters, initializes Aeon, routes 50 test prompts, logs routing decisions + latencies to `results/poc_latest.json`.

### Config gates (run before pushing config/`src/` changes)

```bash
python scripts/validate_domains.py            # 32-domain list consistency
python scripts/validate_rank_schema.py        # rank ∈ {4,8,12,16,32} · alpha = 2·rank
python scripts/validate_curriculum_order.py   # foundations before niches
python scripts/validate_no_pre_pivot.py       # no Qwen3.5-4B leaks in src/
```

Forgetting angle (OPLoRA phase 1a, informational): `python scripts/measure_forgetting.py --prior-adapter ... --new-adapter ...` — see `docs/training/forgetting-gate.md`.

### Train a single domain (Mac Studio only)

Training is owned by the sibling repo — this README shows the driver only:

```bash
# From ~/KIKI-Mac_tunner, pointing at a config here
python -m mlx_lm.lora \
  --model Qwen/Qwen3.5-35B-A3B \
  --data ~/micro-kiki/data/merged/kicad-dsl/ \
  --config ~/micro-kiki/configs/lora/kicad-dsl.yaml \
  --output ~/micro-kiki/outputs/stacks/stack-01-kicad-dsl/
```

MLX Metal budget: `mx.set_memory_limit(460)` GB, `mx.set_cache_limit(32)` GB. Peak usage ≈ 106 GB.

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
  --model Qwen/Qwen3.5-35B-A3B \
  --quantization awq \
  --tensor-parallel-size 1 \
  --port 8001 \
  --gpu-memory-utilization 0.95
```

Supports Q4_K_M base + 2-4 active adapters simultaneously.

## Hardware reality

| Role | Machine | Why |
|---|---|---|
| Training | Mac Studio M3 Ultra 512 GB | Only host with enough unified memory for BF16 LoRA on 35B-A3B (peak 106 GB) |
| Teacher inference | Mac Studio (CPU) | `llama.cpp` on the 1.1 TB `Qwen3-Coder-480B-A35B`, ~5-10 tok/s |
| Production inference | kxkm-ai (RTX 4090 24 GB) | Q4_K_M base + 2-4 adapters, ~30-50 tok/s |
| Cognitive layer | Tower | Qdrant (Atlas) + Neo4j (Trace), ~16 GB RAM |

**Do not train on kxkm-ai** — 35B-A3B BF16 LoRA does not fit in 24 GB. **Do not use QLoRA / BitsAndBytes on 35B-A3B** — known MoE-layer corruption.

## Current work

Active PoC branch: **Aeon Latent Predictor** (PoC B) — stack scaling, stream alignment, centering ablation, DinoV3-style running-mean. See recent commits under `merge: PoC B …`. Eval scripts live in `scripts/`; ablation reports in `docs/research/`.

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
  routing/       35-sigmoid router, MetaRouter, domain classifier
  memory/        Aeon — Atlas (SIMD vector) + Trace (neuro-symbolic graph)
  negotiator/    CAMP arbitration + Catfish dissent
  eval/          Reward functions, forgetting gate, bias metrics
  serving/       MLX server (adapter hot-swap) + vLLM server (Q4)
scripts/         70+ entry points (train drivers, eval, distill, benchmarks)
configs/         YAML recipes — one per domain, lora/ + serving/
data/merged/     Per-domain JSONL train/valid/test
tests/           Router, memory, negotiator, reward tests (no 35B loading)
docs/            specs/ (decisions), research/, plans/
deploy/          launchd (Mac Studio) + systemd (kxkm-ai) + docker-compose
```

## License

Apache License 2.0. See [`LICENSE`](./LICENSE).
