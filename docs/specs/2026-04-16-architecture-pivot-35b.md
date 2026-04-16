# Architecture Pivot: Qwen3.5-4B → Qwen3.5-35B-A3B

Date: 2026-04-16
Status: Approved
Decision: Replace 4B dense base with 35B-A3B MoE sparse

## Why

- Qwen3.5-9B beats GPT-OSS-120B on MMLU-Pro (82.5%). The 35B-A3B is the same architecture family, with 256 experts and only 3B active per token.
- The model **is already a MoE** — our custom MoE-LoRA (4 experts per projection) becomes redundant. Simple LoRA per domain suffices.
- Mac Studio M3 Ultra 512 GB handles BF16 LoRA training at 74 GB.
- Teacher is local: Qwen3-Coder-480B-A35B MLX 4bit (1.1 TB, already downloaded).
- 32 domain datasets already prepared (70K+ examples, classified + deduped).

## What changes

### Simplified: MoE-LoRA → LoRA

| Before (4B) | After (35B-A3B) |
|---|---|
| Custom MoE-LoRA: 4 experts, rank 16, top-2 routing | Standard LoRA: rank 16, target q/k/v/o projections |
| OPLoRA init for forgetting prevention | OPLoRA still applies (orthogonal projection per adapter) |
| Custom `src/stacks/moe_lora.py` | **Deleted** — use PEFT `LoraConfig` directly |
| Custom sigmoid meta-router (32→37 outputs) | **Simplified** — domain classifier, no expert routing needed |
| Max 4 stacks active (VRAM 24 GB cap) | Max 4 stacks active (74 GB BF16 budget) |

### Unchanged

- 32 domain adapters (one LoRA per domain)
- Forgetting check framework (angle + win-rate)
- Dispatcher (7 meta-intents)
- Aeon Memory Palace
- Negotiator + Anti-bias
- Phase 14 agentic capabilities (search, critique, ralph)
- Eval harness + curriculum order

### New constraints

- Training: Mac Studio ONLY (74 GB BF16 LoRA). RTX 4090 for inference only (Q4 ~20 GB).
- No QLoRA for 35B-A3B MoE (BitsAndBytes issues with MoE layers).
- Unsloth MoE optimizations: 12x faster training, 35% less VRAM.
- Set `UNSLOTH_COMPILE_DISABLE=1` to avoid mixed-precision kernel errors.

## Hardware allocation (revised)

| Machine | Role |
|---|---|
| Mac Studio (512 GB) | Training (BF16 LoRA), teacher inference (480B MLX), MLX serving |
| kxkm-ai (RTX 4090) | Q4 inference only, benchmarks |
| Tower | Aeon backends (Qdrant, Neo4j), Piper TTS |

## Model specs

- **Base**: `Qwen/Qwen3.5-35B-A3B` (Apache 2.0)
- **Architecture**: 256 experts, 3B active per token, GatedDeltaNet + MoE
- **Context**: 262K native (extensible 1M)
- **BF16 size**: ~70 GB
- **Q4_K_M size**: ~20 GB
- **LoRA rank**: 16 (same as before)
- **LoRA targets**: q_proj, k_proj, v_proj, o_proj (attention only, not MoE FFN)
- **Trainable params per adapter**: ~931M (2.58% of 36B)

## Training budget

- Per-stack LoRA training: ~45 min on Mac Studio (BF16, Unsloth 12x MoE)
- 32 stacks × 45 min = ~24h total training
- Teacher distillation: Qwen3-Coder-480B local, ~2s/example, 70K examples = ~39h
- Total: ~3 days end-to-end on Mac Studio alone

## Risk

- 74 GB training + model weights = tight on 512 GB with other processes. Mitigation: close other apps during training, use gradient checkpointing.
- vLLM LoRA serving for 35B-A3B MoE has known issues (NVIDIA forum thread). Mitigation: MLX serving on Mac Studio as primary, vLLM as fallback.
- No kxkm-ai for training. Mitigation: Mac Studio is sufficient; kxkm-ai handles inference only.
