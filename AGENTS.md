<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# micro-kiki

## Purpose
micro-kiki is a 35-domain expert system built on the Qwen3.5-35B-A3B MoE base (256 experts, 3B active per token) via standard LoRA adapters applied only to attention projections (q/k/v/o). On top of the adapter layer sits a cognitive stack: sigmoid meta-router + YAML dispatcher, Aeon memory palace (Atlas SIMD + Trace graph), CAMP/Catfish negotiator, and KnowBias+RBD anti-bias. Training runs sequentially per domain via MLX on a Mac Studio M3 Ultra 512 GB; inference runs on MLX (Mac) or vLLM Q4 (kxkm-ai RTX 4090). Distillation uses Qwen3-Coder-480B-A35B MLX 4bit as a local teacher. Version 0.2.0-dev.

## Architecture

```
query
  |
  v
[routing/]          --> 35-output sigmoid (34 niche domain + 1 base; capabilities served separately)
  |                    thresholds: 0.12 general, 0.20 chat-mode, max 4 stacks
  v
[orchestrator/]     --> dispatcher maps router output to 7 meta-intents
  |
  v
[stacks/] LoRA      --> attention-only LoRA (q/k/v/o); NEVER FFN on MoE
  |                    rank 4-16 niches, 32 foundations; scale 2.0
  v
[cognitive/]        --> negotiator (CAMP+Catfish), KnowBias, RBD, forgetting_gate
  |
  v
[memory/] Aeon      --> Atlas SIMD ANN + Trace graph; native or Qdrant/Neo4j
  |
  v
[serving/]          --> MLX primary (Mac), vLLM Q4 (kxkm-ai)
```

## Key Files
| File | Description |
|------|-------------|
| `CLAUDE.md` | Authoritative conventions (base model, adapter strategy, commits, Do/Don't) |
| `MODEL_CARD.md`, `MODEL_CARD-v0.3.md` | Published model cards |
| `COOKBOOK.md` | End-user recipes for training / inference |
| `README.md` | Short project handle (content cached line 1 only) |
| `BRANCH-neuroscience.md` | v0.3 neuroscience branch notes (SpikingBrain / LIF / LAS) |
| `MIGRATION.md` | v0.1 -> v0.2 -> v0.3 migration guide |
| `pyproject.toml` | hatchling build, Python 3.11+, optional extras: `train`, `mlx`, `serve`, `agentic`, `dev` |
| `VERSION` | `0.2.0-dev` |
| `train_micro_kiki_v3_gpu.py` | Top-level v3 GPU training entrypoint (4090 prototyping) |

## Subdirectories
| Directory | Purpose | AGENTS.md |
|-----------|---------|-----------|
| `src/` | Python package (`src/*` -> installed top-level via hatch) | yes |
| `tests/` | pytest suite, conftest fixtures | yes |
| `scripts/` | Training drivers, distillation, eval, pipeline helpers | yes |
| `configs/` | Per-stack YAML, MLX curricula, meta-intent & capability maps | yes |
| `docs/` | Specs, plans, research, superpowers, training READMEs | yes |
| `research/` | Exploratory work (ANE hybrid pipeline) | yes |
| `examples/` | Minimal usage snippets (chat, memory, bias, forgetting) | yes |
| `deploy/` | launchd (Mac) + systemd (Linux) units | yes |
| `docker/` | vllm Dockerfile | yes |
| `hardware/` | KiCad schematics (STM32H743 bootloader, SPI bus) | yes |
| `data/` | Datasets and distilled JSONL — data-only, no AGENTS.md |
| `outputs/` | Training outputs (adapters, checkpoints) — data-only, no AGENTS.md |
| `output/` | Legacy output dir — data-only, no AGENTS.md |
| `results/` | Eval result JSON — data-only, no AGENTS.md |

## For AI Agents

### Working In This Repository
- READ `CLAUDE.md` first; it overrides defaults (base model, adapter rule, commit format).
- Enforce hard rules:
  - NEVER LoRA-tune MoE FFN layers — attention projections only.
  - NEVER use QLoRA / BitsAndBytes on 35B-A3B (MoE mixed-precision kernels break).
  - NEVER train on kxkm-ai (RTX 4090 24 GB cannot hold 35B BF16 LoRA).
  - NEVER drop below Q4 quantization at inference.
  - NEVER route > 4 stacks simultaneously.
  - NEVER train stacks in parallel — sequential curriculum only.
  - Run forgetting check after EVERY stack; rollback if angle < 30 AND win-rate drop > 0.03.
- `UNSLOTH_COMPILE_DISABLE=1` before any training on the Mac Studio.
- Python 3.11+, ruff + black, line length 100, loguru for logging (no `print()`).
- Commits: `feat|fix|docs(<area>): <imperative>`, subject <= 50 chars, no `Co-Authored-By` trailer (pre-commit hook rejects it).

### Testing Requirements
```bash
uv run python -m pytest                  # full
uv run python -m pytest tests/routing    # targeted
uv run python -m pytest -m "not integration"   # skip real-model / SSH tests
```
`asyncio_mode = auto` is set in pyproject; integration tests marked `@pytest.mark.integration` are opt-in.

### Common Patterns
- `from __future__ import annotations` at the top of every module.
- Explicit device placement; never rely on implicit CUDA.
- Dataclass configs with `frozen=True` or Pydantic v2 `BaseModel`.
- YAML configs loaded through `src.routing.dispatcher.load_intent_mapping` or equivalents.
- Adapter outputs live under `~/KIKI-Mac_tunner/output/micro-kiki/stack-NN-<domain>/`.

## Dependencies

### External (core)
`httpx>=0.27`, `pyyaml>=6.0`, `numpy>=1.26`, `huggingface-hub>=0.28`.

### Optional extras
- `train`: torch, transformers, peft, trl, accelerate, bitsandbytes, datasets
- `mlx`: mlx>=0.26, mlx-lm>=0.30
- `serve`: vllm, fastapi, uvicorn
- `agentic`: beautifulsoup4
- `dev`: pytest, pytest-asyncio

### External services / hardware
- Mac Studio M3 Ultra 512 GB — training + MLX serving + teacher (480B)
- kxkm-ai RTX 4090 24 GB — Q4 inference only
- Tower — Qdrant + Neo4j Aeon backends, Piper TTS
- HuggingFace: `Qwen/Qwen3.5-35B-A3B` (base), `Qwen/Qwen3-Coder-480B-A35B-Instruct` (teacher)

<!-- MANUAL: -->
