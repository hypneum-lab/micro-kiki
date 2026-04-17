---
license: apache-2.0
language:
  - fr
  - en
tags:
  - moe
  - lora
  - multi-domain
  - embedded-systems
  - cognitive
base_model: Qwen/Qwen3.5-35B-A3B
pipeline_tag: text-generation
---

# micro-kiki

**35-domain expert model** built on Qwen3.5-35B-A3B (MoE, 256 experts, 3B active/token) with LoRA adapters and a cognitive layer (memory palace + negotiator + anti-bias).

## Model Description

micro-kiki is a multi-domain language model designed for technical applications spanning electronics, firmware, CAD, manufacturing, and general-purpose conversation. It uses a router-based architecture that selects up to 4 domain-specific LoRA stacks per request.

| Property | Value |
|----------|-------|
| Base model | Qwen3.5-35B-A3B |
| Architecture | MoE (256 experts, 3B active/token) |
| Adapter | LoRA rank 16 (q/k/v/o projections) |
| Domains | 35 |
| Max active stacks | 4 |
| Context length | 262,144 tokens |
| Quantization | Q4_K_M (inference), BF16 (training) |
| License | Apache 2.0 |

## Architecture

```
                         +-------------------+
                         |   Domain Router   |
                         | (classifier, top4)|
                         +--------+----------+
                                  |
              +----------+--------+--------+----------+
              |          |                 |          |
         +----v----+ +---v---+       +----v----+ +---v---+
         | Stack 1 | |Stack 2|  ...  |Stack 34 | |Stack35|
         | chat-fr | |python |       |ml-train | |securi.|
         +---------+ +-------+       +---------+ +-------+
              |          |                 |          |
              +----------+--------+--------+----------+
                                  |
                         +--------v----------+
                         |    Negotiator     |
                         | CAMP + Catfish    |
                         +--------+----------+
                                  |
                         +--------v----------+
                         |    Anti-Bias      |
                         | KnowBias + RBD   |
                         +--------+----------+
                                  |
                         +--------v----------+
                         |   Aeon Memory     |
                         | Atlas + Trace     |
                         +-------------------+
```

## Intended Use

- **French/English conversational AI** with domain expertise
- **Code generation** (Python, C/C++, Rust, TypeScript, embedded firmware)
- **Electronics design** (KiCad DSL, schematic review, component selection, SPICE)
- **Manufacturing** (process optimization, quality control)
- **Multi-domain routing** with cognitive arbitration

## Limitations

- Not designed for medical, legal, or financial advice
- Optimized for technical domains; general knowledge may be weaker than base model
- Requires Q4_K_M or higher quantization; quality degrades below Q4
- Maximum 4 concurrent LoRA stacks; performance varies with stack combinations
- Memory (Aeon) requires external backends (Qdrant/Neo4j) for production use

## Training Data — V3 (489K examples, 35 domains)

### Sources

| Source | Examples | Description |
|--------|----------|-------------|
| Claude CLI sessions | 50,116 | Real user-tool interactions extracted from 5 machines (GrosMac, kxkm-ai, Studio, Tower, CILS) |
| Codex/Copilot sessions | 2,529 | OpenAI Codex + GitHub Copilot sessions extracted from 4 machines |
| HuggingFace datasets | 364,045 | 19 open datasets (see below) |
| Opus teacher distillation | — | chat-fr, reasoning domains |
| Original curated | — | 32 domain seed datasets |

### HuggingFace Datasets

| Dataset | Examples | License |
|---------|----------|---------|
| CodeFeedback-Filtered-Instruction | 157,000 | Apache 2.0 |
| French-Alpaca-Instruct-110K | 110,000 | Apache 2.0 |
| Electronics StackExchange | 95,000 | CC-BY-SA-3.0 |
| CJJones/LLM_EE_Educational_Synthetic_Dialog | 50,000 | CC-BY-NC-SA-4.0 |
| MuratKomurcu/stm32-hal-dataset | 29,700 | MIT |
| redcathode/thingiverse-openscad | 7,400 | — |
| ThomasTheMaker/OpenSCAD | 4,900 | — |
| STEM-AI-mtl/Electrical-engineering | 1,100 | — |
| JITX open-components-database | 151 | — |
| Vrindarani/netlistgen | 106 | — |

### 35 Domains

| Group | Domains |
|-------|---------|
| Conversation | chat-fr, reasoning |
| Code | python, typescript, cpp, rust, html-css, shell, sql, yaml-json, lua-upy |
| Infrastructure | docker, devops, llm-orch, llm-ops (NEW), ml-training (NEW) |
| Electronics | kicad-dsl, kicad-pcb, spice, electronics, components (NEW), power, emc, dsp |
| Hardware | embedded, stm32, iot, platformio |
| CAD | freecad |
| Web | web-frontend, web-backend |
| Other | music-audio, math, security |

**Changes from V2:** 3 new domains (components, llm-ops, ml-training). `spice-sim` merged into `spice`. `stm32` is a sub-category of `embedded`.

### New Domain: components

57K Q&A about electronic component specs, datasheets, sourcing, BOM, and cross-reference. Sources: Electronics StackExchange (filtered by component tags) + JITX open-components-database.

## Training — V3

| Property | Value |
|----------|-------|
| Base model | Qwen3.5-4B |
| Adapter | MoE-LoRA: 4 experts/projection, rank 16, top-2 routing |
| Null-space projection | ENABLED (prevents catastrophic forgetting between stacks) |
| Curriculum | Sequential, 35 stacks trained in order |
| Platform (MLX) | Mac Studio M3 Ultra 512 GB |
| Platform (CUDA) | kxkm-ai RTX 4090 24 GB |

## Evaluation

| Metric | Value |
|--------|-------|
| Router accuracy (35-class) | [PENDING] |
| Forgetting check (angle) | [PENDING] |
| Perplexity (base) | [PENDING] |
| Perplexity (debiased) | [PENDING] |
| Aeon recall@1 | [PENDING] |
| Aeon recall@5 | [PENDING] |
| Aeon recall@10 | [PENDING] |
| Anti-bias flag rate | [PENDING] |
| Average inference latency | [PENDING] |

## Hardware Requirements

| Setup | RAM/VRAM | Use |
|-------|----------|-----|
| Mac Studio M3 Ultra | 512 GB unified | Training (BF16 LoRA) + serving (MLX) |
| RTX 4090 | 24 GB VRAM | Q4 inference (vLLM) |
| Apple Silicon 32 GB+ | 32 GB unified | Q4_K_M inference (MLX/llama.cpp) |

## Citation

```bibtex
@misc{micro-kiki-2026,
  title={micro-kiki: Multi-Domain Expert Model with Cognitive Layer},
  author={L'Electron Rare},
  year={2026},
  url={https://huggingface.co/electron-rare/micro-kiki}
}
```
