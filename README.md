# micro-kiki — Triple-Hybrid Domain Expert LLM System

A quantum-neuromorphic-classical routing architecture for hardware/EDA domain expertise. Combines 10 niche LoRA adapters on Qwen3.5-35B-A3B with a cognitive layer (Aeon Memory Palace + CAMP Negotiator + Anti-bias detection) and experimental SNN routing via LAS conversion.

## Overview

**micro-kiki** is a production-ready system that specializes a 35B MoE LLM for 10 hardware engineering domains (KiCad, SPICE, STM32, embedded, power delivery, DSP, EMC, FreeCAD, PlatformIO, and electronics). Rather than training a monolithic 32-domain LoRA stack, the system identifies which domains the base model already handles well (22 "known" domains) and focuses expert adapters only on the genuine gaps (10 "niche" domains).

The architecture layers are:

1. **Quantum VQC Router** — 4-qubit quantum circuit for ultra-compact domain classification (200 parameters vs 3.4M classical)
2. **SNN Backbone** — Latency-Aware Spiking conversion of the 35B MoE for energy-efficient alternative routing
3. **Classical Backbone** — Qwen3.5-35B-A3B with 10 LoRA adapters (rank 4-16 depending on domain size)
4. **Cognitive Layer** — Aeon memory palace (SIMD vector index + neuro-symbolic episodic graph), CAMP arbitration, Catfish dissent, and KnowBias double-application for bias detection

This is the first published system combining quantum, neuromorphic, and classical routing for production LLM inference. See `docs/paper-outline-triple-hybrid.md` for the full paper roadmap.

## Architecture Diagram

```
Domain Query
    |
    v
[Quantum VQC Router] (4 qubits, 6 variational layers)
    |                    |
    | (confidence > θ)   | (confidence < θ)
    v                    v
[SNN SpikingKiki]  [Classical MetaRouter]
    |                    |
    +----+--------+------+
         |
         v
[Domain Classifier] → 10 niche domains + 1 passthrough
         |
         v
[Model Router] → Select 35B base or 35B+LoRA<domain>
         |
         v
[Aeon Memory Recall] → Inject context (Atlas + Trace)
         |
         v
[MLX/vLLM Inference]
         |
         v
[Negotiator Layer] → CAMP arbitration + Catfish dissent
         |
         v
[Anti-bias Filter] → KnowBias + RBD detector + DeFrame
         |
         v
[Aeon Memory Write] → Persist episode
         |
         v
Response to User
```

## Domains

The 10 specialized niche domains are:

| Domain | Category | Estimated Examples | LoRA Rank | Notes |
|--------|----------|-------------------|-----------|-------|
| kicad-dsl | Hardware | 8,500+ | 16 | KiCad schematic/PCB DSL, S-expressions |
| spice | Simulation | 9,200+ | 16 | SPICE netlists, circuit simulation |
| emc | Compliance | 4,100+ | 16 | Electromagnetic compliance, shielding |
| stm32 | Firmware | 2,700+ | 16 | STM32 HAL, peripheral configuration |
| embedded | Systems | 13,800+ | 16 | General embedded systems, RTOS, baremetal |
| freecad | CAD | 1,200+ | 8 | FreeCAD parametric design, Python scripting |
| platformio | Build | 2,100+ | 8 | PlatformIO framework, build configuration |
| power | Power | 3,500+ | 16 | Power delivery, buck/boost, thermal |
| dsp | Signal Processing | 5,600+ | 16 | DSP algorithms, FFT, filtering |
| electronics | Components | 1,900+ | 8 | Component selection, datasheet interpretation |

The remaining 22 "known" domains are served by passthrough (base 35B-A3B without LoRA) because the base model already achieves < 5% loss delta vs specialist performance.

## Quick Start

### POC Pipeline

Run the full POC pipeline (all 10 domains + cognitive layers):

```bash
python3 scripts/poc_pipeline_v2.py --scenario all
```

This will:
- Load the 10 trained LoRA adapters
- Initialize Aeon memory palace
- Run domain router on 50 test prompts (5 per domain)
- Log routing decisions, adapter hot-swaps, and latency
- Output results to `results/poc_latest.json`

### Training a Single Domain

Train a niche LoRA adapter on Mac Studio M3 Ultra:

```bash
python3 -m mlx_lm.lora \
  --model "Qwen/Qwen3.5-35B-A3B" \
  --data ./data/merged/kicad-dsl/ \
  --config ./configs/lora/kicad-dsl.yaml \
  --output ./outputs/stacks/stack-01-kicad-dsl/
```

Config includes:
- Rank: 16 (or 8 for small domains like freecad/electronics)
- Alpha: 2 × rank (scale 2.0)
- LR: 2e-5 to 5e-5 (curriculum schedule)
- Max sequence: 2048 (niches) or 4096 (foundations)
- Metal memory limit: 460 GB, cache limit: 32 GB

### Training the Quantum Router

Train the 4-qubit VQC domain classifier:

```bash
python3 scripts/train_vqc_router.py \
  --num_qubits 4 \
  --num_layers 6 \
  --epochs 100 \
  --learning_rate 0.1
```

Trains on synthetic domain-labeled embeddings and produces a PennyLane circuit.

### Training SNN Variants

Convert Qwen3.5-27B or 35B-A3B to spiking via Latency-Aware Spiking (LAS):

```bash
python3 scripts/convert_spikingkiki_35b.py \
  --base_model "Qwen/Qwen3.5-35B-A3B" \
  --output_dir ./outputs/snn/spikingkiki-35b/ \
  --num_steps 100 \
  --temperature 2.0
```

Estimated time: ~40 hours on Mac Studio for 35B-A3B.

### MLX Serving

Start the MLX serving pipeline with dynamic adapter loading:

```bash
python3 src/serving/mlx_server.py \
  --model ./outputs/base.safetensors \
  --adapters ./outputs/stacks/ \
  --port 8000 \
  --metal_memory_limit 460GB
```

Clients can then request domain-specific responses by sending routing hints or letting the domain classifier decide.

### vLLM Serving (GPU)

For inference on RTX 4090 (kxkm-ai):

```bash
python3 src/serving/vllm_server.py \
  --model "Qwen/Qwen3.5-35B-A3B" \
  --quantization "awq" \
  --tensor_parallel_size 1 \
  --port 8001 \
  --gpu_memory_utilization 0.95
```

Supports Q4_K_M quantized base + 2-4 active adapters simultaneously.

## Project Structure

```
micro-kiki/
├── README.md                           # This file
├── CLAUDE.md                           # Project context for Claude Code
├── LICENSE                             # Apache 2.0
├── pyproject.toml                      # uv + pytest config
│
├── src/
│   ├── routing/
│   │   ├── domain_classifier.py        # 11-output domain router
│   │   ├── meta_router.py              # Multi-model tier selection
│   │   └── quantum_vqc.py              # 4-qubit PennyLane VQC
│   ├── spiking/
│   │   ├── las_converter.py            # Latency-Aware Spiking
│   │   ├── neuron.py                   # LIF neuron models
│   │   └── energy_proxy.py             # Spike-based energy estimation
│   ├── memory/
│   │   ├── aeon.py                     # Atlas SIMD + Trace graph
│   │   └── backends.py                 # Qdrant/Neo4j/native adapters
│   ├── negotiator/
│   │   ├── camp.py                     # CAMP arbitration
│   │   └── catfish.py                  # Catfish dissent
│   ├── eval/
│   │   ├── reward_functions.py         # Syntax, format, accuracy rewards
│   │   └── metrics.py                  # BLEU, exact-match, win-rate
│   └── serving/
│       ├── mlx_server.py               # MLX inference + adapter swaps
│       └── vllm_server.py              # vLLM Q4 inference
│
├── scripts/
│   ├── poc_pipeline_v2.py              # Full POC for all domains
│   ├── train_vqc_router.py             # Train 4-qubit VQC
│   ├── benchmark_base_vs_lora.py       # Identify niche vs known domains
│   ├── merge_datasets.py               # Merge KIKI-Mac_tunner + HF enrichment
│   ├── convert_spikingkiki_35b.py      # LAS conversion for 35B MoE
│   ├── eval_niche_vs_base.py           # Per-domain evaluation
│   ├── generate_dpo_pairs.py           # Generate preference pairs via 480B
│   ├── train_grpo_niches.py            # GRPO training for reasoning domains
│   └── micro_kiki/
│       └── *.py                        # Utility modules
│
├── configs/
│   ├── lora/
│   │   ├── kicad-dsl.yaml              # Stack-01 config
│   │   ├── spice.yaml                  # Stack-02 config
│   │   └── ...                         # 8 more domain configs
│   ├── quantum/
│   │   └── vqc-4qubit.yaml             # VQC router config
│   └── serving/
│       ├── mlx.yaml                    # MLX server config
│       └── vllm.yaml                   # vLLM server config
│
├── data/
│   ├── merged/
│   │   ├── kicad-dsl/
│   │   │   ├── train.jsonl
│   │   │   ├── valid.jsonl
│   │   │   └── test.jsonl
│   │   └── ... (9 more domains)
│   ├── bias/                           # 5,213-pair bias probe dataset
│   └── eval/
│       └── domain_test_sets/           # Held-out test for each domain
│
├── outputs/
│   ├── base.safetensors                # Base 35B-A3B weights
│   ├── stacks/
│   │   ├── stack-01-kicad-dsl/
│   │   │   └── adapters.safetensors
│   │   └── ... (9 more stacks)
│   └── snn/
│       ├── spikingkiki-27b/
│       ├── spikingkiki-35b/
│       └── spikingkiki-122b/ (optional)
│
├── results/
│   ├── poc_latest.json                 # Latest POC run
│   ├── benchmark_base_vs_lora.json     # Niche identification
│   ├── stacks_vs_base_eval.json        # Per-domain eval
│   ├── snn_eval_27b.json               # SpikingKiki-27B benchmark
│   ├── snn_eval_35b.json               # SpikingKiki-35B benchmark
│   ├── anti_bias_eval_full.json        # Bias detection/mitigation
│   └── energy_benchmark.json           # Theoretical + measured energy
│
├── docs/
│   ├── paper-outline-triple-hybrid.md  # Full paper roadmap
│   ├── research/
│   │   ├── sota-training-2026.md       # Training SOTA
│   │   ├── micro-kiki-moe-research.md  # MoE research
│   │   └── 2026-04-16-landscape.md     # Competitive landscape
│   ├── specs/
│   │   └── 2026-04-16-architecture-pivot-35b.md  # Architecture decisions
│   ├── plans/
│   │   └── v0.2-roadmap.md
│   └── data-sources.md                 # Dataset provenance
│
├── tests/
│   ├── test_routing.py                 # Domain classifier tests
│   ├── test_memory.py                  # Aeon recall/write tests
│   ├── test_negotiator.py              # CAMP/Catfish tests
│   ├── test_snn.py                     # SNN conversion tests
│   └── test_reward_functions.py        # Reward computation tests
│
└── deploy/
    ├── mlx-server.plist                # launchd for Mac Studio
    ├── vllm-server.service             # systemd for kxkm-ai
    └── docker-compose.yml              # Stack orchestration
```

## Hardware Requirements

### Training (Mac Studio M3 Ultra 512 GB)

- **CPU**: Apple M3 Ultra (16-core)
- **Memory**: 512 GB unified memory (required for BF16 LoRA on 35B-A3B)
- **Disk**: 500 GB free SSD (model weights + training artifacts)
- **Time per domain**: 2-8 hours depending on dataset size
- **Total training time**: ~50 hours for all 10 domains (sequential)

LoRA training uses MLX with Metal buffer limits:
```python
mx.set_memory_limit(460)  # GB
mx.set_cache_limit(32)    # GB
```

Peak memory usage for 35B-A3B LoRA: ~106 GB.

### Teacher (Mac Studio)

- **Model**: Qwen3-Coder-480B-A35B (1.1 TB GGUF)
- **Inference**: llama.cpp on CPU (no GPU needed)
- **Purpose**: Distillation, DPO pair generation, and bias probe judging
- **Throughput**: ~5-10 tokens/second on CPU

### Inference — Option A: MLX (Mac Studio)

- **Model**: 35B-A3B + 10 LoRA adapters
- **Memory**: 106 GB active (all 10 adapters fit with hot-swapping)
- **Throughput**: ~2-3 tokens/second (Metal GPU acceleration)

### Inference — Option B: vLLM (RTX 4090, kxkm-ai)

- **Model**: Qwen3.5-35B-A3B Q4_K_M quantized
- **Memory**: 14 GB base + 2-4 GB per active adapter (max 22 GB with 4 active)
- **Throughput**: ~30-50 tokens/second
- **GPU**: NVIDIA RTX 4090 24 GB

### Cognitive Layer (Tower)

- **Aeon Backends**: Qdrant (vector search) + Neo4j (graph)
- **Memory**: 16 GB (both services combined)
- **Disk**: 50 GB (memory palace episodes)

## Status

10 of 50 stories complete (20%):

- **Domain analysis**: 3 stories done (benchmark, dataset merge, validation)
- **Router deployment**: 2 stories done (domain classifier, multi-model router)
- **Cognitive layer**: 3 stories done (Aeon integration, Negotiator, anti-bias)
- **SNN conversion**: 1 story done (SpikingKiki-27B conversion)
- **Serving**: 2 stories done (MLX server, vLLM server)

Remaining work:

- 10 niche LoRA adapter training (stories 4-13)
- Cross-stack forgetting analysis (story 14)
- Per-domain evaluation (story 15)
- Full E2E smoke tests (stories 18-23)
- SNN evaluation & energy benchmarking (stories 26-31)
- DPO/GRPO post-training (stories 41-47)
- Fine-tuned domain embeddings (stories 48-50)

See `.ralph/prd.json` for detailed story descriptions and progress tracking. The paper outline in `docs/paper-outline-triple-hybrid.md` describes the full research contribution.

## License

Apache License 2.0. See LICENSE file for details.
