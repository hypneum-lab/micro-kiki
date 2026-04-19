<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-18 -->

# src

## Purpose
The installable Python package for micro-kiki (hatch packages `src` directly, so imports read `src.routing`, `src.memory`, etc.). It implements the full cognitive stack: base model loading, 35 LoRA stacks, meta-router + dispatcher, negotiator / anti-bias cognitive layer, Aeon memory palace, orchestrator, distillation, evaluation & forgetting, and the MLX / vLLM serving surfaces. A few loose `.h` files (`spi_mmio.h`, `uart_ring.h`) are embedded-C side-material used by hardware-oriented tests.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Package stub (one-liner docstring, post-pivot — references Qwen3.5-35B-A3B) |
| `CLAUDE.md` | Authoritative src-level style guide (imports, tensor conventions, anti-patterns) |
| `spi_mmio.h`, `uart_ring.h` | Embedded-C helpers referenced from `tests/test_uart_ring.c` |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `base/` | Qwen3.5-35B-A3B loader + DiffAttention patch (`diff_attention.py`) |
| `stacks/` | Trainer + OPLoRA init + QTHA + legacy `moe_lora.py` (attention-only LoRA) |
| `routing/` | Meta-router (sigmoid 35), dispatcher, hybrid & quantum routers |
| `orchestrator/` | Engine + HTTP bridge for request flow |
| `cognitive/` | Negotiator (CAMP+Catfish), antibias/KnowBias, RBD, forgetting_gate, judge, sleep_tagger, consolidation, bias_probe, argument_extractor |
| `memory/` | Aeon palace (Atlas SIMD + Trace graph), aeonsleep, Qdrant/Neo4j backends |
| `distill/` | Teacher client (Qwen3-Coder-480B), generator, dedup |
| `eval/` | Forgetting framework, stack_eval, MAP dispatcher/negotiator/harness benches, reward_functions |
| `compress/` | CompactifAI tensor-network compression |
| `critique/` | Best-of-N, self-refine, agentic loop, templates |
| `search/` | Exa / Semantic Scholar / docs backends + Aeon indexer + cache |
| `serving/` | MLX server, vLLM server, switchable router, Aeon hook, ANE draft/router/scorer |
| `spiking/` | SpikingBrain adapter, LIF neuron, LAS converter (v0.3 branch) |
| `ralph/` | Autonomous / research / self_review loops, forgetting_auto |

## For AI Agents

### Working In This Directory
- Follow `src/CLAUDE.md`: type hints on all public functions; `from __future__ import annotations` in every module; early returns; explicit device placement; never use `print()` (use `logging`).
- Tensor rules: BF16 training, Q4_K_M inference; annotate non-obvious shapes; never silently reshape.
- Config rules: no hardcoded paths; configs/env vars only; immutable configs (frozen dataclass or Pydantic).
- LoRA rule: when editing `stacks/` or any trainer, target ONLY `self_attn.{q,k,v,o}_proj`. Never touch MoE FFN keys — the routing is already learned.
- Do NOT import mlx unconditionally at package level — keep it behind the `mlx` extra guard so vLLM-only installs work.

### Testing Requirements
`tests/` mirrors this layout one-to-one (`tests/routing/`, `tests/cognitive/`, `tests/memory/`, etc.). Every new module gets a smoke test. Mock heavy objects — never load the real 35B in unit tests.

### Common Patterns
- Router output: 35 domain sigmoids (34 niches + 1 base); capability sigmoids (5) are served separately (see `configs/capabilities.yaml`).
- Dispatcher is training-free: YAML mapping from router logits to 7 meta-intents (`quick-reply`, `reasoning`, `coding`, `creative`, `research`, `agentic`, `tool-use`).
- Forgetting gate: angle between base and adapted gradient subspaces + win-rate on held-out; rollback if `angle_deg < 30 AND win_rate_delta < -0.03`.
- Teacher client targets local MLX 4bit 480B — no network dependency expected.

## Dependencies

### Internal
Self-contained package; subpackages import each other (e.g. `cognitive/` uses `memory/` Aeon, `orchestrator/` composes routing+cognitive+serving).

### External
`torch`, `transformers`, `peft`, `trl`, `accelerate` (train extra); `mlx`, `mlx-lm` (mlx extra); `vllm`, `fastapi`, `uvicorn` (serve extra); `httpx`, `beautifulsoup4` (agentic). Core always-on: `httpx`, `pyyaml`, `numpy`, `huggingface-hub`.

<!-- MANUAL: -->
