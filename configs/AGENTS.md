<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# configs

## Purpose
All declarative configuration for micro-kiki. Two layers: (1) MLX LoRA training configs — one per domain plus curriculum batches and a rank-32 foundations variant; (2) runtime configs for the cognitive stack — the 32-domain -> 7 meta-intent map, capability flags (web search, self-critique levels, deep eval), search backend credentials/caches, and `micro_kiki/` brainstacks + domain definitions consumed by the data pipeline. No code; everything here is loaded by `src/routing`, `src/orchestrator`, `src/search`, and the MLX trainers.

## Key Files
| File | Description |
|------|-------------|
| `meta_intents.yaml` | Domain indices 0-31 -> 7 meta-intents (quick-reply, reasoning, coding, creative, research, agentic, tool-use) |
| `capabilities.yaml` | 5 capability outputs at indices 32-36 (web_search, self_critique_{token,response,task}, deep_eval), per-capability thresholds, best-of-N confidence bands, agentic loop max iterations |
| `dispatcher_capabilities.yaml` | Maps capability-flag combinations to orchestrator actions (search_and_respond, critique_and_respond, agentic_task, full_agentic, evaluate) |
| `search_backends.yaml` | Exa / Semantic Scholar / docs-scraper config, SQLite cache paths (`data/search_cache.sqlite`, `data/docs_index.sqlite`) |
| `curriculum-adaptive.json` | Adaptive curriculum scheduling state |
| `mlx-foundations-r32.yaml` | Rank-32 foundations config (alpha 64, scale 2.0, attention-only keys, 500 iters @ seq 4096) |
| `mlx-lora-micro-kiki.yaml` | Base MLX LoRA template |
| `mlx-server.json` | MLX serving runtime config |
| `stack-01-chat-fr.yaml` ... `stack-32-security.yaml` | One per-domain LoRA config (rank 16, alpha 32, LR 2e-4, 3 epochs, seq 4096, attention-only) |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `micro_kiki/` | Brainstacks + domain definitions for the data pipeline (`brainstacks.yaml`, `domains.yaml`) — 32 domains with keywords, regex patterns, teachers, existing HF sources |
| `mlx-curriculum/` | Curriculum batch configs: `foundations.yaml`, `coding-core.yaml`, `coding-secondary.yaml`, `technical.yaml`, `apps.yaml`, `complements.yaml` |
| `mlx-per-domain/` | Per-domain MLX configs for all 32 stacks (chat-fr, reasoning, python, typescript, cpp, rust, html-css, shell, sql, yaml-json, docker, kicad-dsl, spice, lua-upy, embedded, stm32, iot, freecad, platformio, power, emc, dsp, spice-sim, electronics, kicad-pcb, web-frontend, web-backend, music-audio, devops, llm-orch, math, security) |

## For AI Agents

### Working In This Directory
- Every stack YAML MUST target only attention keys: `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`. NEVER add MoE FFN keys (`gate_proj`, `up_proj`, `down_proj`, `experts.*`) — the MoE routing is already learned and editing it breaks the model.
- Rank policy: niches 4-16, foundations 32 (`mlx-foundations-r32.yaml`). `alpha = 2 * rank`, `scale = 2.0`.
- LR: 2e-5 to 5e-5 on foundations; 2e-4 works for niche stacks but higher values diverge at 35B (validated ceiling).
- Seq length: 2048 for niches, up to 4096 for foundations.
- `meta_intents.yaml` — each of the 32 domains must appear in exactly ONE intent bucket (the file enforces this by construction).
- When adding a new capability, update BOTH `capabilities.yaml` (index + threshold) AND `dispatcher_capabilities.yaml` (action mapping). `total_outputs = num_domain_outputs + num_capability_outputs`.
- Do NOT put secrets inline — search backends use `api_key_env` env vars (e.g. `EXA_API_KEY`).
- `grad_checkpoint: true` is mandatory in every training config (74 GB Metal envelope is tight).

### Testing Requirements
- `tests/test_phase_v_configs.py`, `tests/test_phase_vi.py`, `tests/test_phase_vii.py` load these YAMLs and assert invariants (intent coverage, key restriction, threshold bounds). Update these tests when changing structure.
- `tests/test_dispatcher.py` exercises `meta_intents.yaml`.
- `tests/routing/test_router_37.py` validates the 32+5 capability shape.

### Common Patterns
- YAML over JSON for anything human-edited (training, routing). JSON only for machine-generated (`curriculum-adaptive.json`) and runtime server (`mlx-server.json`).
- Paths are relative to repo root (`data/distilled/<domain>.jsonl`, `data/search_cache.sqlite`).
- Domain-index stability: once a domain has an index, it is locked. New domains get the next free index; never renumber.

## Dependencies

### Internal
Consumed by `src.routing.dispatcher.load_intent_mapping`, `src.routing.model_router`, `src.search.*`, `src.serving.mlx_server`, and MLX trainers in `scripts/` + `~/KIKI-Mac_tunner`.

### External
- `mlx-lm` reads the `mlx-*.yaml` schema.
- `vllm` serving reads `mlx-server.json` analog at runtime.

<!-- MANUAL: -->
