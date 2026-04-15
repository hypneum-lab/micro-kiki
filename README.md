# micro-kiki

32 domain experts (MoE-LoRA) stacked on Qwen3.5-4B base. Fits on RTX 4090 24 GB. Distilled from Mistral-Large-Opus and Qwen3.5-122B teachers.

## What

- **Base**: Qwen3.5-4B (Apache 2.0, GatedDeltaNet hybrid, 262K ctx, 201 languages, thinking mode)
- **Adapters**: 32 MoE-LoRA stacks (4 experts × rank 16 each), ~150 MB per stack, ~4.8 GB total
- **Router**: Sigmoid meta-router (32 outputs, 2 M params, ~5 ms overhead)
- **Budget**: ~4-8 GB VRAM at inference (base Q4 + 2-4 active stacks + KV cache)

## Domains (32)

Organized in 6 curriculum phases: foundations (chat-fr, reasoning) → coding core → coding secondary → technical (embedded, STM32, KiCad, SPICE, EMC, DSP, power, RF) → applications (web, music, devops, LLM orch) → complements (math, security).

Full list: see `docs/specs/2026-04-15-micro-kiki-design.md`.

## Teachers

- Mistral-Large-Opus (123B, Studio) — reasoning, general
- Qwen3.5-122B-A10B Opus-v3 (Studio) — same-family distillation
- Qwen3.5-35B-A3B Opus (kxkm-ai) — fast teacher
- Devstral-v3/v4 Opus (kxkm-ai) — coding-specific

## Hardware

| Machine | Role |
|---------|------|
| Mac Studio M3 Ultra 512 GB | Teacher serving + primary training |
| RTX 4090 24 GB (kxkm-ai) | Inference + Unsloth training |

## Structure

```
micro-kiki/
├── docs/
│   ├── specs/       # Design documents
│   ├── research/    # MoE research, teacher comparisons
│   └── plans/       # Implementation plans
├── src/
│   ├── base/        # Base model loading, quantization
│   ├── stacks/      # MoE-LoRA stack training + management
│   ├── routing/     # Meta-router training + inference
│   ├── distill/     # Teacher query + dataset generation
│   ├── eval/        # Per-stack + end-to-end evaluation
│   └── serving/     # vLLM / llama-server / mlx-lm deployment
├── configs/         # Per-stack training configs (YAML)
├── data/            # Raw + processed datasets (gitignored)
├── scripts/         # Orchestration scripts
└── tests/           # Unit + integration tests
```

## Status

- [x] Design (2026-04-15) — see `docs/specs/`
- [x] MoE approach research — see `docs/research/`
- [ ] Implementation plan (next)
- [ ] Base model setup
- [ ] Data pipeline for domain 1 (chat-fr)
- [ ] First stack trained
- [ ] Meta-router v0
- [ ] 32 stacks complete
- [ ] Triple pipeline (ANE + GPU + CPU on Mac)

## License

Apache 2.0. Base model (Qwen3.5-4B) is also Apache 2.0.

## Related

Part of the KIKI family:
- [KIKI-Mac_tunner](https://github.com/L-electron-Rare/KIKI-Mac_tunner) — MLX fine-tuning toolkit
- [KIKI-models-tuning](https://github.com/L-electron-Rare/KIKI-models-tuning) — Unsloth/LoRA training registry
