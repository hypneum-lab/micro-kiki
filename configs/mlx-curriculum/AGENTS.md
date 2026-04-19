<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# configs/mlx-curriculum

## Purpose
Phase-bundle MLX LoRA configs — one YAML per curriculum phase, grouping the domains that share the same rank/LR/sequence-length profile. These are the canonical recipes for sequential sweep training (Phase I → II → III → IV → V → VI), used by `scripts/micro_kiki/train_all_stacks.sh` and the `scripts/train_niches_mlxtune.py` batch runner. The per-domain YAMLs in `../mlx-per-domain/` mirror the fields here; edit both in lockstep when tuning a tier.

## Key Files

| File | Phase | Rank / Alpha | Seq / Iters / LR | Domains |
|------|-------|--------------|------------------|---------|
| `foundations.yaml` | I | 16 / 32 | 4096 / 500 / 5e-5 | chat-fr, reasoning |
| `coding-core.yaml` | II | 12 / 24 | 4096 / 400 / 3e-5 | python, typescript, cpp, rust |
| `coding-secondary.yaml` | III | — | — | html-css, shell, sql, yaml-json, docker, kicad-dsl, spice, lua-upy |
| `technical.yaml` | IV | 4 / 8 | 2048 / 200 / 2e-5 | embedded, stm32, iot, freecad, platformio, power, emc, dsp, spice-sim, electronics, kicad-pcb |
| `apps.yaml` | V | — | — | web-frontend, web-backend, music-audio, devops, llm-orch |
| `complements.yaml` | VI | — | — | math, security |

Schema (shared):

```yaml
model: models/qwen3.5-35b-a3b
fine_tune_type: lora
lora_parameters: { dropout, scale: 2.0, keys: [q/k/v/o_proj], rank, alpha }
num_layers: 40
grad_checkpoint: true
save_every / steps_per_report / steps_per_eval / val_batches
seed: 42
learning_rate / max_seq_length / batch_size / grad_accumulation_steps / iters
lr_schedule:
  name: cosine_decay
  warmup: <~10% of iters>
  arguments: [<iters>, 1.0e-06]
```

## For AI Agents

### Working In This Directory

- These are **bundles** — the covered domains are listed in the file header comment (`# Domains: ...`). Keep that comment in sync with any change to which domains use the bundle.
- `warmup` is normally 10% of `iters`. The cosine-decay `arguments` end value (`1.0e-06`) is the minimum LR at schedule end — don't set it above `learning_rate`.
- Changing the rank tier means also changing the `../mlx-per-domain/<domain>.yaml` for every domain in the bundle — the per-domain files are authoritative at training time.
- The technical phase uses `batch_size: 2, grad_accumulation: 2` (eff. 4) at seq 2048; the longer-context phases use `batch_size: 1, grad_accumulation: 4` (eff. 4) at seq 4096. Don't blindly copy batch/accumulation between phases.

### Testing Requirements

- `yaml.safe_load` must succeed; `scripts/micro_kiki/train_all_stacks.sh` parses and validates before dispatching.
- Dry-run the bundle with `iters: 10` on a single domain from the list before launching the full sweep.
- After each phase completes, run the forgetting gate for every prior stack (sequential curriculum invariant).

### Common Patterns

- Header comment convention:
  ```
  # micro-kiki MLX LoRA — <phase-name>
  # Domains: <comma-separated list>
  # Rank N, seq L, I iters
  ```
- LR decays with phase depth: foundations 5e-5 → coding 3e-5 → niches 2e-5. This is intentional: foundations carry the heaviest gradient signal; niches want a gentler touch to avoid overriding prior stacks.

## Dependencies

### Internal
- Driven by `scripts/micro_kiki/train_all_stacks.sh` and related orchestrators in `scripts/micro_kiki/`.
- Per-domain overrides live in `../mlx-per-domain/`.
- Curriculum ordering is duplicated in `../micro_kiki/brainstacks.yaml` (`curriculum:` list) — keep both lists consistent.

### External
- `mlx-lm` lora entry-point.
- Base weights at `models/qwen3.5-35b-a3b` (BF16).

<!-- MANUAL: -->
