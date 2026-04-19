<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# configs/mlx-per-domain

## Purpose
Per-domain MLX LoRA training configs consumed by `python -m mlx_lm lora --config <file>`. One YAML per expert domain (32 total, one per brainstack). Each file pins the exact rank/alpha/seq_len/iters/LR profile that `scripts/micro_kiki/train_stack.py` and the `train_niches_mlxtune.py` / `train_stack.py` MLX runners expect. These files override the broader `mlx-curriculum/*.yaml` phase bundles when you need to train (or re-train) a single domain in isolation, e.g. after a forgetting-check rollback.

## Key Files

Naming convention: `<domain>.yaml`, where `<domain>` matches the slug in `configs/micro_kiki/domains.yaml` and the `stack-NN-<domain>.yaml` files in the parent `configs/` directory.

| File | Description |
|------|-------------|
| `chat-fr.yaml` | Foundation domain, rank 32 / alpha 64 / seq 4096 / 500 iters, LR 5e-5. |
| `reasoning.yaml` | Foundation domain, rank 32 (paired with chat-fr for Phase I). |
| `python.yaml` | Coding-core, rank 16 / alpha 32 / seq 4096 / 500 iters. |
| `math.yaml`, `security.yaml` | Complements (Phase VI), rank 16. |
| `embedded.yaml`, `stm32.yaml`, `iot.yaml`, `freecad.yaml`, `platformio.yaml`, `power.yaml`, `emc.yaml`, `dsp.yaml`, `spice-sim.yaml`, `electronics.yaml`, `kicad-pcb.yaml` | Technical niches (Phase IV), rank 4 / alpha 8 / seq 2048 / 200 iters. |
| `cpp.yaml`, `typescript.yaml`, `rust.yaml`, `html-css.yaml`, `shell.yaml`, `sql.yaml`, `yaml-json.yaml`, `docker.yaml`, `kicad-dsl.yaml`, `spice.yaml`, `lua-upy.yaml` | Other coding domains. |
| `web-frontend.yaml`, `web-backend.yaml`, `music-audio.yaml`, `devops.yaml`, `llm-orch.yaml` | Applications (Phase V). |

All 32 share the same top-level schema:

```yaml
model: models/qwen3.5-35b-a3b
fine_tune_type: lora
grad_checkpoint: true
save_every: 100
steps_per_report: 10
steps_per_eval: 100
val_batches: 25
train: true
seed: 42
clear_cache_threshold: 50000000000
lora_parameters:
  rank: <4 | 16 | 32>
  alpha: <2 * rank>
  dropout: 0.0
  scale: 2.0
  keys:
    - self_attn.q_proj
    - self_attn.k_proj
    - self_attn.v_proj
    - self_attn.o_proj
num_layers: 40
learning_rate: <2e-5 | 5e-5>
batch_size: <1 | 2>
grad_accumulation_steps: <2 | 4>
iters: <200 | 400 | 500>
max_seq_length: <2048 | 4096>
```

## For AI Agents

### Working In This Directory

- **Never** add MoE FFN layers to `lora_parameters.keys` — LoRA targets attention projections only (q/k/v/o). See `/home/kxkm/micro-kiki/CLAUDE.md` "Don't".
- Rank budget per tier is fixed: foundations = 32, coding-core = 12–16, niches = 4. Bumping rank on a niche doubles memory and is usually unnecessary.
- `alpha` MUST equal `2 * rank` (scale 2.0 convention). If you change one, change the other.
- `model:` is a relative path from the Mac Studio working dir (`~/micro-kiki/` or `~/KIKI-Mac_tunner/`) — don't absolutize.
- When adding a 33rd domain (don't, but if you must): also add it to `configs/micro_kiki/domains.yaml`, `configs/micro_kiki/brainstacks.yaml` `curriculum:` list, the matching `configs/mlx-curriculum/<phase>.yaml` bundle, and `src/router/` routing tables.
- After editing a config, re-run the matching train step — don't assume paused training reloads YAML.

### Testing Requirements

- YAML must parse under `yaml.safe_load`. A lightweight schema check lives in `scripts/micro_kiki/train_stack.py` (it will raise on missing keys).
- Before a full training run, do a dry-run with `iters: 10` to confirm data path + tokenizer load.
- Forgetting check (`uv run python -m src.eval.forgetting`) is mandatory after every single-domain run — rollback if angle < 30° AND win-rate drop > 0.03.

### Common Patterns

- Comment-first header: `# <domain> — rank N, seq L, I iters` — one line, keeps grep-friendly inventory.
- Keys in this fixed order: `model`, `fine_tune_type`, `grad_checkpoint`, schedule flags, `seed`, `clear_cache_threshold`, `lora_parameters`, `num_layers`, `learning_rate`, `batch_size`, `grad_accumulation_steps`, `iters`, `max_seq_length`.
- `num_layers: 40` is the Qwen3.5-35B-A3B transformer depth — don't change unless the base model swaps.

## Dependencies

### Internal
- Consumed by `scripts/micro_kiki/train_stack.py` and `scripts/train_niches_mlxtune.py` (single-domain runs).
- Cross-referenced by `configs/micro_kiki/brainstacks.yaml` (`curriculum:` list) and `configs/micro_kiki/domains.yaml` (data source mapping).
- Mirrors and overrides the phase bundles in `configs/mlx-curriculum/`.

### External
- `mlx-lm` LoRA runner (`python -m mlx_lm lora --config <file>`).
- Base weights at `models/qwen3.5-35b-a3b` (BF16, ~65 GB on disk).

<!-- MANUAL: -->
