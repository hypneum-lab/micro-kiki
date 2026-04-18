<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# configs/micro_kiki

## Purpose
Project-level configs that describe the 32-domain micro_kiki system as a whole, rather than any individual training run. `domains.yaml` is the single source of truth for what "a domain" means (keywords, regex patterns, teacher model, example target, curriculum phase, existing data sources). `brainstacks.yaml` is the legacy MoE-LoRA (custom) hyperparameter set — retained for reference and for the null-space / residual-boost pipeline in `scripts/micro_kiki/train_stack.py`, but superseded for training by the MLX-native configs in `../mlx-per-domain/` and `../mlx-curriculum/` after the 2026-04-16 pivot.

## Key Files

| File | Description |
|------|-------------|
| `domains.yaml` | 32-domain taxonomy (~14 KB). Per domain: `phase`, `teacher`, `teacher_secondary?`, `target`, `keywords[]`, `patterns[]` (regex), `existing_sources[]` (data paths). Also top-level: `version`, `target_per_domain`, `valid_ratio`, `chat_format`, `thinking_tags`. |
| `brainstacks.yaml` | Legacy MoE-LoRA config (4 experts, top-2, rank 16, rsLoRA, null-space projection rank 32, residual boost 2 rounds). Contains the canonical `curriculum:` list — 6 phases, 32 ordered domains. Still read by `scripts/micro_kiki/train_stack.py` and `null_space.py` / `residual_boost.py`. |

### `domains.yaml` schema shape

```yaml
version: 1
target_per_domain: 2000
valid_ratio: 0.1
chat_format: "messages"
thinking_tags: ["<thinking>", "</thinking>"]

domains:
  <slug>:
    phase: 1..6
    teacher: "<teacher model name>"
    teacher_secondary: "<optional fallback>"
    target: <int examples>
    keywords: ["..."]
    patterns: ["regex"]
    existing_sources: ["data/..."]
```

### `brainstacks.yaml` top-level keys

`model`, `moe_lora`, `null_space`, `residual_boost`, `training`, `forgetting`, `data`, `output`, `curriculum`.

## For AI Agents

### Working In This Directory

- `domains.yaml` slugs are load-bearing: they appear in `../mlx-per-domain/<slug>.yaml`, in `../stack-NN-<slug>.yaml`, in `../mlx-curriculum/*.yaml` `Domains:` headers, in `src/router/` tables, and on disk under `data/micro-kiki/<slug>/train.jsonl`. **Never rename a domain without grepping the whole repo.**
- Adding a new domain is a cross-cutting change: update `domains.yaml` + `brainstacks.yaml` (`curriculum:` list) + create `../mlx-per-domain/<slug>.yaml` + add to one of the `../mlx-curriculum/*.yaml` bundles + add routing weights + create `data/micro-kiki/<slug>/`.
- `teacher:` values in `domains.yaml` still reference 122B Opus for some domains; the project-level teacher post-pivot is Qwen3-Coder-480B-A35B (see `/home/kxkm/micro-kiki/CLAUDE.md`). Don't mass-rewrite — update only when you actually re-distill.
- `brainstacks.yaml`'s `curriculum:` is the master ordering. If you reorder, update `../mlx-curriculum/*.yaml` phase assignments and the header comments there.
- `thinking_tags` are emitted by teachers for CoT; stripping them is the job of data-prep, not training. Don't change tag strings.

### Testing Requirements

- `yaml.safe_load` both files and assert 32 entries in `domains.yaml:domains` and 32 entries in `brainstacks.yaml:curriculum`.
- Regex `patterns` MUST compile under Python `re` — malformed regex silently drops classification signal (`scripts/micro_kiki/classify_domains.py` logs but continues).
- `valid_ratio * target_per_domain` must be >= `val_batches` in the MLX configs (currently 25) — otherwise eval batches will be undersized.

### Common Patterns

- Slug style: lowercase kebab-case, no underscores (`kicad-pcb`, `web-frontend`, `chat-fr`).
- Phase assignments match the 6-phase curriculum:
  1. Foundations (2 domains), 2. Coding-core (4), 3. Coding-secondary (8), 4. Technical (11), 5. Apps (5), 6. Complements (2).
- `existing_sources` paths are relative to repo root (`data/...`), not absolute.

## Dependencies

### Internal
- `domains.yaml` is read by `scripts/micro_kiki/classify_domains.py`, `classify_parallel.py`, `generate_missing.py`, `split_domains.py`, `deduplicate.py`.
- `brainstacks.yaml` is read by `scripts/micro_kiki/train_stack.py` (legacy MoE-LoRA path) and `scripts/micro_kiki/train_all_stacks.sh`.
- Router configs in `src/router/` consume the same slugs.

### External
- PyYAML (`yaml.safe_load`).
- Regex patterns use Python `re` (case-insensitive substring for keywords, full `re.search` for patterns).

<!-- MANUAL: -->
