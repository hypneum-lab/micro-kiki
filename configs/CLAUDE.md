# configs/ — training recipes, curriculum, router/dispatcher

Authoritative source for every number the training and runtime pipelines consume. Changes here **are** the changes — never override these values in code.

## Layout

| Path | Owns |
|---|---|
| `stack-<NN>-<domain>.yaml` | Per-stack LoRA recipe (rank, alpha, LR, max_seq, epochs). 35 stacks, numbered in curriculum order. |
| `mlx-per-domain/<domain>.yaml` | MLX-LM per-domain training override (paths, iters, seq len) consumed by `~/KIKI-Mac_tunner`. |
| `mlx-curriculum/*.yaml` | Grouped curriculum manifests: `foundations`, `coding-core`, `coding-secondary`, `technical`, `apps`, `complements`. |
| `mlx-lora-micro-kiki.yaml`, `mlx-foundations-r32.yaml` | Top-level MLX-LM entry configs. |
| `mlx-server.json` | MLX serving config. |
| `curriculum-adaptive.json` | Adaptive curriculum state (updated by training scripts). |
| `capabilities.yaml`, `dispatcher_capabilities.yaml`, `meta_intents.yaml` | Router/dispatcher contract: 35 domains → 7 meta-intents. |
| `search_backends.yaml` | Aeon backend selection (native / Qdrant / Neo4j). |
| `micro_kiki/brainstacks.yaml`, `micro_kiki/domains.yaml` | Domain registry + ordering (mirrored in `~/KIKI-Mac_tunner/configs/micro_kiki/`). |

## Stack recipe invariants

Every `stack-NN-*.yaml` / `mlx-per-domain/*.yaml` MUST satisfy:

- `target_modules` ⊆ `{q_proj, k_proj, v_proj, o_proj}`. Never FFN / experts / router.
- `rank`: 4–16 niches, 32 foundations. `alpha = 2 * rank`. `scale = 2.0`.
- `dropout`: 0.01 (don't tune without a specs/ entry justifying it).
- Quantization: BF16 for training, Q4_K_M for inference recipes. Never below Q4.
- `max_seq_length`: 2048 niches, 4096 foundations.
- Stack number = curriculum position. Do not renumber without updating `micro_kiki/brainstacks.yaml` and the router.

## Curriculum discipline

- Order matters: foundations (01–03) first, then coding-core, secondary, technical, apps, complements.
- OPLoRA orthogonalisation kicks in from stack 04. Stacks 01–03 intentionally skip it.
- Adding a stack = new file + entry in the right `mlx-curriculum/*.yaml` + entry in `micro_kiki/brainstacks.yaml` + router retrain schedule.

## Router / dispatcher contract

- `capabilities.yaml` lists the 35 domain labels — this is the router's output dimension. If you change it, the trained router is invalidated.
- `meta_intents.yaml` has exactly 7 intent categories. Adding one is a router + downstream-contract change.
- `dispatcher_capabilities.yaml` is training-free: a YAML mapping router output → meta-intent. Pure data.

## Anti-patterns (configs-specific)

- Don't add a LoRA target outside `q_proj/k_proj/v_proj/o_proj` "just to test" — it breaks the adapter contract across every stack.
- Don't bump `rank` past 32 — memory budget on Studio is computed for this ceiling.
- Don't introduce softmax into the router recipe — 35 sigmoid outputs is load-bearing.
- Don't edit `mlx-per-domain/*.yaml` paths to point outside `~/KIKI-Mac_tunner/data/micro-kiki/` — dataset ingestion assumes that root.
- Don't hand-edit `curriculum-adaptive.json` — it's script-owned state; edit the recipe and let the script regenerate.
- Don't duplicate values between `stack-NN-*.yaml` and `mlx-per-domain/*.yaml` without keeping them in sync; the MLX file wins at training time.
