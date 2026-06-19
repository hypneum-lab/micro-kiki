# configs/mlx-per-domain/ — per-domain MLX-LM training overrides

35 single-domain YAMLs consumed by `python -m mlx_lm lora --config <file>` from
`~/KIKI-Mac_tunner/`. Override the broader `configs/mlx-curriculum/*.yaml` phase
bundles for isolated re-training (e.g. after a forgetting-gate rollback).

> **At training time the MLX file wins over `stack-NN-*.yaml`** (root `configs/CLAUDE.md`).
> Schema invariants from the parent CLAUDE.md still apply — this file documents the
> per-domain coupling, not a separate contract.

## Naming + filesystem coupling

| Edge | Rule |
|---|---|
| Filename | `<domain>.yaml`, kebab-case slug. |
| ↔ `data/corpus-real/<domain>/` | Domain rename = atomic rename across this dir + `corpus-real/` + `configs/stack-NN-<domain>.yaml` + `~/KIKI-Mac_tunner` (4 places). |
| ↔ `configs/stack-NN-<domain>.yaml` | Numbered curriculum twin in parent `configs/`. Rank/alpha/seq/iters must agree; if they diverge, the MLX file is the source of truth at training time. |
| ↔ `configs/micro_kiki/{brainstacks,domains}.yaml` | Domain registry. New file here without a registry entry = orphan. |
| ↔ `configs/capabilities.yaml` | Router output dimension — adding a domain here without bumping capabilities makes the new adapter unrouteable. |

## Per-domain profile (current canonical values)

| Profile | rank | seq_len | iters | LR | Domains |
|---|---|---|---|---|---|
| Foundations | 32 | 4096 | 1000 (chat-fr/reasoning), 500 retrain | 5.0e-5 | `chat-fr`, `reasoning` |
| Coding-core | 16 | 4096 | 500 | 5.0e-5 | `python` |
| Coding-secondary | 8–16 | 2048–4096 | 200–500 | 3.0–5.0e-5 | `cpp`, `typescript`, `rust`, `shell`, `sql`, `html-css`, `yaml-json`, `docker`, `kicad-dsl`, `spice`, `lua-upy` |
| Technical niches | 8 | 2048 | 200–500 | 3.0e-5 | `embedded`, `stm32`, `iot`, `freecad`, `platformio`, `power`, `emc`, `dsp`, `electronics`, `kicad-pcb` |
| Apps + complements | 8–16 | 2048–4096 | 200–500 | 3.0–5.0e-5 | `web-frontend`, `web-backend`, `music-audio`, `devops`, `llm-orch`, `llm-ops`, `math`, `security`, `components`, `ml-training` |

## Coupled-bump discipline (R1)

Any of `{corpus checksum, rank, alpha, target_modules, seq_len, iters, LR}` changes
forces a coupled bump:

1. New corpus version in `data/corpus-real/<domain>/SOURCES.md`.
2. Bump this YAML (commit message: `feat(domain): bump <domain> to rN/iI`).
3. Re-run forgetting gate (`src/eval/forgetting.py --stack <domain>`).
4. C-version EC bump (`STABLE → UNSTABLE` until gate passes).
5. Re-pin eval JSONL + `MODEL_CARD.md` row.

A YAML edit without steps 3–5 silently invalidates every prior R1 claim for that domain.

## Pre-pivot drift warning

Files here still encode `target_modules = q,k,v,o_proj` — pre-pivot. Root CLAUDE.md
(2026-04-18) supersedes the **module surface** to **all 17 module kinds**; when
re-training under the post-pivot rule, write the new keys directly, do not hand-edit
silently — bump the file under the coupled-bump rule above and note the pivot in the
commit. (`alpha = 2*rank` is unchanged across the pivot and stays correct — the earlier
"alpha = rank (1:1)" note was a doc error, reverted.)

## Anti-patterns

- New `<domain>.yaml` without a matching `data/corpus-real/<domain>/` dir, capabilities
  entry, and `stack-NN-<domain>.yaml` — orphan config, dead training run.
- Editing `rank`, `alpha`, or `keys` without bumping the C-version EC and re-running the
  forgetting gate — invalidates the model card and the router's calibration.
- Diverging `rank` / `alpha` / `keys` between this file and `stack-NN-<domain>.yaml`
  without flagging which one wins — by parent rule the MLX file wins, but silent drift
  produces phantom regressions.
- Pointing `data:` paths outside `~/KIKI-Mac_tunner/data/micro-kiki/` — ingestion
  assumes that root (parent rule).
- Rewriting `keys` to include MoE FFN modules (`switch_mlp.*`, `shared_expert.*`) on
  one domain only — the all-17-modules pivot is a project-wide invariant; per-domain
  module surface is not a tunable knob.
