<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# scripts

## Purpose
Operator-facing command-line drivers for the full micro-kiki lifecycle: downloading the base model, dataset curation & distillation, per-stack training (MLX on Mac, small GPU variants on 4090), DPO/GRPO refinement on niche domains, router training, evaluation benches, forgetting checks, and POC pipelines. These scripts wrap `src/*` modules and are meant to be run with `uv run python scripts/<name>.py`. Subdirectory `micro_kiki/` houses the legacy / research-oriented shell + Python pipeline that predates the MLX pivot.

## Key Files
| File | Description |
|------|-------------|
| `download_base.py` | Fetch `Qwen/Qwen3.5-35B-A3B` (or fallback) from HF |
| `train_stack.py`, `train_stack_kxkm.py`, `train_stack02.py`, `train_stack03.py` | Per-stack trainers (standard LoRA, attention-only) |
| `train_niches_mlxtune.py` | MLX per-domain niche trainer (ranks 4-16) |
| `train_dpo_niches.py`, `train_grpo_niches.py` | Preference-tuning on niche domains |
| `train_router.py`, `train_router_v0.py`, `train_router_kxkm.py`, `train_vqc_router.py` | Sigmoid meta-router trainers (37 outputs) |
| `train_embeddings.py`, `train_forgetting_gate.py`, `train_qtha_stack.py` | Auxiliary trainers |
| `distill_domain.py`, `distill_fast.py`, `distill_niche.py` | Teacher distillation via Qwen3-Coder-480B local MLX |
| `generate_dataset_codex.py`, `generate_dataset_mcp.py`, `generate_bias_pairs.py`, `generate_dpo_pairs.py`, `expand_prompts_x25.py` | Synthetic data generators |
| `merge_datasets.py`, `merge_all_sources.py`, `dataset_quality_filter.py`, `curate_bias_dataset.py`, `scrape_stackexchange.py`, `import_kiki_datasets.py`, `import_kiki_datasets.sh` | Dataset curation |
| `eval_aeon.py`, `eval_base_knowbias.py`, `eval_niche_vs_base.py`, `group_eval.py`, `energy_bench.py`, `benchmark_base_vs_lora.py`, `benchmark_quantum_router.py` | Evaluation benches |
| `run_eval_stack01.py`, `run_eval_stack01_fast.py`, `run_eval_v2.py`, `run_eval_mini.py`, `run_eval_3.py` | Targeted eval runners |
| `run_full_eval.sh`, `run_forgetting.sh`, `run_pipeline.sh` | Shell orchestrators |
| `smoke_e2e.py`, `smoke_test_e2e.py`, `smoke_spikingbrain.py` | Smoke flows |
| `poc_micro_kiki.py`, `poc_pipeline.py`, `poc_pipeline_v2.py`, `e2e_final.py` | End-to-end POCs (v2 = VQC router + negotiator + 10 domains) |
| `aeon_compress_daemon.py` | Long-running compress loop (systemd unit available) |
| `release_hf.py` | Push adapters / merged model to HuggingFace |
| `convert_spikingkiki_35b.py`, `probe_spikingbrain_hf.py`, `quantize_spikingbrain.py`, `fork_qwen_diffattn.py` | v0.3 neuroscience branch tooling |
| `orchestrate_remote.sh`, `setup_neuro_env.sh` | Remote orchestration & environment bootstrap |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `micro_kiki/` | Legacy / research pipeline (pre-MLX pivot): `classify_domains.py`, `deduplicate.py`, `eval_stack.py`, `moe_lora.py`, `null_space.py`, `residual_boost.py`, `split_domains.py`, `pipeline_data.sh`, `train_all_stacks.sh` |

## For AI Agents

### Working In This Directory
- Scripts are run with `uv run python scripts/<name>.py` from repo root. Absolute paths only.
- Training scripts here are for non-MLX pathways (4090 prototyping, router training). Full 35B LoRA training happens in `~/KIKI-Mac_tunner` via `python -m mlx_lm lora --config configs/...`, not here.
- NEVER run `train_stack*.py` against the full 35B on kxkm-ai — it will OOM.
- Set `UNSLOTH_COMPILE_DISABLE=1` before any trainer that uses unsloth kernels.
- `release_hf.py` writes to clemsail's HF org — double-check the repo name before pushing.
- After every stack, run `run_forgetting.sh` (rollback if angle < 30 AND win-rate drop > 0.03).
- `aeon_compress_daemon.py` is long-running — use the systemd unit in `deploy/systemd/aeon-compress.service`, not a bare `&`.

### Testing Requirements
Several scripts have counterparts in `tests/`:
- `generate_dataset_codex.py` -> `test_generate_codex.py`
- `dataset_quality_filter.py` -> `test_quality_filter.py`
- `scrape_stackexchange.py` -> `test_scrape_stackexchange.py`
- `convert_spikingkiki_35b.py` -> `test_convert_spikingkiki.py`

Add a `tests/test_<script>.py` when logic is non-trivial (parsing, filtering, reward shaping). Don't test shell wrappers.

### Common Patterns
- CLI parsing with `argparse`; `--domain <name>` + `--prior-stacks <list>` is the common forgetting-check signature.
- Reads `configs/*.yaml` for stack/meta-intent/capability definitions.
- Writes artifacts into `outputs/` (training), `results/` (eval), `data/` (datasets).
- Shell scripts set `set -euo pipefail` and source the correct venv (`~/KIKI-Mac_tunner/.venv` on Mac, local `.venv` on kxkm-ai).

## Dependencies

### Internal
Uses nearly every `src/*` subpackage: `src.base.loader`, `src.distill.*`, `src.eval.forgetting`, `src.routing.*`, `src.cognitive.*`, `src.stacks.*`, `src.memory.*`, `src.spiking.*`.

### External
Depends on `train`, `mlx`, `serve`, and `agentic` extras depending on the script. Shell scripts also call `python -m mlx_lm lora` (requires `mlx-lm>=0.30`).

<!-- MANUAL: -->
