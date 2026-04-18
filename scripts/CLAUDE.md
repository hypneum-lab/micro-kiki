# scripts/ — CLIs for training, distillation, dataset gen, eval

All scripts are `uv run python scripts/<name>.py` entry points. **No business logic here** — scripts orchestrate code from `src/` and configs from `configs/`. If you find yourself writing algorithms in a script, move them to `src/`.

## Taxonomy (what to reach for)

| I want to… | Reach for |
|---|---|
| Distill a domain from the teacher (Qwen3-Coder-480B) | `distill_domain.py` |
| Generate a domain-specific dataset from scratch | `gen_<domain>_dataset.py` (one per domain: spice, emc, kicad, platformio, web_frontend, …), or `gen_remaining_datasets.py` for batch |
| Expand prompts / augment data | `expand_prompts_x25.py`, `generate_dpo_pairs.py`, `gen_dpo_pairs.py` |
| Filter / dedupe / merge raw data | `dataset_quality_filter.py`, `merge_all_sources.py`, `merge_datasets.py`, `merge_domains.py`, `import_kiki_datasets.py` |
| Scrape public Q&A | `scrape_stackexchange.py` |
| Train something | `train_stack.py` (generic), `train_lora_all.sh`, `train_niches_mlxtune.py`, `train_dpo_niches.py`, `train_grpo_niches.py`, `train_router.py`, `train_embeddings.py`, `train_vqc_router.py`, `train_with_restart.{py,sh}` |
| Convert / quantize an artifact | `convert_gguf.py`, `convert_spikingkiki_35b.py`, `quantize_spikingbrain.py` |
| Run the full eval / forgetting gate | `run_full_eval.sh`, `run_forgetting.sh`, `eval_niche_vs_base.py`, `eval_v2_v3.py`, `benchmark_base_vs_lora.py`, `group_eval.py`, `eval_aeon*.py`, `eval_base_knowbias.py` |
| Benchmark a router variant | `benchmark_quantum_router.py` |
| Run a POC pipeline end-to-end | `poc_micro_kiki.py`, `poc_pipeline.py`, `poc_pipeline_v2.py` |
| Release an adapter to HF | `release_hf.py` |
| Orchestrate a run on a remote node | `orchestrate_remote.sh` |
| Classify raw data into domains (parallel) | `micro_kiki/classify_parallel.py`, `micro_kiki/classify_domains.py` |

## Where training actually runs

- **Adapter training (per-stack)**: **Mac Studio** via MLX-LM from `~/KIKI-Mac_tunner/`. The `train_*.py` scripts here are either the MLX-tune variant (`train_niches_mlxtune.py`) or non-MLX experiments (DPO/GRPO on kxkm-ai).
- **Router / embeddings / VQC**: runnable locally.
- **Distillation**: teacher is the local 480B on Studio (no network). `distill_domain.py` talks to it over the MLX server.

## Discipline

- Every script must accept `--config <path>` or read `configs/`. No hardcoded paths.
- Long-running scripts write artifacts to `checkpoints/`, `output/`, `outputs/`, or `results/` — not to `src/` or `configs/`.
- Before running a dataset-gen script, check that the corresponding `configs/mlx-per-domain/<domain>.yaml` exists.
- Before training a stack, verify the dataset is classified + deduped — the training scripts **do not** dedupe.
- After training a stack, the forgetting gate is not automatic from these scripts — run `run_forgetting.sh` / `eval_niche_vs_base.py` explicitly.

## Anti-patterns (scripts-specific)

- Don't put model/algorithm code in a script — import it from `src/`.
- Don't train on kxkm-ai from a `train_*.py` here for the 35B base — it won't fit. DPO/GRPO niche experiments on small heads are the only kxkm-ai exceptions.
- Don't call `gen_<domain>_dataset.py` and feed its output straight into training — run `dataset_quality_filter.py` + dedup first.
- Don't hand-edit the JSON/YAML state written by these scripts (e.g. `configs/curriculum-adaptive.json`) — rerun the owning script instead.
- Don't skip `--require-verify` defaults on `distill_teacher.py` without reading the known-gotcha note (it drops records silently).
- Don't parallelise stack training across machines — stacks interfere.
