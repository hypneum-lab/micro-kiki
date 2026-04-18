<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# stacks — Per-domain LoRA adapter kernel

## Purpose
Holds the adapter math, config scaffolding, and training entry point for micro-kiki's 32 domain experts on Qwen3.5-35B-A3B (native MoE, 256 experts / 3B active). Post-2026-04-16 pivot, the *production* training path runs via MLX on Mac Studio using the KIKI-Mac_tunner pipeline (see `src/stacks/CLAUDE.md`), but this module still owns: the in-process PyTorch `StackTrainer` used for local smoke tests and the pre-pivot v0.2 recipe; the OPLoRA orthogonal-projection init for forgetting prevention at stacks ≥ 04; a QTHA experimental compression adapter targeted at the attention projections; and the legacy MoLoRA reference implementation. All four `.py` files are real, tested code — not stubs. See `CLAUDE.md` in this directory for the locked LoRA recipe (rank 4-32 by domain, alpha = 2×rank, LR 2e-5→5e-5, seq 2048/4096, attention-only targets).

## Key Files (verified line counts: `wc -l src/stacks/*.py` → 0 + 64 + 23 + 29 + 160 = 276 lines)
| File | Description |
|------|-------------|
| `__init__.py` | 0 lines (truly empty — the only genuine no-content file here). Package marker only. |
| `trainer.py` | **160 lines, fully implemented.** Not a stub — earlier AGENTS.md was wrong. Defines `LoRAConfig` (frozen dataclass: `rank=16`, `alpha=32`, `target_modules=("q_proj","k_proj","v_proj","o_proj")`, `dropout=0.0`), module constants `REQUIRED_KEYS` and `DEFAULT_BASE="Qwen/Qwen3.5-35B-A3B"`, helpers `load_training_config(path)` (yaml + required-key check) and `lora_config_from_dict(cfg)`. `StackTrainer` class with `__init__`, classmethod `from_config(path)`, and `train(dataset_path)` that: sets `UNSLOTH_COMPILE_DISABLE=1`, loads `AutoModelForCausalLM` in `torch.bfloat16` with `device_map="auto"`, applies PEFT `LoraConfig(init_lora_weights=...)`, loads the dataset via `datasets.load_dataset("json", ...)`, and runs a full `trl.SFTTrainer` loop with gradient checkpointing + `bf16=True`. Returns a `metrics` dict with `train_loss`, `output_dir`, `epochs`, `base_model`, `lora_rank`. Used by `.claude/plans/micro-kiki-v0.2-implementation.md` for the legacy per-stack CLI (`uv run python -m src.stacks.trainer --config configs/stack-NN-*.yaml`). Covered by `tests/test_trainer.py` (`TestStackTrainer.test_init`, `test_from_config`). |
| `oplora.py` | 23 lines. Orthogonal Projection LoRA init (arxiv 2510.13003). `orthogonal_projection(prior_subspace, dim)` builds `I - QQ^T` from the prior stack's subspace via QR in float32 then casts back to `prior_subspace.dtype`. `init_oplora_experts(in_features, rank, num_experts, prior_subspace=None)` returns a list of projected Gaussian-init `A` matrices (`std=0.01`); passes through unprojected when `prior_subspace is None` (stacks 01-03). Covered by `tests/test_oplora.py` (orthonormality + projection-correctness assertions). Not yet wired into the MLX pipeline. |
| `qtha.py` | 29 lines. Quantum-inspired Tensor Hybrid Adapter scaffolding. `QTHAConfig(bond_dim=8, target_modules=[…])` — frozen dataclass with `__post_init__` + `object.__setattr__` default `["q_proj","k_proj","v_proj","o_proj"]`. `estimate_qtha_params(hidden_dim, bond_dim, num_layers, num_modules=4)` returns the per-layer × module parameter cost (~500K for rank-16-equivalent coverage vs ~2M for MoLoRA). Covered by `tests/test_qtha.py` and `tests/compress/test_compactifai.py`. The full tensor-network forward pass is not yet implemented — only the config + estimator — but the file is not dead: compression tests and docs reference it by name. |
| `moe_lora.py` | 64 lines. Legacy MoLoRA (arxiv 2603.15965) reference implementation. **Fully functional, not a header stub** — earlier AGENTS.md was wrong. `MoLoRAConfig` (frozen dataclass) + `_get_layer_class()` (lazy-imports torch and returns a real `nn.Module` with gating, top-K expert routing, `nn.ParameterList` of `lora_a`/`lora_b`, softmax-weighted expert sum) + factory `MoLoRALayer(in_features, out_features, config)`. Covered by `tests/test_moe_lora.py` (config defaults, forward shape, gating correctness). **Status post-pivot**: no longer the production path — `src/stacks/CLAUDE.md` marks it superseded by standard PEFT LoRA on Qwen3.5-35B-A3B. Still referenced by `.claude/plans/micro-kiki-v0.2-implementation.md` and `docs/superpowers/plans/2026-04-16-phases-i-iii-foundations.md` (the v0.2 test plan), and the test module is kept green. Treat as archival-but-alive: do not extend, but do not delete until the v0.2 plan is fully retired. |

## For AI Agents

### Working In This Directory
- **Trust `wc -l`, not earlier doc claims.** The previous AGENTS.md called `trainer.py` a "docstring stub" and `moe_lora.py` a "header only"; both are false. Verify with `wc -l /home/kxkm/micro-kiki/src/stacks/*.py` before any deletion.
- **LoRA targets are attention only**: `q_proj`, `k_proj`, `v_proj`, `o_proj`. Project `Don't`: *don't LoRA-tune MoE FFN layers — the MoE routing is already learned* (root `CLAUDE.md`). `LoRAConfig.target_modules` and `QTHAConfig.target_modules` both encode this default.
- **Sequential stacks only**: *don't train stacks in parallel (interference)*. Run forgetting check after EACH stack — see `src/eval/forgetting.py` and the 30° / 0.03 rollback rule.
- **OPLoRA activates at stack 04+**: earlier stacks have no prior subspace to project against. Pass `prior_subspace=None` for stacks 01-03; pass the stacked `A` matrices of previous stacks for 04+. Currently pure PyTorch math — not yet wired into the MLX training loop on Mac Studio.
- **No QLoRA / BitsAndBytes on this base**: project `Don't` — known issues with MoE layers. Train in BF16 (~106 GB peak on 35B-A3B), export to Q4_K_M for serving.
- **Set `UNSLOTH_COMPILE_DISABLE=1`** before any training call (MoE mixed-precision kernel fix, per project `CLAUDE.md`). `StackTrainer.train` sets this via `os.environ.setdefault` — do not remove that line.
- **`moe_lora.py` is archival, not dead.** It is the reference implementation for the v0.2 MoE-LoRA path and remains the backing code for `tests/test_moe_lora.py`. "Archival" ≠ "deletable" — removing it red-lines a test module and breaks two plan documents. If you need a new adapter pattern, add it as a new file.
- **`trainer.py` is the in-process PyTorch path.** Production training runs on Mac Studio via `mlx_lm lora` (see `src/stacks/CLAUDE.md`); the in-module `StackTrainer.train` is used for local validation on smaller bases and for the v0.2 per-stack CLI invocations listed in `.claude/plans/micro-kiki-v0.2-implementation.md`. Both paths share the same `LoRAConfig` contract — keep them in sync.

### Testing Requirements
- `tests/test_trainer.py` — `StackTrainer` init/`from_config` smoke tests.
- `tests/test_oplora.py` — QR orthonormality (`Q^T Q ≈ I`), projected vectors have zero component in prior subspace, `prior_subspace=None` pass-through.
- `tests/test_qtha.py` + `tests/compress/test_compactifai.py` — `QTHAConfig` defaults, `estimate_qtha_params` math (must match any new closed-form change exactly).
- `tests/test_moe_lora.py` — `MoLoRAConfig` defaults, `MoLoRALayer` forward shape, gating / top-K correctness. Keep green even post-pivot.
- Forgetting regression: `tests/eval/test_forgetting.py` covers the 30° / 0.03 rollback gate — run after any change to OPLoRA init logic.

### Common Patterns
- `@dataclass(frozen=True)` with `__post_init__` + `object.__setattr__` for mutable default lists (see `QTHAConfig.target_modules`).
- Float32 compute, cast back to `prior_subspace.dtype` on return in OPLoRA — avoids BF16 QR instability.
- Lazy torch import via a nested `_get_layer_class()` factory in `moe_lora.py` — keeps the module importable without torch at collection time.
- Rank is a per-domain scalar (niches 4-16, foundations 32); alpha is always `2 * rank`. `LoRAConfig` default is `rank=16, alpha=32` which matches the foundations tier.

## Dependencies

### Internal
- Consumed by `src/eval/forgetting.py` (subspace angle uses the same QR approach as `oplora.orthogonal_projection`).
- Interacts with `src/base/loader.py` for hot-swap adapter loading at inference.

### External
- `torch`, `peft`, `transformers`, `trl`, `datasets`, `pyyaml` — required for `trainer.py::StackTrainer.train` (lazy-imported inside the method so module import stays cheap).
- Production training runtime: `mlx_lm` on Mac Studio — external, not imported here.

<!-- MANUAL: -->
