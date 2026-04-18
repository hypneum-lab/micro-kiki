<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# docs/training

## Purpose
Operator-facing training runbook for the MLX LoRA pipeline on Mac Studio M3 Ultra. This is the doc you reach for when you're about to launch (or resume) a training run — it documents prerequisites, the 3-phase curriculum (seq 512 → 1280 → 4096), config locations, observed wall-clock/peak-memory numbers, constraints that have bitten us before, the mandatory forgetting check, and the distillation path for sparse niche domains. Pairs with the per-domain and per-phase YAMLs in `configs/mlx-per-domain/` and `configs/mlx-curriculum/`.

## Key Files

| File | Description |
|------|-------------|
| `README.md` | Training workflow guide. Covers: prereqs (venv, Metal wired-limit sysctl, base weights), data layout, 3-phase curriculum table, run commands, switching domain, key parameters table (rank 64 / scale 64 / batch 1 / grad-accum 16 / checkpointing on), constraints ("batch 4 hangs GPU", "LR > 2e-4 diverges"), observed metrics per phase, forgetting check, curriculum order (Phases I–VI), output tree, distillation for sparse domains. |

## For AI Agents

### Working In This Directory

- This README is the **authoritative operator guide** — if you change training behavior (new phase, different LR, new config location), update it here in the same commit.
- The "Observed Metrics" table is load-bearing for capacity planning — update numbers after every run that changes them materially, don't let it rot.
- Constraints section has hard-won rules ("batch 4 causes GPU hang when other MLX processes run", "LR > 2e-4 diverges on 35B MoE"). Don't silently remove a constraint; if it no longer applies, mark it with a date and a new observation.
- Note: this README references `~/KIKI-Mac_tunner/` paths and `rank 64` (122B-style config). The post-2026-04-16 pivot uses `models/qwen3.5-35b-a3b` locally with ranks 4/16/32 per tier (see `/home/kxkm/micro-kiki/CLAUDE.md` and `configs/mlx-per-domain/*.yaml`). If you touch this doc, reconcile it with the pivot — don't propagate the old rank-64 numbers into new text.

### Testing Requirements

- Command snippets should be executable as-is from the stated working directory. If you change a script path, grep-check that this README still points at something real.
- The forgetting-check command (`uv run python -m src.eval.forgetting --new-stack <domain> --prior-stacks <list>`) must match the actual CLI.

### Common Patterns

- Tables for phase parameters and observed metrics — one row per phase, columns pinned.
- Fenced bash blocks for every concrete command.
- Pitfalls and constraints in bullet form, each starting with a concrete failure mode ("batch 4 causes...", "LR > 2e-4 diverges...").

## Dependencies

### Internal
- Consumes configs in `../../configs/mlx-curriculum/` and `../../configs/mlx-per-domain/`.
- References `scripts/distill_with_local_teacher.py` and `src.eval.forgetting` for concrete command invocations.
- Paired with `../specs/2026-04-16-architecture-pivot-35b.md` (design) and `../plans/2026-04-15-micro-kiki-plan2-brainstacks-training.md` (execution plan).

### External
- `mlx-lm` lora subcommand. `uv` for env management.
- Qwen3.5-35B-A3B base weights (BF16, ~65 GB) and Qwen3-Coder-480B teacher (MLX 4bit, ~1.1 TB) on local disk.

<!-- MANUAL: -->
