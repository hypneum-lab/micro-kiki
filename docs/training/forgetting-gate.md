# Forgetting gate — operator reference

## Purpose

After each sequential LoRA stack is trained, we must check it has not
catastrophically damaged any previously-trained stack. Per top-level
`CLAUDE.md`:

> Run forgetting check after EACH stack; rollback if `angle < 30°`
> **AND** win-rate drop `> 0.03`.

Phase 1a ships the **angle half** of that gate as a standalone CLI.
The win-rate half (paired evaluation vs. base + prior) is deferred to
phase 1b.

## Usage

```bash
python scripts/measure_forgetting.py \
    --prior-adapter output/stacks/stack-03-cpp/adapter_model.safetensors \
    --new-adapter   output/stacks/stack-04-rust/adapter_model.safetensors \
    --output        results/forgetting-stack04-vs-stack03.json
```

Loguru progress logs go to stderr; the JSON report is written to stdout
and (optionally) to `--output`. Exit code is always `0` — this tool is
informational, not gating.

## Output

```json
{
  "angle_degrees_mean": 72.3,
  "angle_degrees_per_module": {
    "self_attn.q_proj": 71.8,
    "self_attn.k_proj": 74.1,
    "self_attn.v_proj": 70.9,
    "self_attn.o_proj": 72.4,
    "linear_attn.in_proj_qkv": 83.2,
    "mlp.switch_mlp.down_proj": 45.7
  },
  "warning": null,
  "gate_status": "angle_only_partial",
  "note": "Win-rate half not measured; run paired eval for full gate."
}
```

- `angle_degrees_mean` — mean over all module groups present in both adapters.
- `angle_degrees_per_module` — per-module-kind breakdown (grouped across
  layers). Covers every module the adapter was trained on, including
  `self_attn.{q,k,v,o}_proj`, `linear_attn.*`, and MoE
  (`mlp.{gate,shared_expert,switch_mlp,shared_expert_gate}.*`).
- `warning` — `"angle below threshold"` when mean `< 30°`, else `null`.
- `gate_status` — always `"angle_only_partial"` in phase 1a.

## Interpretation

| Mean angle | Action |
|---|---|
| `> 60°` | Healthy — stacks nearly orthogonal. Proceed. |
| `30°–60°` | Watch, but no rollback signal. Proceed. |
| `< 30°` | **Potential forgetting.** Run phase 1b win-rate eval before deciding. Rollback only if `Δwin-rate > 0.03`. |

## Current limitation

Angle alone cannot trigger rollback. A stack with `angle=20°` may still
be harmless if win-rate on prior domains is unchanged. The full gate
(angle **AND** win-rate drop) requires the paired eval wiring in
`src/eval/forgetting.py::check_all_previous` — see phase 1b.

## Pairwise sweep across a fleet of adapters

`scripts/run_forgetting_sweep.py` runs `measure_forgetting_signal()` for
every ordered pair `(prior, new)` in a directory of adapters and emits a
single JSON matrix. Useful for cross-checking N post-pivot adapters at
once without N² manual invocations.

```bash
python scripts/run_forgetting_sweep.py \
    --adapters-dir /path/to/lora-qwen36-35b \
    --output results/forgetting-matrix.json
```

Each immediate subdirectory containing `adapters.safetensors` is treated
as one adapter (subdir name → adapter label). Output shape:

```json
{
  "adapters": ["chat-fr", "python", ...],
  "pairs": [
    {
      "prior": "chat-fr",
      "new": "python",
      "angle_degrees_mean": 78.4,
      "angle_degrees_per_module": { "self_attn.q_proj": 80.1, ... },
      "gate_status": "angle_only_partial"
    },
    ...
  ],
  "flags": {
    "any_pair_below_30": false,
    "min_mean_angle": 66.56,
    "worst_pair": ["typescript", "python"],
    "angle_threshold_degrees": 30.0
  }
}
```

- **Exit 0** — every pair's mean angle ≥ 30°.
- **Exit 1** — at least one pair fell below 30° (worst pair surfaced in
  `flags.worst_pair`, minimum in `flags.min_mean_angle`). Treat as a
  forgetting-risk signal; run the full phase 1b win-rate gate on the
  flagged pair before accepting the new stack.
- **Exit 2** — fewer than two adapters discovered (CLI misuse).

Progress logs go to stderr via stdlib `logging`, so the script runs on
the Mac Studio training venv (torch/safetensors/numpy only — no
`loguru` dependency).

## Reference

Full protocol, alternatives considered, and roadmap:
`.omc/brainstorm-oplora.md`.
