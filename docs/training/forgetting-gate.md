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
  "angle_degrees_per_layer": {
    "q_proj": 71.8, "k_proj": 74.1, "v_proj": 70.9, "o_proj": 72.4
  },
  "warning": null,
  "gate_status": "angle_only_partial",
  "note": "Win-rate half not measured; run paired eval for full gate."
}
```

- `angle_degrees_mean` — mean over `{q,k,v,o}_proj` groups.
- `angle_degrees_per_layer` — per-projection breakdown.
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

## Reference

Full protocol, alternatives considered, and roadmap:
`.omc/brainstorm-oplora.md`.
