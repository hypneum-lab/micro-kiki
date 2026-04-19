<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# research/ane-hybrid

## Purpose
Research spike: run Qwen3.5-35B-A3B (DeltaNet gated linear attention + full-attention hybrid) on Apple Neural Engine (ANE) via CoreML, combined with Metal GPU for the MoE experts. The spike produced a concrete verdict: **MLX pure (14.2 tok/s end-to-end) beats every ANE-hybrid variant (5.7–9.9 tok/s)** on M3 Ultra. `mlx-vlm` native hits 45–89 tok/s — the clear winner. ANE is only useful when the GPU is saturated (e.g. ~14 tok/s on ANE while training runs on Metal). The phased experiments in this directory are preserved as reference for anyone tempted to revisit the ANE hybrid idea.

See `CLAUDE.md` in this directory for the verdict table and anti-patterns; `README.md` is an older phase index.

## Key Files

| File | Phase | Description |
|------|-------|-------------|
| `deltanet_reference.py` | 1.1 | Reference PyTorch recurrent + chunkwise DeltaNet. Diff vs official: 0.0. |
| `deltanet_conv2d.py` | 1.2 | Conv2d rewrite for ANEMLL (ANE-friendly op set). Diff vs reference: 6.64e-15. |
| `convert_deltanet.py` | 1.3 | CoreML conversion — produces `.mlpackage`. |
| `deltanet_real.py` | 1.4 | Load real Qwen3.5 weights into the ANE DeltaNet. 474 tok/s per layer on ANE. |
| `phase2_full_stack.py` | 2 | Stack all 40 layers on ANE. 14.4 tok/s end-to-end. |
| `phase3_moe_hybrid.py` | 3 | Hybrid ANE (attention) + CPU (MoE experts). 9.9 tok/s. |
| `phase3b_gpu_experts.py` | 3b | Hybrid ANE + GPU experts. 5.7 tok/s (pipeline overhead dominates). |
| `mlx_pure_full_model.py` | — | MLX-only control run. 14.2 tok/s end-to-end. |
| `CLAUDE.md` | — | Verdict table + anti-patterns (CoreML dispatch overhead, numpy↔MLX marshalling, Qwen3.5 DeltaNet unsupported by ANEMLL). **Read first.** |
| `README.md` | — | Older phase index (partial, pre-verdict). |

## For AI Agents

### Working In This Directory

- **Read `CLAUDE.md` first.** The verdict is final for M3 Ultra + Qwen3.5-35B-A3B: MLX pure wins. Don't re-run the hybrid hoping for a different number — you will get the same number.
- This is **archived research**, not production code. Don't import from here into `src/` or `scripts/`.
- If you revisit ANE work for a different model (smaller, different arch), start a new directory — don't edit these files.
- Anti-patterns recorded in `CLAUDE.md`:
  - CoreML per-layer dispatch overhead (~2 ms/layer) dominates the hybrid pipeline.
  - numpy ↔ MLX marshalling cancels the parallel-pipeline speedup.
  - Qwen3.5 DeltaNet produces garbage through ANEMLL — unsupported architecture.
- The DeltaNet reference (30 gated linear + 10 full-attention layers, GQA 16Q/2KV head_dim=256, `[1, 32, 128, 128]` state per DeltaNet layer, depthwise Conv1d kernel=4) is a useful reference implementation even if ANE is off the table.

### Testing Requirements

- Reproducibility smoke: `python deltanet_reference.py` asserts recurrent vs chunkwise diff = 0.0.
- `deltanet_conv2d.py` asserts diff vs reference ≈ 6.64e-15.
- No CI for this directory — it's research scratch.

### Common Patterns

- Each phase-N file is self-contained: load weights, run a small batch, print throughput.
- Throughput reported in tok/s, end-to-end on M3 Ultra 512 GB.
- Comments and identifiers are mixed French / English (author preference). Don't bulk-translate; when adding new code stay consistent with the file you're editing.

## Dependencies

### Internal
- Standalone. No imports from `src/` or `scripts/`.
- Referenced by `docs/plans/2026-04-15-micro-kiki-plan4-ane-pipeline.md` (the plan this research invalidated).

### External
- PyTorch (reference), CoreML Tools (conversion), ANEMLL (ANE deployment), MLX (control run), numpy.

<!-- MANUAL: -->
