<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# research

## Purpose
Exploratory code that isn't part of the shipping `src/` package. Currently holds a single investigation: the `ane-hybrid/` pipeline, which tried to run Qwen3.5-35B-A3B on the Apple Neural Engine via a DeltaNet (linear-recurrent attention) variant. The verdict recorded in its `CLAUDE.md` is that pure MLX (14.2 tok/s) beats the ANE hybrid (5.7-9.9 tok/s) on an M3 Ultra, with mlx-vlm native hitting 45-89 tok/s — ANE is only useful when the GPU is already busy with training. Code stays here for reproducibility and for the follow-up paper outlined in `docs/paper-outline-triple-hybrid.md`.

## Key Files
No files directly at this level — all content lives under `ane-hybrid/`.

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `ane-hybrid/` | ANE+Metal hybrid pipeline for Qwen3.5-35B-A3B. Phases: 1.1 `deltanet_reference.py` (PyTorch reference, diff=0.0), 1.2 `deltanet_conv2d.py` (ANEMLL Conv2d form, diff=6.64e-15), 1.3 `convert_deltanet.py` (CoreML conversion), 1.4 `deltanet_real.py` (real weights, 474 tok/s/layer), 2 `phase2_full_stack.py` (40-layer stack, 14.4 tok/s ANE), 3 `phase3_moe_hybrid.py` (ANE+CPU hybrid, 9.9 tok/s), 3b `phase3b_gpu_experts.py` (5.7 tok/s pipeline), `mlx_pure_full_model.py` (reference pure-MLX at 14.2 tok/s). Contains its own `CLAUDE.md` and `README.md` |

## For AI Agents

### Working In This Directory
- This is exploratory — not wired into `src/` or `scripts/`. Run directly: `cd research/ane-hybrid && python deltanet_reference.py`.
- Before proposing an ANE path for a new problem, re-read `ane-hybrid/CLAUDE.md`: CoreML dispatch overhead (~2 ms/layer) dominates and numpy<->MLX marshalling erases pipeline gains. ANE pays off only when the GPU is pinned (e.g. during training).
- Qwen3.5 DeltaNet produces garbage via ANEMLL — the architecture is not supported. Don't rediscover this; pick pure MLX or GPU instead.
- DeltaNet state shape is `[1, 32, 128, 128]` per layer; Conv1d depthwise kernel = 4. 30 DeltaNet layers + 10 Full-Attention (GQA 16Q/2KV, head_dim=256).
- No tests cover this directory — keep it that way unless you promote a file into `src/`.

### Testing Requirements
None. This is research code; equivalence is checked ad-hoc (`diff=0.0`, `diff=6.64e-15`) when comparing to the PyTorch reference. Do not gate CI on these scripts.

### Common Patterns
- Each phase file prints throughput in tok/s and a numerical diff against the previous phase.
- Results recorded in the directory-local `CLAUDE.md` — update that table if you add a new phase.

## Dependencies

### Internal
Imports the Qwen3.5-35B-A3B weights loader (independently; does not go through `src/base/loader.py`).

### External
`torch` (reference), `coremltools` (CoreML conversion), `mlx` (pure MLX baseline), `anemll` toolchain for ANE dispatch. Apple Silicon required; CoreML won't build on kxkm-ai.

<!-- MANUAL: -->
