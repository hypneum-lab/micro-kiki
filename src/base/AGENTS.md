<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-18 -->

# base — Core Model Loading + Architectural Mods

## Purpose
Thin layer between the pretrained Qwen3.5-35B-A3B checkpoint and everything else in `src/`. Owns the hot-swap LoRA loader (so the orchestrator can flip adapters per router decision, up to 4 active stacks) and the Differential Attention arch mod (ICLR 2025, arxiv 2410.05258) used experimentally on foundation stacks.

## Key Files
| File | Description |
|------|-------------|
| `loader.py` | `BaseModelLoader` — loads Qwen3.5-35B-A3B and manages hot-swap LoRA attachment/detachment. The single source of truth for device placement, dtype, and adapter merge/unmerge semantics. |
| `diff_attention.py` | Differential Attention (arxiv 2410.05258, ICLR 2025). Optional arch swap for attention blocks; reduces attention-noise on long contexts. |
| `__init__.py` | Re-exports `BaseModelLoader`. Docstring references Qwen3.5-35B-A3B (post-pivot, aligned with parent `CLAUDE.md` §architecture-pivot). |

## For AI Agents

### Working In This Directory
- **Base model is Qwen3.5-35B-A3B** (native MoE, 256 experts, 3B active/token, 262K ctx, Apache 2.0). The 2026-04-16 pivot is locked; all docstrings in this directory now reflect it.
- **Don't train on kxkm-ai**: project `Don't` — the 24 GB RTX 4090 can't hold BF16 LoRA for 35B-A3B. This machine is Q4 inference only. Training lives on Mac Studio via MLX.
- **Hot-swap means UNMERGE before switching**: merging a LoRA into the base weights and then attaching a second one corrupts both. `BaseModelLoader` must always detach-then-attach, never merge-then-attach.
- **Max 4 active stacks** per project `Don't`. The loader should refuse to attach a 5th adapter.
- **Device placement is explicit**: follow `src/CLAUDE.md` — `device_map`, `.to(device)`, never silent migration. BF16 for training, Q4_K_M for inference; never drop below Q4 (quality cliff).
- **Differential Attention is opt-in**: do not enable globally. It interacts with LoRA targets (q/k/v/o) — validate on a foundation stack with the forgetting check before any rollout.

### Testing Requirements
- Mirror tests in `tests/base/`. `test_loader.py` must cover the hot-swap invariant: load → attach A → detach → attach B → verify weights match "fresh attach B".
- Differential attention tests use a tiny toy model (not 35B) — full-scale validation is a training-job concern, not unit-test scope.

### Common Patterns
- Explicit device + dtype arguments on every public entry point; no inferred defaults based on global CUDA state.
- Context managers around VRAM-sensitive operations (`src/CLAUDE.md` convention).

## Dependencies

### Internal
- Used by `src/orchestrator/` (the engine asks the loader to flip adapters per router decision).
- Used by `src/spiking/` (converter reads layer weights through the loader).
- Used by `src/compress/` (CompactifAI reads the base model to build its tensor-network factorisation).

### External
- `transformers` (model/tokenizer loading), `peft` (LoRA attach/detach via `PeftModel`).
- `torch` for everything else.

<!-- MANUAL: -->
