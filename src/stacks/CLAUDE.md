# src/stacks/ — LoRA stack adapters

Per-domain LoRA adapters that plug on top of Qwen3.5-35B-A3B at serving time.

## Modules

- `moe_lora.py` — **legacy** (custom 4-expert MoE-LoRA from the 4B era). Kept for reference; do not extend. New stacks use PEFT `LoraConfig` directly.
- `oplora.py` — OPLoRA orthogonal projection init. Applied from stack 04 onward to prevent catastrophic forgetting.
- `qtha.py` — quantum-inspired tensor-head adapter (experimental).
- `trainer.py` — training-loop utilities invoked by scripts; the actual training run happens in `~/KIKI-Mac_tunner/` via MLX.

## Adapter discipline

- Target modules = `q_proj, k_proj, v_proj, o_proj` only. Never FFN, never experts, never router.
- One stack = one domain = one `adapters.safetensors` artifact. No merged-into-base checkpoints.
- Adapters are swappable at runtime; keep them small and independent.

## Adding a stack (runtime side)

1. Dataset already classified + deduped in `~/KIKI-Mac_tunner/data/micro-kiki/<domain>/`.
2. Add `configs/mlx-per-domain/<domain>.yaml` + entry in the relevant curriculum file (see `configs/CLAUDE.md`).
3. Train from `~/KIKI-Mac_tunner` (not from here). Artifacts land in `checkpoints/`.
4. Run the forgetting gate (`src/eval/forgetting.py`) **immediately** after training.
5. Register the adapter in the router and re-run router tests.

## OPLoRA specifics

- Orthogonalise new adapter vs. the span of previously trained adapters for that layer.
- Only active from stack 04+ (foundations 01–03 train without it).
- Not yet wired into the MLX pipeline — TODO tracked in `docs/specs/`.

## Memory budget (for reference; training runs elsewhere)

- BF16 training peak: ~195 GB with gradient checkpointing on Mac Studio.
- Inference: max 4 active stacks per query (VRAM + interference).

## Anti-patterns (stacks-specific)

- Don't add a new adapter type in `moe_lora.py` — it's frozen.
- Don't silently change `lora_alpha` / `scale` — these are part of the per-stack contract; change in the YAML, not in code.
- Don't skip OPLoRA from stack 04 onward.
- Don't touch experts or router weights — adapters are attention-only.
