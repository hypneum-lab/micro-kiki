# micro-kiki — project context

32 domain experts (MoE-LoRA) on Qwen3.5-4B base. Fits RTX 4090 24 GB.

## Key decisions locked in

- Base: Qwen3.5-4B (Apache 2.0, 262K ctx)
- Adapter technique: MoE-LoRA (MoLoRA approach recommended, Brainstacks fallback)
- Quantization: Q4 for serving, BF16 for training
- Router: sigmoid meta-router, 32 outputs, threshold 0.12, chat floor 0.20
- Max 4 active stacks simultaneously (VRAM constraint)

## Teachers

See README.md for full list. Mistral-Large-Opus is the primary reasoning teacher.

## Don't

- Don't change the base model without updating all specs
- Don't drop below Q4 quantization (quality cliff below that)
- Don't route > 4 stacks simultaneously (VRAM exceeds 24 GB)
- Don't train stacks in parallel on the same GPU (interference)

## Do

- Train stacks sequentially, in curriculum order (foundations first)
- Run forgetting check after each new stack trained
- Keep per-domain datasets isolated (each example lives in ONE domain only)
- Test meta-router after every 4 stacks added
