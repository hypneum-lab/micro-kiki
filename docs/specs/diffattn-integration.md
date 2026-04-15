# DiffAttn integration — step 2 spec

## Context

Dragon LLM's Hymba architecture (https://dragonllm.substack.com/p/inside-dragons-architecture) demonstrated that applying Differential Attention (arxiv 2410.05258, ICLR 2025) to the few global/full-attention layers yields meaningful quality gains:
- Long-context retrieval (critical for Aeon memory palace in Phase VIII)
- Hallucination reduction
- Reduced activation outliers (benefits Q4 serving quantization)
- Improved in-context learning (helps few-shot stack training)

Qwen3.5-4B has 49 transformer blocks: **13 full_attention** + **36 linear_attention** (GatedDeltaNet). The DiffAttn fork applies only to the 13 full_attn layers, mirroring Dragon's approach on their 3 global layers.

## Formula

Standard attention: `softmax(Q·K / √d) · V`

Differential Attention: `(softmax(Q1·K1 / √d) − λ · softmax(Q2·K2 / √d)) · V`

- `λ` is a learnable scalar per head (or per layer, choose in implementation)
- Initialized as `λ_init = 0.8 − 0.6 · exp(−0.3 · (layer_idx − 1))` per paper recipe
- Q2/K2 duplicate Q1/K1 projections (parameter doubling in Q/K, not in V)

## Implementation

1. Fork `models/qwen3.5-4b/bf16/` → `models/qwen3.5-4b-diffattn/`
2. For each of the 13 full_attn layers:
   - Duplicate `q_proj` → `q1_proj` + `q2_proj` (init q2 ≈ q1 + small noise)
   - Duplicate `k_proj` → `k1_proj` + `k2_proj` (idem)
   - Add learnable `lambda` parameter
3. Rewrite forward pass to compute the differential formula
4. SkyLadder calibration: 5K tokens sample, unfreeze only `lambda` + small Q2/K2 LR
5. Save to target directory with identical `config.json`/`tokenizer.json` structure

## Acceptance

- Perplexity delta vs vanilla base on 1K held-out ≤ **2%**
- Activation outliers (max/mean ratio per layer) reduced ≥ **30%** on full-attn layers
- Forward pass works end-to-end (same interface as vanilla Qwen3.5-4B)

## Rollback

If delta > 3% OR outliers not reduced:
- Auto-fall back to vanilla Qwen3.5-4B
- Rewrite all `configs/stack-NN-*.yaml` to set `base_model: models/qwen3.5-4b/bf16`
- Emit `results/diffattn-rollback.json` with failure metrics
- Do NOT proceed with stack training on degraded base

## Dependencies

- Step 1: base Qwen3.5-4B downloaded
- Reference impl: official Differential Transformer code (if available), or PyTorch hook-based fork

## References

- Differential Transformer (arxiv 2410.05258, ICLR 2025, Microsoft)
- Dragon LLM architecture analysis (https://dragonllm.substack.com/p/inside-dragons-architecture)
- Hymba hybrid architecture (NVIDIA) — context on global-vs-linear attention
