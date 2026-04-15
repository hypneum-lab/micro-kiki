# Differential Attention Integration

## Source
arxiv 2410.05258 (ICLR 2025, Microsoft)

## Application
Applied to the **13 full_attention layers** of Qwen3.5-4B only.
The 36 linear_attention (GatedDeltaNet) layers remain untouched.

## Mechanism
scores = softmax(Q1*K1) - lambda * softmax(Q2*K2)
- lambda is learnable per head, initialized scaling with depth
- Q2/K2 warm-started from Q1/K1 with small perturbation

## Calibration
Short SkyLadder-style pass on ~5K tokens to stabilize lambda.
Duration: ~30 min on RTX 4090.

## Rollback
If perplexity delta > 3% OR activation outliers not reduced:
- Fall back to vanilla Qwen3.5-4B
- Emit results/diffattn-rollback.json
- Update all configs to use base model path
