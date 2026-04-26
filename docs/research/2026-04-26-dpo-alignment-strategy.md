# DPO Alignment Strategy for micro-kiki

**Date**: 2026-04-26
**Status**: Design — awaiting implementation decision
**Author**: Claude (design), Clems (direction)

---

## 1. Problem Statement

### The specialization-generalization tradeoff

micro-kiki's 35 LoRA adapters deliver strong in-domain gains but regress general coding capability:

| Metric | Value | Direction |
|--------|-------|-----------|
| Domain PPL improvement | -71% average (35/35 domains) | Good |
| MBPP (python adapter) | -7pp | Bad |
| MBPP (math adapter) | -24pp | Toxic |
| MBPP (cpp adapter) | -6pp | Bad |

### Why SFT alone cannot solve this

SFT optimizes a single distribution — the domain-specific training data. When an adapter is trained on embedded systems examples, the gradient updates that improve perplexity on embedded content simultaneously shift the model's attention away from general Python coding patterns. The loss function has no mechanism to penalize degradation on out-of-distribution tasks.

This is not a data quality issue. Stub filtering (removing lazy/incomplete responses from training data) recovered only +1pp on MBPP. The regression is structural: LoRA specializes the model's attention heads and FFN projections toward the domain distribution, narrowing the capability manifold.

### Why DPO can help

DPO (Direct Preference Optimization) introduces a contrastive signal: the model learns not just "what is good for this domain" but also "do not lose the ability to do X." By training on (preferred, rejected) pairs where "preferred" maintains general capability and "rejected" is the overspecialized output, the adapter learns to specialize without catastrophic narrowing.

---

## 2. DPO Data Generation Strategy

### Option A: MBPP-contrastive pairs (Recommended)

For each MBPP problem where the adapter fails and the base model succeeds:

1. Run base Qwen3.6-35B (no adapter) on MBPP — record correct outputs
2. Run adapter-equipped model on MBPP — record incorrect outputs
3. Build pairs: `chosen = base_correct`, `rejected = adapter_incorrect`
4. Only keep pairs where base passes all assertions and adapter fails

**Yield estimate**: ~100-200 pairs per coding-adjacent domain (python, cpp, math, reasoning). Fewer for hardware-only domains (embedded, kicad-dsl) where MBPP overlap is lower.

**Strengths**: Directly targets the regression. Each pair encodes "this is what you should still be able to do."

**Weaknesses**: Limited to MBPP distribution. May not generalize to other general capabilities.

### Option B: Teacher-distilled pairs

Use Qwen3-Coder-480B teacher to generate both good and bad responses:

1. For each domain prompt, teacher generates a high-quality response (chosen)
2. Synthetically degrade it (truncation, wrong-domain injection, stub patterns) for rejected

**Note**: This approach already exists in `scripts/gen_dpo_pairs.py` using synthetic degradation, and in `scripts/generate_dpo_pairs.py` using the 480B judge. However, these scripts target in-domain quality (making the adapter better within its domain), not cross-domain preservation (preventing regression on general tasks).

**Strengths**: More diverse, scales to all 35 domains.

**Weaknesses**: Less targeted — doesn't directly address the MBPP regression signal.

### Option C: Mixed — MBPP contrastive + in-domain SFT (Recommended for production)

Combine both objectives in a multi-task loss:

```
L_total = lambda_SFT * L_SFT + lambda_DPO * L_DPO
```

- `L_DPO`: DPO loss on MBPP-contrastive pairs (preserve general capability)
- `L_SFT`: Standard SFT loss on domain data (maintain specialization)
- Starting point: `lambda_SFT = 0.3`, `lambda_DPO = 0.7` (emphasize preservation)

**Rationale**: Pure DPO on MBPP pairs risks overcorrecting — the adapter might lose some domain specialization to recover MBPP. The SFT term acts as an anchor.

---

## 3. Implementation Options

### Option 1: MLX DPO (custom implementation)

`mlx_lm` does not natively support DPO training. A custom implementation is needed.

**DPO loss**:

```
L_DPO = -log(sigma(beta * (log pi(y_w|x) - log pi(y_l|x)
                          - log pi_ref(y_w|x) + log pi_ref(y_l|x))))
```

Where:
- `pi` = policy (model with LoRA adapter being trained)
- `pi_ref` = reference model (base Qwen3.6 without LoRA, frozen)
- `y_w` = preferred (winning) response
- `y_l` = rejected (losing) response
- `beta` = temperature (typically 0.1-0.5)

**Memory requirements**:
- Reference model (base Qwen3.6 BF16): ~19 GB
- Training model (base + LoRA): ~19 GB + adapter
- Activations, optimizer state: ~10-20 GB
- Total: ~40-60 GB — fits comfortably in 512 GB Mac Studio

**Implementation path**:
1. Fork `mlx_lm`'s LoRA training loop
2. Add reference model loading (freeze all params)
3. Implement DPO loss computation (forward pass on both models)
4. Compute per-token log-probs for chosen/rejected on both policy and ref
5. Apply the DPO gradient only to the LoRA params

**Effort**: 3-5 days. Risk: medium (MLX autograd works, but custom training loops need careful validation).

### Option 2: PyTorch DPO on kxkm-ai (RTX 4090)

Use HuggingFace TRL `DPOTrainer` — the existing `scripts/train_dpo_kxkm.py` already implements this.

**Problem**: 35B model does not fit in 24 GB VRAM, even with QLoRA (known MoE-layer issues per CLAUDE.md: "Don't use QLoRA / BitsAndBytes on 35B-A3B"). The existing script defaults to `Qwen3.5-4B` for this reason.

**Verdict**: Not viable for the 35B model. Could work as proof-of-concept on 4B.

### Option 3: 4B proof-of-concept, then port to MLX

1. Run DPO on `Qwen3.5-4B` using `train_dpo_kxkm.py` on kxkm-ai
2. Validate that MBPP-contrastive DPO recovers general capability
3. If it works, implement the MLX DPO for 35B

**Effort**: 1-2 days for PoC. Low risk.

**Value**: Proves the approach before investing 3-5 days in MLX DPO.

### Option 4: SimPO (reference-free DPO)

SimPO eliminates the reference model, saving ~19 GB memory and simplifying implementation:

```
L_SimPO = -log(sigma(beta * (log pi(y_w|x)/|y_w| - log pi(y_l|x)/|y_l| - gamma)))
```

Where:
- Length normalization (`/|y|`) prevents length bias
- `gamma` is a margin term (typically 0.5-1.0)
- No `pi_ref` needed — only one forward pass per sample

**Advantages**:
- ~50% less memory than standard DPO (no reference model)
- Simpler implementation (no dual forward pass)
- Recent papers show SimPO matches or exceeds DPO on code tasks

**Disadvantages**:
- Less theoretically grounded (no KL anchor to reference)
- Risk of mode collapse without the reference regularization
- Newer technique, less battle-tested

**Effort**: 2-3 days in MLX. Lower risk than full DPO.

---

## 4. Existing Infrastructure Inventory

The repo already has significant DPO infrastructure:

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/gen_dpo_pairs.py` | Synthetic degradation pairs from SFT data | Working, but targets in-domain quality |
| `scripts/generate_dpo_pairs.py` | 480B judge-based pairs (SFT vs teacher) | Working, requires teacher server |
| `scripts/train_dpo_kxkm.py` | PyTorch QLoRA DPO on kxkm-ai | Working, 4B only |
| `scripts/train_dpo_niches.py` | MLX DPO training (mlx_tune or subprocess) | Partially working — `mlx_tune.DPOTrainer` import fails, subprocess fallback uses `fine_tune_type: "dpo"` in mlx_lm config |
| `kiki-forge/bin/eval_student.py` | GRPO eval loop — student generates, teacher rates, builds DPO pairs | Working, but depends on student + teacher servers |
| `scripts/eval_mbpp_v4.py` | MBPP benchmark (pass@1) | Working |

**Key gap**: No script generates MBPP-contrastive pairs (base-correct vs adapter-incorrect). All existing pair generators target in-domain quality improvement, not cross-domain preservation.

---

## 5. Training Protocol

### Phase 1: Generate MBPP-contrastive pairs (~2h)

```bash
# For each coding-adjacent domain:
# 1. Run base model on MBPP (500 problems), record pass/fail + output
# 2. Run adapter-equipped model on MBPP, record pass/fail + output
# 3. Filter: keep only (base_pass, adapter_fail) problems
# 4. Format: {prompt, chosen: base_output, rejected: adapter_output}
```

New script needed: `scripts/gen_mbpp_contrastive_pairs.py`

Domains to target (ordered by MBPP regression severity):
1. **math** (-24pp) — highest priority
2. **python** (-7pp) — core coding domain
3. **cpp** (-6pp) — significant regression
4. Other coding domains (javascript, typescript, etc.) — evaluate first

### Phase 2: DPO/SimPO training (~2h per domain)

Train each affected adapter with the contrastive pairs:

- **Loss**: SimPO (recommended for first iteration — simpler, no ref model)
- **Learning rate**: 1e-6 (10x lower than SFT, per existing `train_dpo_niches.py`)
- **Steps**: 50-100 (small dataset, risk of overcorrecting)
- **Beta**: 0.1 (start conservative)
- **Batch size**: 1, grad accumulation 4 (matching existing config)

### Phase 3: Evaluate (~4h)

For each DPO-aligned adapter, measure:
1. **MBPP pass@1** (must recover, target: regression < 2pp)
2. **Domain PPL** (must not regress significantly, target: still > 50% improvement)
3. **Forgetting angle** (must remain > 30° between stacks)

### Phase 4: Scale (~1 day if Phase 3 succeeds)

If the approach works on math/python/cpp:
1. Run MBPP eval on all 35 adapters to identify others with regression
2. Generate contrastive pairs for each
3. DPO-align all affected adapters

---

## 6. Success Criteria

| Metric | Current | Target | Hard Limit |
|--------|---------|--------|------------|
| MBPP regression (python) | -7pp | < -2pp | < -3pp |
| MBPP regression (math) | -24pp | < -2pp | < -5pp |
| MBPP regression (cpp) | -6pp | < -2pp | < -3pp |
| Domain PPL improvement | -71% avg | > -50% avg | > -40% avg |
| Forgetting angle | ~80° | > 30° | > 30° |
| No adapter "toxic" | math is toxic | 0 toxic adapters | 0 toxic |

"Toxic" = adapter causes > 5pp regression on any general benchmark.

---

## 7. Timeline and Dependencies

| Phase | Duration | Dependencies |
|-------|----------|-------------|
| Write `gen_mbpp_contrastive_pairs.py` | 2h | `eval_mbpp_v4.py` (exists) |
| Generate pairs (3 domains) | 2h | Base model + 3 adapters loaded |
| 4B PoC on kxkm-ai (Option 3) | 1-2 days | `train_dpo_kxkm.py` (exists) |
| SimPO in MLX (Option 4) | 2-3 days | MLX autograd, custom training loop |
| Full DPO in MLX (Option 1) | 3-5 days | MLX autograd, dual model loading |
| DPO training (3 domains) | 6h | Pair data + DPO implementation |
| Evaluation | 4h | MBPP fixture + PPL eval |
| Scale to all domains | 1 day | Successful Phase 3 |

**Total (recommended path)**: ~1 week
- Day 1: Generate contrastive pairs + 4B PoC
- Day 2-3: Implement SimPO in MLX
- Day 4: Train 3 priority domains
- Day 5: Evaluate + iterate

---

## 8. Recommendation

**Recommended path: Option 3 (4B PoC) then Option 4 (SimPO in MLX)**

1. **Day 1**: Write `gen_mbpp_contrastive_pairs.py`, generate pairs, run 4B PoC on kxkm-ai using existing `train_dpo_kxkm.py`
2. **Day 2-4**: If PoC validates, implement SimPO loss in MLX (simpler than full DPO, no reference model needed, ~40% less code)
3. **Day 5**: Train math/python/cpp adapters, evaluate

SimPO over full DPO because:
- No reference model = 19 GB less memory, simpler code path
- Length normalization is useful for code (avoids rewarding verbose stubs)
- If SimPO proves insufficient, upgrading to full DPO is incremental (add ref model loading + modify loss)

**Fallback**: If SimPO shows instability (mode collapse, overcorrection), switch to full DPO with the reference model anchor. The extra 19 GB is well within the 512 GB budget.

---

## 9. Open Questions

1. **Should DPO-aligned adapters replace the SFT adapters, or stack on top?**
   Current `train_dpo_niches.py` loads the SFT adapter as warm-start. This is correct — DPO refines the SFT adapter, not replaces it.

2. **How many contrastive pairs are enough?**
   Literature suggests 200-500 pairs for DPO on small adapters. Start with all available MBPP failures (~100-200 per domain), scale up with HumanEval if needed.

3. **Should we DPO-align domains that don't show MBPP regression?**
   Probably not initially. If a domain's adapter doesn't regress MBPP, DPO alignment is unnecessary overhead. Evaluate all 35 first, then target only the regressors.

4. **What about non-coding general capabilities?**
   MBPP measures Python coding. Adapters might also regress on reasoning, chat quality, or instruction following. Consider adding MT-Bench or similar as a secondary eval after the MBPP recovery is proven.
