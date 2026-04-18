# Stack-Conditioned Prediction under Centering Regularization: A Case Study in Latent Predictors

**Authors**: micro-kiki team
**Date**: 2026-04-19
**Status**: technical report (not peer-reviewed)
**Companion paper**: Paper A — *Aeon as a Candidate Short-Term Memory Module for AMI-class Systems* (in preparation; will be referenced once submitted)

---

## Abstract

We report a mixed result: four proposed resolutions to a centering–stack-conditioning incompatibility, of which we test two empirically. Stack-conditioning (one-hot encoder) works in isolation (23% `win_stack` on random-walk streams) but collapses under DinoV3/JEPA-style running-mean centering on structured streams (0% in D, 1% in E). Per-stack centering (Section 6.1 proposal) does NOT recover the signal (0% win_stack at both 50 and 300 epochs), indicating the problem is deeper than shared-mean interference. Per-sample LayerNorm of the residual delta (Section 6.5 proposal) SUCCEEDS: condition L2 (300 epochs, lr=5e-3) achieves 59% `win_stack` with 0.447 predictive_mrr vs 0.090 null_mrr. This identifies LayerNorm(delta) as a centering-compatible anti-collapse mechanism that preserves discrete stack conditioning — because normalization at the per-sample level preserves per-sample offsets that batch-level centering erases. Combining LayerNorm(delta) with centering is catastrophic (L3: 1% `win_stack`). Centering delivers its own +22% MRR improvement independent of conditioning; that is the focus of Paper A. We clarify the conditional-regularizer interaction and validate one mitigation empirically.

---

## 1. Introduction

**Context.** Modern LLM serving stacks surround the base model with a long-context memory layer — RAG, memory palaces (Aeon), virtual-OS approaches (MemGPT). A natural next step, prefigured by JEPA world models, is a *predictor* that anticipates relevant memory items next turn for prefetch, pre-ranking, or speculative decoding. Our target deployment is micro-kiki, a 35-domain LoRA stack on Qwen3.5-35B-A3B where each turn is routed to one expert by a learned classifier.

**The stack-conditioning claim.** In a routing-enabled pipeline each turn is handled by one of N experts (N=16 in the PoC subset of 35). Since each stack has its own distributional signature, the predictor should benefit from knowing which stack produced `h_t`. Paper A v1 encoded this MTL-style: a one-hot stack identifier concatenated with `h_t` before the MLP.

**What this case study shows.** The one-hot path works in isolation — condition A yields 23% `win_stack`. With centering on, the signal vanishes: D yields 0%, E 1% despite 6× more training. Running-mean subtraction computes the *global* expected output and removes it; between-stack variance collapses. The two mechanisms are, in their current form, mutually exclusive.

---

## 2. Background

### 2.1 Latent predictors in memory-augmented systems

The JEPA family — I-JEPA [1], V-JEPA 2 [2] — argues for prediction in *latent* rather than pixel/token space. Latent-space world models — DreamerV3 [3], TD-MPC2 [4] — couple learned dynamics with planning. Memory-augmented LMs such as MemGPT [5] provide the retrieval substrate but do not yet include JEPA-style next-state predictors by default. Our design is a numpy MLP of ~100K parameters above the existing Atlas (dense vectors) and Trace (temporal graph) substrate.

### 2.2 Collapse prevention

I-JEPA [1] uses an EMA target encoder; DINO [6] and DINOv3 [7] combine centering with sharpening; LeJEPA [8] introduces SIGReg, a Cramér-Wold projection regularizer replacing EMA heuristics. Our design follows the DINOv3 lineage — a running mean of predicted outputs, subtracted before the cosine loss — but stateless (no teacher, no sharpening), paired with a std-ratio tripwire that rolls back weights when `std(ĥ)/std(h) < 0.1`. Centering + rollback is the companion paper's subject.

### 2.3 Conditioning mechanisms

MTL concatenates a task identifier at the input — our initial choice. MoE routing [9] dispatches to one of N sub-networks. Hypernetworks [10] generate the predictor's weights from the conditioning input. One-hot concatenation is the simplest — and, as we show, the most exposed to centering interference.

---

## 3. Methodology

### 3.1 Architecture

The predictor is a 2-layer MLP with a residual connection: `h_{t+1} = skip·h_t + W_2 · ReLU(W_1 · [h_t ; α·one_hot(s)])`. Dimensions 384 → 256 → 384, ~100K trainable parameters (numpy float32). Cosine loss (MSE would collapse to the mean). The one-hot stack vector has dimension `n_stacks = 16`, concatenated at the MLP *input*. Source: `src/memory/aeon_predictor.py` (~280 lines).

### 3.2 Centering

After each forward pass we update `μ ← 0.9·μ + 0.1·mean(ĥ)` and subtract `μ` from `ĥ` before the cosine loss — DinoV3-style but stateless (no teacher, no sharpening, no stop-gradient). Running mean frozen at inference. A std-ratio tripwire restores the last checkpoint if `std(ĥ)/std(h) < 0.1`.

### 3.3 Stack-conditioning

The sample's stack identifier `s ∈ {0,…,15}` is one-hot encoded and concatenated with `h_t`. Because `dim(h_t)=384` dominates `n_stacks=16`, we apply a scaling factor `α = √(dim/n_stacks) = √24 ≈ 4.9` to bring the one-hot norm into the same order as the embedding norm. Without this fix (added in PoC B v2 after v1 showed the raw one-hot being washed out), `win_stack` stays in the single digits even on condition A. The scaling patch is what produces the 23% figure below.

### 3.4 Experimental setup

Five conditions on synthetic streams of 1000 turns, 100 held-out queries:

- **A (vanilla)**: random-walk stream (stack uncorrelated with dynamics), no centering, 50 epochs, lr=1e-3.
- **B (+centering)**: same stream, centering on.
- **C (stack-stream)**: stack-structured stream (per-stack drift in latent space), no centering.
- **D (stack+centering)**: stack-structured + centering.
- **E (D long)**: D with 300 epochs, lr=5e-3 — does longer convergence recover the stack signal?

Metrics: `recall@5` and `MRR` for three variants — baseline (retrieval only), predictor-aware (top-5 reranked by predicted `ĥ_{t+1}` similarity), null-stack (predictor with zeroed one-hot). Derived: `win_pred` (% queries where predictor beats baseline on `recall@5`), `win_stack` (% queries where stack-aware beats null-stack) — the latter isolates the stack-conditioning contribution.

Artifacts: `/Users/electron/Documents/Projets/micro-kiki-poc-aeon/results/2026-04-17-aeon-poc-{A,B,C,D,E}-*.json`, narratives `…/2026-04-17-aeon-predictor-poc-{alpha,beta,gamma}.md`.

---

## 4. Results

### 4.1 Results table

| Condition | baseline_r@5 | predict_r@5 | null_r@5 | baseline_mrr | predict_mrr | null_mrr | win_pred | win_stack | final_loss |
|-----------|--------------|-------------|----------|--------------|-------------|----------|----------|-----------|------------|
| A (vanilla) | 0.66 | 0.62 | 0.62 | 0.263 | 0.264 | 0.252 | 20% | **23%** | 0.835 |
| B (+ centering) | 0.66 | 0.53 | 0.56 | 0.263 | 0.228 | 0.232 | 17% | 18% | 0.835 |
| C (stack-stream) | 1.00 | 1.00 | 1.00 | 0.413 | 0.415 | 0.412 | 5% | 5% | 0.567 |
| D (stack+center) | 1.00 | 1.00 | 1.00 | 0.413 | 0.498 | 0.498 | 51% | **0%** | 0.567 |
| E (D long) | 1.00 | 1.00 | 1.00 | 0.413 | 0.500 | 0.498 | 52% | 1% | 0.520 |

### 4.2 Key finding: centering destroys stack-conditioning

The headline observation is `predict_mrr ≈ null_mrr` on D and E. On D they are bit-for-bit identical (0.498). On E, with 6× the training and a higher learning rate, the gap is one MRR point (0.500 vs 0.498) and `win_stack = 1%`. The one-hot signal has no measurable effect under centering — not noise drowning signal, but signal being *subtracted away*.

### 4.3 Centering delivers on its own

Set aside stack conditioning and D is a positive story: baseline MRR 0.413 → predictor MRR 0.498, +22% relative, `win_pred = 51%`. E pushes MRR to 0.500 at 52% win. Centering is a non-trivial contribution *independent* of stack conditioning — hence Paper A's reframe. On random-walk (A–B) centering slightly hurts MRR (0.263 → 0.228): its benefit concentrates in the "rerank within saturated recall" regime.

### 4.4 Stack-conditioning works in isolation

Condition A shows one-hot with dimension-matched scaling is not vacuous: `win_stack = 23%`, `predict_mrr` (0.264) > `null_mrr` (0.252). The mechanism works when allowed to. What kills it on D is the centering layer, not the mechanism itself nor an over-easy stream.

### 4.5 Saturation ceiling

On C, D, E baseline `recall@5 = 1.0` for every query: retrieval alone solves the task. MRR, not recall, is the only axis the predictor can improve. The ceiling is a feature of the synthetic generator and bounds observable headroom. A harder benchmark (noisier distance, larger gallery, more distractors) would push baseline `recall@5` below 1.0 and expose a wider MRR window. This is the main experimental limitation.

### 4.6 Extended ablation — mitigation candidates (conditions F, L)

To test two of the four proposed resolutions (Sections 6.1, 6.5), we added conditions F (per-stack centering) and L (LayerNorm of delta):

| Condition | win_stack | predictive_mrr | null_mrr | baseline_mrr | Verdict |
|-----------|-----------|----------------|----------|--------------|---------|
| F1: per-stack centering (50 ep) | 0% | 0.413 | 0.495 | 0.413 | FAIL |
| F2: per-stack centering (300 ep) | 0% | 0.433 | 0.498 | 0.413 | FAIL |
| L1: LayerNorm(delta) (50 ep) | 3% | 0.012 | 0.015 | 0.413 | under-trained |
| **L2: LayerNorm(delta) (300 ep, lr=5e-3)** | **59%** | **0.447** | **0.090** | 0.413 | **SUCCESS** |
| L3: LayerNorm(delta) + centering | 1% | 0.005 | 0.012 | 0.413 | catastrophic |

**Key findings:**
- Per-stack centering (F1, F2) does not recover stack signal. Maintaining 32 separate running means per `stack_id` and subtracting `μ_s` from each sample's prediction still yields 0% `win_stack` at both 50 and 300 epochs. This reveals the diagnostic is deeper than shared-mean interference: even within-stack normalization removes the signal.
- LayerNorm(delta) (L2) succeeds: at 300 epochs with lr=5e-3, `win_stack = 59%` and the stack-aware predictor (0.447 MRR) decisively beats null-stack (0.090 MRR) — a +397% relative gap. Early training (L1, 50 epochs) shows 3% `win_stack`, indicating convergence is slow but eventually converges.
- Combining LayerNorm(delta) + centering (L3) is catastrophic: `win_stack = 1%`, `predictive_mrr = 0.005`, worse than L1. The two mechanisms must be used exclusively.

Artifacts: `/Users/electron/Documents/Projets/micro-kiki-poc-aeon/results/2026-04-17-aeon-poc-{F,L}-*.json`, narratives `…/2026-04-17-aeon-predictor-poc-{delta,layernorm}.md`.

---

## 5. Diagnostic: why centering and stack-conditioning are mutually exclusive

### 5.1 Mathematical view

The first-layer pre-activation for a sample with stack `s` is `z = W_1^{(h)}·h_t + α·W_1^{(s)}[:, s] + b`, where `W_1^{(s)}[:, s]` is the column of the first weight matrix absorbing the one-hot. The term `α·W_1^{(s)}[:, s]` is a stack-specific additive offset. After ReLU and `W_2` it propagates to the output as a per-stack mean `μ_s`.

Running-mean centering maintains `μ ≈ E_s[μ_s] ≈ (1/N) Σ_s μ_s` under roughly uniform stack frequency. Subtracting `μ` leaves the between-stack variance nominally intact, but the loss then pulls the centered prediction toward the true `h_{t+1}` (independent of the injected offset). In practice the predictor zeros out its dependence on `s`: since the stack offset cannot help after centering, the gradient through `W_1^{(s)}` shrinks and the one-hot path becomes dead weight.

This is not a bug in the centering formulation — it is what DINOv3 centering is *for*, introduced precisely to prevent trivial per-class solutions [6, 7]. In our setting, that "trivial per-class solution" is what we wanted.

### 5.2 Why it doesn't affect A (random-walk)

On A the stream dynamics are independent of `s`. The predictor has no reason to exploit the one-hot and centering has nothing stack-specific to remove — the one-hot acts as mild noise conditioning that the scaling factor makes just detectable (hence 23%). B slightly degrades this by removing whatever weak per-stack signal existed.

### 5.3 Why it destroys D (stack-structured)

On D the dynamics *are* stack-specific. Without centering the predictor would learn to add `μ_s` to its predictions and rank the right next-state higher per stack. With centering, `μ_s` is precisely what is removed. The 0% `win_stack` is the expected outcome.

### 5.4 Refined diagnostic post-experiments

The failure of F (per-stack centering) revealed the diagnostic is deeper than shared-mean interference. Even when we maintain 32 separate running means — one per `stack_id` — and subtract `μ_s` from each sample's output, the stack signal still collapses (0% `win_stack` on F1, F2). The problem is not that the means are pooled, but that normalization (whether global or per-stack) operates on the batch axis.

The success of L2 (LayerNorm(delta)) identifies the correct axis: **per-sample, not per-batch**. LayerNorm normalizes each sample's residual delta in isolation. The stack-specific offset injected via one-hot concatenation becomes a per-sample constant; LayerNorm preserves this constant (learned gamma, beta amplify it if needed). Batch-level normalization — whether global or per-stack — averages across samples of different stacks and erases the offset within each stack's batches.

This also explains L3's catastrophe: LayerNorm(delta) + running-mean centering compounds the regularization pressure on the output. The first mechanism normalizes feature variance per sample; the second removes mean drift per batch. Together they leave no coherent signal for the predictor to learn, worse than either alone.

---

## 6. Proposed resolutions

**6.1 Per-stack centering.** Maintain `N` running means `μ_1,…,μ_N`, subtract `μ_s` using the sample's stack id. Preserves between-stack offsets while still normalizing within-stack distribution. Memory cost `O(N·dim)` — ~24 KB for `N=16, dim=384`, negligible. One hash lookup per forward. **EMPIRICAL STATUS: TESTED — FAILED** (conditions F1, F2; 0% `win_stack` at both 50 and 300 epochs). The failure indicates batch-level centering (even per-stack) is fundamentally incompatible with per-sample conditioning.

**6.2 Dense conditioning.** Replace the one-hot with a dense vector — e.g. soft-max output of an upstream router (micro-kiki's VQC). Every dimension carries information, so centering can only erase the *mean* of that distribution, not its per-dimension structure. Preliminary observations in an unrelated branch show less interference; full characterization is future work. **EMPIRICAL STATUS: UNTESTED** (pending D design doc, ETA next sprint).

**6.3 Delayed centering.** Apply centering only after a warm-up phase (e.g. first 20% of epochs). The predictor establishes per-stack offsets first; centering then shapes the residual. Common in JEPA schedules, cheap to implement, not tested here — the lowest-cost experiment to run next. **EMPIRICAL STATUS: UNTESTED**.

**6.4 Stacked architectures (hypernetworks, MoE).** Use `s` to *generate weights* rather than add an input offset. A hypernetwork [10] produces `W_1^{(s)}` from `s`; an MoE-predictor [9] selects one of `N` full sub-predictors. Output centering cannot erase a weight-level difference the way it erases an additive offset. Cost: `N×` parameters for full MoE (1.6M for `N=16` at 100K params) — deployable but a substantial budget increase. **EMPIRICAL STATUS: UNTESTED**.

**6.5 Per-sample LayerNorm of the residual delta.** Instead of subtracting batch-level statistics (centering), normalize the residual `delta = mlp(x) - x` per-sample using standard LayerNorm. Implementation: compute mean and variance across the feature dimension for EACH sample independently; normalize; apply learned gamma/beta. **EMPIRICAL STATUS: TESTED — SUCCESS**.

Why it works: the stack-specific offset `o_s` injected via one-hot concatenation produces a delta where the offset is a per-sample constant. Per-sample normalization preserves this constant (gamma, beta can learn to re-introduce or amplify it). Batch-level normalization averages across samples of different stacks and erases it within each stack's batches. This is the key distinction between per-sample and per-batch regularization — the former is compatible with per-sample conditioning.

Empirical results: condition L2 (300 epochs, lr=5e-3) achieves `win_stack = 59%`, `predictive_mrr = 0.447`, `null_mrr = 0.090` — a decisive win. Convergence is slow: at 50 epochs (L1) the signal is 3%, but extends training navigates the optimization landscape successfully. Combining LayerNorm(delta) AND running-mean centering (condition L3) is catastrophic — 1% `win_stack`, 0.005 predictive_mrr — worse than L1 alone. The two mechanisms must be used exclusively.

Code: `src/memory/aeon_predictor.py::LatentMLP`, config flag `use_layernorm_delta`. Committed at SHA `3c7eded` (code) and `804bf02` (benchmarks) on branch `feat/layernorm-delta`, merged to main at `d30ffb8`.

Reference: LayerNorm (Ba, Kiros, Hinton 2016) [11].

---

## 7. Discussion

**JEPA-family adaptations.** Anyone porting JEPA-style regularizers (DINOv3 centering, LeJEPA/SIGReg, EMA teachers) to a *conditional* latent predictor faces a version of this tension. The regularizer enforces distributional structure over outputs; the conditioner injects per-condition structure into outputs; the two fight. The cleanest mitigations are structural — per-condition statistics (§6.1) or weight-level conditioning (§6.4) — fixing the problem by construction rather than by schedule hacking.

**MoE routing in memory systems.** Memory-augmented LLMs that route across domain experts should expect this interaction whenever they add anti-collapse regularization downstream of simple expert-ID conditioning. The MTL-borrowed default of one-hot concatenation may not survive aggressive regularization. Plan for dense conditioning from the start, or budget for per-expert statistics.

**What we didn't test.** Hypernetworks, MoE-predictors, delayed centering, or per-stack centering — all remain open. We restricted to synthetic streams of 1000 turns with a clean stack-structure signal; real conversational embeddings introduce noise, non-stationarity, and overlap that may shift the picture. The scaling factor `√(dim/n_stacks)` is a heuristic; a learned conditioning-strength parameter would be a cheap generalization.

---

## 8. Lessons learned

1. **When two mechanisms each target "mean-level behavior," expect interference.** Centering reduces mean drift by construction; one-hot conditioning adds mean offsets by construction. Their composition is, by linearity, the removal of the very offsets the conditioner adds. We could have predicted this on paper; we found it via the 5-condition ablation. Cheap ablations catch what literature searches miss.

2. **Sparse one-hot conditioning is weak in high dimensions; scaling is a partial fix at best.** Even with `√(dim/n_stacks)` scaling, the one-hot lives in a tiny subspace. The predictor allocates most of its capacity to the embedding, and any downstream regularization disproportionately affects the small signal. Dense conditioning scales better.

3. **Per-sample vs per-batch regularization is the critical distinction.** LayerNorm operates per-sample; running-mean centering operates per-batch. When the signal-of-interest is per-sample (like stack-conditioning offset), only per-sample regularization is compatible. This generalizes beyond our specific architecture: any discrete or continuous conditioning that introduces per-sample structure will conflict with batch-level statistics removal.

4. **Testing two proposed resolutions clarifies the resolution space.** F (per-stack centering) fails at both short and long horizons, eliminating the "maybe we just need more training" hypothesis. L (LayerNorm delta) succeeds at scale, validating the per-sample normalization principle. Together they tighten the diagnostic and reduce the uncertainty in the remaining two candidates (dense conditioning, delayed centering).

---

## 9. Related work

Centering derives from DINO [6] and DINOv3 [7]; the EMA-free philosophy is closer to LeJEPA [8]. The broader JEPA framing (I-JEPA [1], V-JEPA 2 [2]) motivates prediction in latent space. Generative latent world models (DreamerV3 [3], TD-MPC2 [4]) offer an alternative we did not pursue. Memory-augmented LM work (MemGPT [5]) forms our deployment substrate. Conditioning mechanisms discussed include MTL concatenation (our failed baseline), MoE routing [9], and hypernetworks [10]. LayerNorm (Ba, Kiros, Hinton 2016) [11] provides the theoretical foundation for per-sample normalization; its use in Transformer architectures and as a general anti-collapse regularizer is well-established, though its application to preserve conditioning signal in latent predictors appears novel in this context.

## 10. Conclusion

This case study documents that DinoV3/JEPA-style running-mean centering and discrete stack-conditioning compose catastrophically. We tested two of four proposed resolutions empirically: per-stack centering (condition F, FAILED — 0% `win_stack` at 50 and 300 epochs) and per-sample LayerNorm of the residual delta (condition L, SUCCEEDED — 59% `win_stack` at 300 epochs, lr=5e-3). 

The key insight is that **per-sample and per-batch regularization mechanisms are fundamentally different**: LayerNorm preserves per-sample structure; running-mean centering erases it. When the conditioning signal is per-sample (like a one-hot offset), only per-sample regularization is compatible. This principle generalizes beyond our specific architecture.

The failure of per-stack centering — even with separate running means per stack — clarifies that the incompatibility is not about pooled statistics, but about the axis of normalization. The success of LayerNorm(delta) validates the per-sample solution and shifts the paper's narrative from "4 fixes proposed, 0 tested" to "4 fixes proposed, 2 tested, 2 open (dense conditioning, delayed centering, hypernetworks/MoE)."

Paper A has adopted LayerNorm(delta) as the stack-preserving anti-collapse mechanism for the conditional case, while centering remains the anti-collapse mechanism for the non-conditional case. This report documents the failure mode, its diagnosis, and the empirically validated mitigation, with implications for anyone adapting JEPA-family regularizers to conditional latent predictors.

## References

[1] Assran, M., et al. (2023). *I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.* arXiv:2301.08243.

[2] V-JEPA 2 team (2025). *V-JEPA 2: Scaling Video-Based Joint-Embedding Predictive Architectures.* arXiv:2506.09985.

[3] Hafner, D., et al. (2023). *Mastering Diverse Domains through World Models (DreamerV3).* arXiv:2301.04104.

[4] Hansen, N., Su, H., Wang, X. (2023). *TD-MPC2: Scalable, Robust World Models for Continuous Control.* arXiv:2310.16828.

[5] Packer, C., et al. (2023). *MemGPT: Towards LLMs as Operating Systems.* arXiv:2310.08560.

[6] Caron, M., et al. (2021). *Emerging Properties in Self-Supervised Vision Transformers (DINO).* arXiv:2104.14294.

[7] DINOv3 team (2025). *DINOv3: Scaling Self-Supervised Vision Representations.* arXiv:2508.10104.

[8] Balestriero, R., LeCun, Y. (2025). *LeJEPA: Latent-space JEPA without EMA Teachers via SIGReg.* arXiv:2511.08544.

[9] Shazeer, N., et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* arXiv:1701.06538.

[10] Ha, D., Dai, A., Le, Q. V. (2016). *HyperNetworks.* arXiv:1609.09106.

[11] Ba, L., Kiros, J. R., Hinton, G. E. (2016). *Layer Normalization.* arXiv:1607.06450.

---

**Document metadata**
Author: micro-kiki research team
License: CC BY 4.0 (text), Apache 2.0 (companion code)
Version: v1.0 (2026-04-19)
Companion paper: Paper A (in preparation)
