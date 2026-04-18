# VQC-Accelerated CEM Planning in V-JEPA 2-AC — Research Note

**Date**: 2026-04-19
**Status**: research sketch, not empirical
**Author context**: prep for micro-kiki Paper A discussion / potential Paper B track

---

## 1. V-JEPA 2-AC CEM algorithm (exact)

Source: Assran et al., "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning" (arXiv:2506.09985), Section 5 — Planning with V-JEPA 2-AC.

**Goal-conditioned energy function.** At each control step `k`, given current state `s_k`, current latent `z_k`, and goal latent `z_g` (the frozen V-JEPA 2 encoder applied to the goal image), V-JEPA 2-AC scores an action sequence `a^{1:T}` by

```
E(a^{1:T}; z_k, s_k, z_g) := || P_φ(a^{1:T}; s_k, z_k) − z_g ||_1
```

where `P_φ` is the action-conditioned predictor (≈300 M params, 24 transformer layers, 16 heads, hidden 1024, block-causal, GELU; feature map shape 16×16×1408 tokens). The L1 distance is computed in the predictor's latent space.

**CEM loop (quoted / reconstructed from Section 5).**

- **Samples per refinement step**: 800 for V-JEPA 2-AC (the paper says "10× more samples in each refinement step" than Cosmos, which uses 80).
- **Refinement iterations**: 10.
- **Planning horizon (MPC receding horizon)**: T = 1 — i.e., only the first action of the optimised sequence is executed, then CEM is re-run. The rollout horizon used internally by the predictor over which the L1 goal distance is measured is short (the paper frames planning as single-step receding-horizon, with the predictor unrolled over the candidate action sequence).
- **Action constraint**: L∞-ball of radius 0.075 ≈ 13 cm max Cartesian displacement per action; actions parameterise end-effector deltas.
- **Elite fitting**: standard CEM — pick the top-k lowest-energy samples, refit a Gaussian (mean, diagonal std) over the action dimensions, sample the next batch from this Gaussian (clipped to the L∞-ball). The paper does not publish the elite fraction; standard CEM practice and open V-JEPA 2 code use k = 10–20% of the population (≈80–160 elites of 800).
- **Goal encoding**: the goal image is passed once through the frozen V-JEPA 2 encoder to get `z_g`; only `P_φ` is queried inside the CEM loop.

**Wall-clock**: "V-JEPA 2-AC world model requires only 16 seconds per action … Cosmos … takes 4 minutes per action step" — a ~15× speed-up vs. the diffusion baseline, on the same Franka arm deployment. Hardware is not explicitly specified in the quoted excerpt but is described as a single GPU node in the robot loop.

---

## 2. Classical CEM cost breakdown

Total predictor forward passes per control decision:

```
N_forward = N_samples × N_iters = 800 × 10 = 8 000
```

At 16 s/action, that is **~2 ms per predictor forward** (batched). Given the predictor is ≈300 M params over a short action-conditioned rollout, with V-JEPA 2's block-causal attention and cached key/values over the spatial tokens, 2 ms/forward on an H100-class GPU is plausible when the 800 samples in one iteration are batched together (batch = 800, 10 sequential CEM rounds).

Where the time actually goes:
1. **Predictor forward (>90 % of wall-clock)** — 8 000 forwards of a 300 M-param transformer. The L1 distance to `z_g` is a trivial tensor op on top.
2. **Gaussian refit (negligible)** — sort 800 scalars, take top-k, compute mean/std over action dim (≤20 dims).
3. **Sampling + clipping (negligible)** — draw 800 samples from N(μ, diag σ²) inside the L∞ ball.

Takeaway: **CEM is predictor-bound, not sampler-bound**. Any quantum acceleration that does not touch `P_φ` saves at most the last 5 % of the budget.

---

## 3. Where VQC could help (theoretical)

Three disjoint intervention points:

**Option A — Quantum sampler (replace the Gaussian).** Encode the action distribution as a parametrised quantum state, measure → candidate, refit QAOA-style (Farhi et al., arXiv:1411.4028). Targets ≤5 % of the wall-clock. Even a perfect quantum sampler buys very little.

**Option B — Quantum surrogate cost model.** Train a VQC `Ẽ_θ(a)` approximating `E(a; z_k, z_g)` and filter candidates before paying for a real `P_φ` forward. The only intervention that touches the dominant cost — but it inherits the surrogate-optimisation trade-off: `Ẽ_θ` must be much cheaper than `P_φ` *and* faithful to the elite ranking. Shaffer et al. (arXiv:2404.02951, PNAS 2025) show this works classically; in quantum hybrids, the VQC typically runs *as* the expensive model, not as the surrogate.

**Option C — Quantum warm-start.** Use the VQC once at iteration 0 to propose a low-energy initial population, then let classical CEM converge in fewer iterations (6 instead of 10). Savings scale with iterations cut. Closest analogue to how QAOA is actually useful in hybrid pipelines today, and the route QI-MPC (Khan & Al-Karaki, arXiv:2504.13041, 2026) takes for classical MPC — VQC learns a prior, classical solver refines.

---

## 4. Feasibility analysis

**Qubit budget.** Actions are low-dim (6–7 DoF, clipped to ±0.075). A T ≈ 15 rollout gives d_action_seq ≈ 90 continuous parameters. Amplitude-encoding 90 floats needs ⌈log₂ 90⌉ = 7 data qubits + ancillae; angle-encoding needs 90. Realistically **~20–30 logical qubits** at T=15, or ~10 at T=5.

**State-prep depth.** Loading 90 arbitrary amplitudes is worst-case O(2^n) gates. Even with sparse tree loaders, depth lands at ~50–200 on 20 qubits — the boundary of NISQ coherence today.

**Measurement cost.** One expectation value per candidate; 10⁴ shots × 800 candidates × 10 iters = **8 × 10⁷ shots per action**. At 10⁴ shots/s on NISQ hardware, ~2 h/action — vastly worse than 16 s classical.

**Bottleneck.** Data loading and shot count. The usual I/O wall for quantum-accelerating small-dim optimisation (Preskill, arXiv:1801.00862).

**Hardware reality check.**
- NISQ today: 50–150 noisy qubits, single-/two-qubit fidelities 99–99.9 %. Enough qubits, not enough depth × fidelity for 20-qubit amplitude loaders at useful precision.
- Tensor-network / state-vector simulator: unlimited qubits but slower than a GPU transformer forward past ~25 qubits.
- micro-kiki's actual VQC: **6 qubits, 6 StronglyEntanglingLayers, 108 variational parameters + a classical 6→11 head** (verified in `src/routing/quantum_router.py`). This is a classifier, not an optimiser, and is ~3 orders of magnitude too small in qubit count to encode a V-JEPA 2-AC action sequence.

---

## 5. What micro-kiki's existing VQC actually does (for contrast)

From `src/routing/quantum_router.py`:

- **6 qubits**, **6 StronglyEntanglingLayers** → 6 × 6 × 3 = 108 variational angles.
- Classical front-end: embedding → AngleEmbedding on the first 6 features.
- Measurement: ⟨Z⟩ on each of 6 qubits → 6 real scalars.
- Classical head: 6 → n_domains linear layer for classification.
- Training: parameter-shift rule, PennyLane `default.qubit` simulator.
- Purpose: **route a prompt to one of N micro-kiki domain experts**, not optimise anything.

Distance from "CEM accelerator":

| Need for CEM-surrogate role | Current state |
|---|---|
| Encode ≈90-dim action sequence | 6-dim angle encoding only |
| Produce scalar energy `Ẽ(a)` | Produces N class logits |
| Operate in V-JEPA 2 latent space (1024-d) | Operates on router embedding (≪ 1024) |
| Trained to match `P_φ` ranking | Trained on domain labels (cross-entropy) |

The gap is a full redesign, not a fine-tune. Anyone claiming our 6-qubit router "accelerates V-JEPA 2 planning" would be stretching the truth to breaking point.

---

## 6. Honest verdict

**Is VQC-accelerated CEM a realistic research direction for micro-kiki in 2026? Near-term: no.**

Reasons:
1. Classical CEM on V-JEPA 2-AC is already predictor-bound at 2 ms/forward batched. You beat 16 s/action by better batching, KV-cache reuse, or distilling `P_φ` to 30 M params — not by touching CEM.
2. NISQ hardware cannot deliver a useful quantum surrogate at 90-d action-sequence scale within the 2 ms/forward budget the classical path already hits.
3. Our current VQC is a 6-qubit classifier, 2–3 orders of magnitude away from even a pedagogical demo of CEM acceleration.

**What *is* scientifically interesting and within reach:**

- **Quantum-classical interface as the research object.** Treat the VQC as a *configurator* over JEPA planners (which predictor checkpoint, which rollout horizon, which exploration temperature) rather than as an optimiser inside the loop. That is a natural extension of the router we already have.
- **Quantum post-hoc critic.** Run classical CEM, then pass the top-k elite action sequences through a small VQC trained as a binary "good / resample" critic. The qubit budget fits 6–8 qubits, and the latency (one VQC eval per control step) is negligible.
- **Position micro-kiki's VQC as a research vehicle for hybrid quantum-classical *decision* interfaces**, not as a planner accelerator. This is honest, defensible, and distinct from QI-MPC which attacks full policy learning.

---

## 7. Implications for Paper A

Three options for how Paper A treats VQC-CEM acceleration:

- **Option A — short "future work" paragraph.** Acknowledge V-JEPA 2-AC's CEM cost structure, note the NISQ I/O wall, cite QI-MPC as adjacent prior art, flag the post-hoc-critic direction as ongoing. Zero implementation risk. Probably what reviewers expect.
- **Option B — minimal empirical demo.** Implement a toy: 8-qubit VQC warm-start for CEM on a downsized JEPA (e.g. a 10 M-param distilled predictor, d_action = 4, T = 5). Report wall-clock for 2–4 classical iters saved vs baseline. 2–3 weeks of work, plausible on simulator, and gives the paper a concrete contribution rather than speculation.
- **Option C — remove entirely.** Avoids the "why is quantum in this paper at all?" reviewer trap. Safest for venues that are strictly JEPA/world-model focused.

**Recommendation**: **Option A for workshop submission** (NeurIPS Workshops, ICML AutoML/Planning, QTML). The honest scope matches a 4-page workshop format. Upgrade to **Option B only if we target a main-track venue that explicitly welcomes hybrid quantum-classical contributions** (TQC, QIP workshop, or Nature Machine Intelligence for a longer piece). Do not ship Option B without at least one simulator benchmark showing *any* iteration saving — otherwise the demo hurts more than it helps.

---

## 8. References

Papers cited above:

- V-JEPA 2: Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, arXiv:2506.09985 (2026).
- Original CEM: Rubinstein, R. Y., *Optimization of Computer Simulation Models with Rare Events*, European Journal of Operational Research 99:89–112 (1997). Tutorial: de Boer, Kroese, Mannor, Rubinstein, *A Tutorial on the Cross-Entropy Method*, Annals of Operations Research (2005).
- QAOA foundational: Farhi, Goldstone, Gutmann, *A Quantum Approximate Optimization Algorithm*, arXiv:1411.4028 (2014).
- NISQ framing: Preskill, J., *Quantum Computing in the NISQ era and beyond*, arXiv:1801.00862 (2018).
- Surrogate optimisation of VQCs: Shaffer et al., *Surrogate optimization of variational quantum circuits*, arXiv:2404.02951, PNAS (2025).
- QI-MPC prior art: Khan, Al-Karaki, *QI-MPC: A Hybrid Quantum-Inspired Model Predictive Control for Learning Optimal Policies*, arXiv:2504.13041 (2026).
- CEM + QAOA (different from CEM-as-planner): Nüßlein et al., *Cross Entropy Hyperparameter Optimization for Constrained Problem Hamiltonians Applied to QAOA*, arXiv:2003.05292 (2020).

Tooling:

- PennyLane documentation, `StronglyEntanglingLayers`, `AngleEmbedding`, parameter-shift rule: https://docs.pennylane.ai

Internal references:

- micro-kiki VQC router: `src/routing/quantum_router.py` (6 qubits, 6 layers, 108 variational params + classical head).
- Paper A outline: `docs/research/paper-outline-triple-hybrid.md`.
