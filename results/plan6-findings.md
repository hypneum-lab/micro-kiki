# Plan 6 — VQC Reproducibility: final findings

**Date:** 2026-04-19
**Branch:** `poc/torch-vqc-mps`
**Method:** Torch VQC (3000× faster than PennyLane parameter-shift) used to execute
Plan 6 tasks 5-7 in minutes instead of days.

## TL;DR

**The Task 14 accuracy of 0.925 is not reproducible**, on identical code, identical
data, with a rigorous modern VQC training loop. The "97% retention at 3× compression"
claim in Paper A §4.4 is built on a non-reproducible reference number.

## Investigation trail

### Task A: git archaeology

- `quantum_router.py` at commit `8bdd376` (Task 14 timestamp, 18 apr 13:51) is
  **byte-identical** to HEAD. Same `QuantumRouterConfig` defaults.
- `eval_text_jepa_vqc.py` at `a804daa` (first commit of script) is **byte-identical**
  to the version that produced Task 15.5. Same `n_qubits=4, n_layers=6`, same split.
- Dataset `data/final/<domain>/train.jsonl` unchanged since before Task 14.

**Conclusion**: nothing in git explains the gap. Task 14 must have been run with
uncommitted local modifications.

### Task B: n_qubits sweep (20 runs × 200 epochs in 22 min on Studio)

```
n_qubits=4:  test_acc = 0.082 ± 0.025
n_qubits=6:  test_acc = 0.080 ± 0.033
n_qubits=8:  test_acc = 0.076 ± 0.016
n_qubits=10: test_acc = 0.058 ± 0.013
```

All configurations stuck near chance (0.10 for 10 classes). More qubits does NOT
help. The architectural limit is **VQC only sees `features[:n_qubits]`** — the
first few dims of MiniLM mean-pooled embeddings don't carry domain-discriminating
information.

### Task C: Task 14 config reconstruction

Decoded Task 14's reported `n_params=20` + `n_test=80`:

| n_classes × n/domain | n_test | nq=4 n_layers=0 params |
|---|---|---|
| 4 × 100 | 80 | **20** ✓✓ exact match |

So Task 14 was run with:
- **4 classes** (not 10): dsp, electronics, emc, embedded (likely)
- **n_layers=0** (no variational layers — just angle embedding + measurement + linear head)
- **100 samples/domain** (400 total)

Reproducing this config with torch VQC:

```
nq=4 nl=6 n_params= 92 → test_acc=0.212
nq=4 nl=0 n_params= 20 → test_acc=0.200  ← Task 14 config reconstructed
nq=2 nl=1 n_params= 18 → test_acc=0.200
nq=3 nl=1 n_params= 25 → test_acc=0.200
```

**Task 14's 0.925 is NOT reproducible.** Current rigorous run gives ~0.20 (chance
for 4 classes).

## Implications

1. **Paper A §4.4** claim of "97% retention at 3× compression" compares Text-JEPA
   at 0.900 vs baseline at 0.925, where 0.925 is an irreproducible number. The
   ratio argument is invalid.

2. **Plan 6's kill criterion is inverted**: it was designed to detect non-determinism
   in the pipeline. Actual result: the pipeline IS deterministic (matched grid v3 +
   torch sweep confirm baseline stability at 0.19 ± 0.01). The gap is in the
   historical number, not the pipeline.

3. **The real finding**: 4-qubit VQC on top of MiniLM embeddings cannot do
   10-class routing. The information isn't in `features[:4]`. No amount of qubits
   or layers rescues this — you'd need a different encoding (PCA to n_qubits dims,
   or a learned projection).

## What this unlocks (mid-term)

With torch VQC at 3000× speedup, we can now actually do the experiments Paper A
needs to be main-track:

- Learned projection layer (384 → n_qubits) instead of truncation: tractable to sweep
- n_qubits × n_layers grid across 5 seeds: 15 min instead of 6 weeks
- DataEmbedding ablation (MiniLM vs BERT vs domain-tuned): hours instead of quarter
- Proper Bayesian bootstrap CIs on every reported number: trivial

## Recommendations

1. **Retract §4.4 numerical claims** in Paper A draft — keep the conceptual
   argument but remove the "97% retention" line until we have a reproducible
   baseline.
2. **Pivot Paper A** from "compression ratio" frame to "torch-native batched
   VQC training" frame — this is the real contribution: making VQC research
   cheap enough to iterate on.
3. **Do real-projection experiments** before submitting anywhere.

## Update (2026-04-19, later): learned projection rescues the architecture

Added optional `input_dim` to `TorchVQCRouter` — inserts a learned linear layer
`(input_dim → n_qubits)` followed by `π·tanh`, trained jointly. This replaces
the naive `features[:n_qubits]` truncation with a class-discriminative projection.

Results on 10-class real data (500 samples, 400 train, 100 test, 300 epochs):

| Config | Test acc (trunc) | Test acc (+ proj) | Δ |
|---|---|---|---|
| nq=4, nl=6 | 0.090 | **0.300** | **+21 pt** (3× chance) |
| nq=6, nl=6 | 0.070 | 0.190 | +12 pt |
| nq=8, nl=6 | 0.110 | 0.200 | +9 pt |

**The architecture is rescued.** Best is counter-intuitively the smallest qubit
count (nq=4 + proj = 0.300) — more qubits optimizer-harder on 400 samples, and
the extra capacity overfits (train 0.40 / test 0.30 for nq=4+proj already).

This is the **honest main-track-publishable contribution**: torch-native batched
VQC + learned projection makes quantum classifier research on real embeddings
tractable, at 3000× speedup vs PennyLane. The "97% retention" claim in Paper A
is still invalid — but the direction (VQC + compression) is alive again under
a proper framing.

## Update 2 (2026-04-19, evening): 3-axis rescue ablation

36 runs on Studio (~5 min) over {data=50 vs 500/dom} × {wd=0, 1e-4, 1e-3} × {linear vs MLP proj} × 3 seeds, 10 classes, nq=4, nl=6, 300 epochs.

**Headlines**:
- **Best single run: 0.390 test_acc** (50/dom + wd=1e-4 + linear + seed 0). Train=0.435, gap=0.045 — not memorizing.
- **Median: 0.250**, range 0.170–0.390. Huge seed-to-seed variance → optimization landscape rugged.
- **Linear always beats MLP** by 7.2 pt mean (likely `π·tanh` saturates MLP hidden units).
- **10× more data adds only 3.8 pt** mean — 4-qubit info channel is the real bottleneck.
- **wd=1e-4 linear is the sweet spot** (0.304 mean across data sizes).

**Upper bound estimate**: 4× chance (~0.40) appears to be the ceiling for this architecture on 10-class routing. A classical linear probe on MiniLM would hit 0.7-0.8. The VQC is NOT competitive classically — but is a rigorous quantum-ML benchmark now.

**Final recommendation for Paper A**:
- Contribution #1 (methodological): torch-native batched VQC @ 3000× over PennyLane.
- Contribution #2 (architectural): learned projection rescues VQC from uselessness on arbitrary pretrained embeddings.
- Contribution #3 (empirical): thorough ablation showing linear proj + small wd is optimal, MLP and more data give diminishing returns due to information-capacity ceiling at 4 qubits.
- **Drop any "competitive with classical" claim**. Frame as "enabling rigorous quantum-classifier research" rather than "winning".
