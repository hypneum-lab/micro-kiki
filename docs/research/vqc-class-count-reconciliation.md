# VQC Class Count Reconciliation

**Date**: 2026-04-19
**Status**: Informational / correction catalog (no fixes applied)

---

## 1. Current state (ground truth from code)

### `src/routing/quantum_router.py` — primary reference

**Lines 3–4** (module docstring):
```python
"""Quantum VQC Router — PennyLane variational circuit for domain classification.

Uses 6 qubits to classify text embeddings into 35 domain classes (34 niches + base).
```

**Line 57** (config docstring):
```python
n_classes: Number of output classes — 35 (34 niches + base).
```

**Line 64** (default):
```python
n_classes: int = 35
```

**Lines 40–41** (domain list):
```python
_NICHE_DOMAIN_LIST: list[str] = sorted(NICHE_DOMAINS)  # 34 entries
_ALL_DOMAINS: list[str] = _NICHE_DOMAIN_LIST + ["base"]  # 35 entries
```

**Conclusion**: Ground truth is **n_classes=35**, **n_qubits=6** (not 4), 35 domains (34 niches + base).

---

## 2. Evolution (git history)

Git log unavailable (stashed refs corruption), but code inspection reveals:

- **`train_vqc_router.py` (line 113)**: `NUM_DOMAINS = len(ALL_DOMAINS)  # 35` — current, authoritative
- **`train_vqc_router.py` (lines 76–111)**: 34-domain list + "base" explicitly enumerated

- **`benchmark_quantum_router.py` (line 47)**: `N_CLASSES = len(_ALL_DOMAINS)  # 11` — **STALE**. Actually only 10 niches + base = 11, hardcoded subset used for benchmarking synthetic data.

**Inference**: The codebase pivoted from an initial 11-class prototype (historical PoC) to the current 35-class router. The benchmark script was never updated after the pivot; it still operates on a reduced, synthetic 11-class domain set for local testing.

---

## 3. Stale references in docs/

### High-priority (docs that claim 4-qubit OR 11-class as current architecture)

| File | Line | Wrong claim | Correct claim | Severity |
|------|------|-------------|---------------|----------|
| `docs/superpowers/plans/2026-04-17-text-jepa-vqc-router.md` | 5 | "11-class domain classification" | 35-class routing; plan may be testing on synthetic 11-class subset | HIGH |
| `docs/superpowers/plans/2026-04-17-text-jepa-vqc-router.md` | 7 | "existing 4-qubit / 6-layer VQC router" | 6-qubit / 6-layer VQC router | HIGH |
| `docs/superpowers/plans/2026-04-17-text-jepa-vqc-router.md` | 19 | "11-class test set" | 35-class test set (or clarify if PoC intentionally uses 11-class subset) | HIGH |
| `docs/superpowers/plans/2026-04-17-text-jepa-vqc-router.md` | 205 | `n_qubits=4` in code block | `n_qubits=6` | HIGH |
| `docs/superpowers/plans/2026-04-17-text-jepa-vqc-router.md` | 340 | `n_qubits=4` in second code block | `n_qubits=6` | HIGH |
| `docs/superpowers/plans/2026-04-17-text-jepa-vqc-router.md` | 398 | "4 qubits, 6 StronglyEntanglingLayers" in spec | 6 qubits, 6 StronglyEntanglingLayers | HIGH |
| `docs/research/vqc-conditioned-aeon-predictor.md` | (§8 discussion) | Already flagged as divergence: "la réalité du code est **35 classes, softmax**… 11-classes était un prototype antérieur" | Document is **self-aware** of this discrepancy; **LOW PRIORITY** |
| `docs/specs/2026-04-16-reorientation-rationale.md` | (router table) | "Router: 32 sigmoid → 11 classifier?" | Update to reflect 35 sigmoid outputs | MEDIUM |
| `docs/paper-outline-triple-hybrid.md` | (multiple lines: 72 params, 4 qubits, 11-class task) | Claims 4-qubit VQC for "11-class routing task" | Clarify: actual router is 6-qubit / 35-class; 4-qubit was a prototype | HIGH |
| `docs/papers/spikingkiki-v3-final.md` | (4-qubit, 11 contexts) | "4-qubit VQC (72 parameters)" | 6-qubit VQC (min 72 params → larger, check actual) | MEDIUM |
| `docs/papers/spikingkiki-v3-draft.md` | (similar) | 4-qubit references | 6-qubit | MEDIUM |
| `docs/blog/2026-04-17-training-35-experts-mac-studio.md` | (4-qubit section) | 4-qubit claims | 6-qubit | LOW |

### Lower-priority (tangential or already qualified as discussion)

| File | Context | Status |
|------|---------|--------|
| `docs/paper-outline-triple-hybrid.md` | "The VQC router (72 parameters, 4 qubits) does not outperform classical sigmoid (3.4M parameters) on our 11-class routing task." | Appears to be discussing **a preliminary 4-qubit experiment**, not the shipped code. **Context-dependent fix** — clarify if this is historical or if PoC A uses this config. |
| `docs/papers/paper-a-reframe-aeon-ami.md` | References VQC as partial contribution with 4-qubit config. | **DEPENDS** on intent of Reframe Paper A — may be intentionally using reduced PoC config. |

### Comments in code

| File | Line | Text | Fix |
|------|------|------|-----|
| `scripts/benchmark_quantum_router.py` | 4 | "Generates synthetic embeddings for **11 domain classes**" | Change comment to "Generates synthetic embeddings for **11-domain subset** (dsp, electronics, ..., base) for benchmarking" |
| `scripts/benchmark_quantum_router.py` | 33–34 | "10 niche domains + base — must match router._ALL_DOMAINS" | **WRONG**: should be 34 niches + base = 35. This comment is misleading. Change to "11-domain synthetic subset for benchmark (does not match full router)" |

---

## 4. Stale references in memory files

| File | Reference | Wrong | Correct | Note |
|------|-----------|-------|---------|------|
| `project_microkiki_triple_hybrid.md` | "domain decision (4 qubits, 11 classes)" | 4 qubits, 11 classes | 6 qubits, 35 classes | **HIGH PRIORITY** — this is a top-level architecture doc |
| `project_microkiki_triple_hybrid.md` | "Classifies into 11 domains. At `src/routing/quantum_router.py`." | 11 domains | 35 domains | **HIGH PRIORITY** — directly contradicts ground truth |

---

## 5. Reconciliation plan

### Tier 1: Critical fixes (ship-blocking)

1. **`project_microkiki_triple_hybrid.md`**
   - Line with "4 qubits, 11 classes" → change to "6 qubits, 35 classes"
   - Line "Classifies into 11 domains" → change to "Classifies into 35 domains (34 niches + base)"

2. **`docs/paper-outline-triple-hybrid.md`**
   - Clarify opening: "VQC domain classifier (**6** qubits, 6 layers, **~108** parameters, PennyLane) for routing into **35 domains**"
   - For section "The VQC router (72 parameters, 4 qubits)…": Either (a) remove as outdated experiment, or (b) add context: "Early experiments used a 4-qubit variant (72 params); production uses 6 qubits and 35 classes."

3. **`docs/superpowers/plans/2026-04-17-text-jepa-vqc-router.md`**
   - Lines 5, 7, 19, 205, 340, 398: update 4-qubit → 6-qubit, 11-class → 35-class
   - OR clarify intent: if PoC A is intentionally **limiting** to 11 classes for fast iteration, state that explicitly and update all references to say "**PoC A uses 11-class subset**; production uses 35 classes."

### Tier 2: Important (documentation coherence)

4. **`docs/specs/2026-04-16-reorientation-rationale.md`**
   - Update router table from "11 classifier" → "35-class sigmoid router"

5. **`docs/papers/spikingkiki-v3-final.md`, `spikingkiki-v3-draft.md`, `blog/2026-04-17-training-35-experts-mac-studio.md`**
   - Update 4-qubit → 6-qubit
   - Consider impact on parameter count claims (72 params was for 4q; 6q will be larger)

### Tier 3: Code comments (low impact, clarity)

6. **`scripts/benchmark_quantum_router.py` comments**
   - Line 4: clarify "synthetic **11-domain subset**"
   - Lines 33–34: fix misleading comment about "must match router._ALL_DOMAINS"

### Decision tree for Tier 1 fixes

**Q**: Does PoC A (Text-JEPA VQC) intentionally use a reduced 11-class domain set for speed?

- **If YES** → add explicit note in plan: "For this PoC, we test on a reduced 11-domain subset to accelerate iteration. Production VQC handles all 35 domains. Benchmark config at line 273 uses `n_qubits=4` and `n_classes=11` for speed only."
- **If NO** → update all references to 35 classes and 6 qubits immediately.

**Current best guess**: Plan was written before the pivot to 35B-A3B (which also pivoted routing from 11 to 35 domains). The plan should reference the **current** 35-class router, but may want to add an alternative branch for PoC-speed testing on a reduced set.

---

## 6. Why it matters

### Credibility risk

Papers and planning docs claiming "4 qubits, 11 classes" will confuse readers who look at the code and find "6 qubits, 35 classes." Peer reviewers and collaborators will flag this as sloppy.

### Technical risk

If contributors build on the plan docs and expect 11 classes, they'll generate incompatible test data or benchmarks. The benchmark script already shows this pitfall: it still hardcodes 11 classes in a comment that says "must match router._ALL_DOMAINS" — but it doesn't.

### Architecture clarity

The pivots from Orion+Gemma 270M → Qwen3.5-35B-A3B and 11-domain router → 35-domain router were **correct decisions** (as noted in decision logs). But the documentation never caught up. Reconciling this improves future onboarding.

---

## Findings summary

- **Stale references found**: 15+ across docs + memory
- **Ground truth (code)**: 6 qubits, 35 classes, located in `src/routing/quantum_router.py`
- **Script benchmark mismatch**: `benchmark_quantum_router.py` still uses synthetic 11-class subset (intentional for speed); comments are misleading
- **Memory (stale)**: `project_microkiki_triple_hybrid.md` needs urgent fix
- **Plan file (stale)**: `2026-04-17-text-jepa-vqc-router.md` has mixed configs (n_qubits=4 in code examples but should be 6)
- **Paper outline (unclear)**: References to "4 qubits, 72 params" may be discussing an older experiment or may be stale

**Recommended cleanup scope**: Single commit (minimal risk) — fix memory doc + key plan doc + one paper-outline clarification, leave benchmark script and older papers for a separate, lower-priority doc-review pass.

