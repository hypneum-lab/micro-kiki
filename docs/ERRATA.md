# Errata

Dated corrections to claims made in this repository (README, `paper/`,
`docs/`, model cards). Each entry records the claim as published, where it
appeared, the root cause, the correction applied and what remains open.
Entries are never deleted; later corrections are appended.

Policy (phase 0 "truth", milestone M0, PI decision of 2026-10-25): a number is
reported only if a committed artifact under `results/` (or a dated external
measurement) reproduces it. Analytical estimates are labelled as such and are
never phrased as measurements. No "saves energy" / "energy-efficient" wording
without a measurement (lexical rule T12 of the hypothesis catalogue).

Source of the review: hypneum-lab/.github,
`docs/research/2026-10-24-revue-hypotheses/` (report §2.6, §7.1, §7.5;
master table `synthese/01-table-maitresse-hypotheses.md`). Tracking issues:
[#10](https://github.com/hypneum-lab/micro-kiki/issues/10),
[#11](https://github.com/hypneum-lab/micro-kiki/issues/11),
[#12](https://github.com/hypneum-lab/micro-kiki/issues/12).

---

## E-2026-10-25-01 — SNN energy: "0.032 mJ/token" and "91.6×" (issue #10)

**Claim as published.** "35B MoE on Loihi-2 (theoretical): 0.032 mJ/tok.
91.6x ops reduction via LAS conversion. Efficiency score: 27B dense = 2.23,
35B MoE = 0.055."

**Where.** `README.md` §"SNN energy estimate";
`docs/paper-outline-triple-hybrid.md` §5.3 and §8.2; `MODEL_CARD-v0.3.md`
("40-60% theoretical energy reduction", "Energy ratio 0.6 — achieved 0.6",
"energy-efficient" tag). `paper/micro-kiki.tex`, `CITATION.cff` and the
published Hugging Face cards do not carry the figures. There is no
`.zenodo.json` in this repository.

**Root cause.** An analytical operation count was presented as a
measurement. `scripts/eval_spikingkiki_35b.py` →
`results/spikingkiki-35b-eval.json` (2026-04-19) divides 58.04 G MACs/token by
633.75 M assumed spike-ops/token (= 91.59×) and multiplies by an assumed
energy per spike-op on Loihi 2 (31.687 µJ/token ≈ 0.032 mJ/token). The
artifact itself records `"status": "theoretical (no weight conversion yet)"`.
No ANN→SNN conversion of the 35B model was ever run, and nothing was executed
on neuromorphic hardware.

**External evidence (2026).** Measurements of spiking LLMs go the other way:
SymbolicLight (194M parameters, trained from scratch) consumes about 3× more
energy and runs about 4× slower on GPU than its dense counterpart; SpikingBrain
2.0 reports −46.5 % only on dedicated silicon; the quality cost of spiking is
0.23–0.34 nats at equal FLOPs (arXiv:2610.15146 — a quality cost, not joules).

**Correction.** Both figures removed everywhere listed above and replaced by:
"not measured on the target system; external 2026 measurements unfavourable;
J/token to be measured (NH-81)". The analytical artifact and script are kept
as a record of the estimate (the script docstring now says so), not as
evidence.

**Open.** Hypothesis NH-81 ([#14](https://github.com/hypneum-lab/micro-kiki/issues/14)):
measure J/token, spiking vs dense, at iso-quality. No new BSK run before that
measurement (roadmap §7.5).

---

## E-2026-10-25-02 — C5 information bound "0.911" (issue #11, master table MK-23)

**Claim as published.** "Holevo + Fano upper bound acc_max ≈ 0.911 for a
4-qubit VQC on 10 classes, at MI = 2.04 bits on PCA-4 features; our measured
0.246 is consistent with but far from the ceiling."

**Where.** `docs/paper-a/c5-info-bound.tex` (§Numerical verification, figure,
§What the bound does NOT say); `docs/paper-a/paper-a-v2.tex` (abstract,
§Findings item 4, §Discussion); `docs/paper-a/c2-downstream-results.md`;
`docs/paper-a/c3-corpus-validation.md`; `docs/paper-a/CLAUDE.md`;
`docs/superpowers/plans/2026-04-19-phase-c-roadmap.md`;
`results/c5-info-bound.json`. Master table of the 2026-10-24 review: row
MK-23 "confirmé (cohérent, non serré)".

**Root cause.** `src/routing/info_bound.py::estimate_mi_bits` (lines 63–70
before this fix) returned `mutual_info_classif(X, y).sum()`, i.e. the sum of
the *per-feature marginal* mutual informations. That quantity is not an
estimate of the joint I(X; Y): it grows with the number of (redundant)
features and is unbounded, whereas I(X; Y) ≤ H(Y) ≤ log₂ K = 3.32 bits for
K = 10. The committed run reached 150.74 bits at d = 384
(`results/c5-info-bound.json`, key `"384"`). The "2.04 bits at d = 4" fed into
the Fano bound is the same invalid quantity, so 0.911 is not a bound on
anything. A second, structural point: with N = 4 qubits and K = 10 classes,
N ≥ log₂ K, so the Holevo cap never binds — the construction carries no
quantum-information content beyond the data-dependent MI estimate (this is
close to the "C5 kill" condition the phase-C roadmap itself listed:
"the 4-qubit ceiling is trivially derivable").

**Correction (code).** `estimate_mi_bits` now estimates the joint I(X; Y)
with the Ross (2014) k-NN estimator applied to the full feature vector
(Chebyshev metric, unit-variance scaling and tie-breaking jitter as in
scikit-learn) and caps the result at the empirical label entropy H(Y),
emitting an explicit `InfoBoundWarning` when the raw estimate exceeds it.
`acc_upper_bound` applies the same cap (effective MI = min(MI, N, H(Y))). A
new helper `label_entropy_bits` exposes H(Y). Regression tests in
`tests/routing/test_info_bound.py`:
`test_joint_mi_estimate_never_exceeds_label_entropy` fails on the previous
formula (≈ 166 bits on 50 redundant copies of a 3.32-bit label);
`test_sum_of_marginal_mi_violates_label_entropy` documents the defect;
`test_acc_upper_bound_caps_mi_above_label_entropy_with_warning` replays the
150.74-bit input.

**Correction (documents).** The 0.911 figure is withdrawn in every document
listed above. MK-23 is requalified from "confirmed" to **refuted**: the
derivation is invalid and no valid bound has been established. Formula
retained for any future statement: acc_max = 1 − max(0, H(Y) − min(I, N, H(Y))
− 1) / log₂(K − 1), with I the *joint* MI.

**Open.** Recomputing `results/c5-info-bound.json` and
`docs/paper-a/c5-bound-figure.pdf` with the corrected estimator requires the
unversioned MiniLM embedding cache `results/.c1-cache.npz` (regenerated by the
C1 pipeline, `scripts/bench_info_bound.py` then runs unchanged). Until then the
JSON and figure stand as the record of the invalid run and must not be cited.
The master-table rows in hypneum-lab/.github
(`synthese/01-table-maitresse-hypotheses.md`, MK-23;
`inventaires/babybrain-microkiki-quantum.md`, MK-23) are to be updated in that
repository.

---

## E-2026-10-25-03 — Router accuracy "70.7 % top-1 / 89.5 % top-3" (issue #12, master table MK-01)

**Claim as published.** "Router V4 (mpnet 768d, 31 merged classes): top-1
70.7 %, top-3 89.5 %; V3 (MiniLM 384d): 46.4 % / 69.6 %; improvement
+24.3 pp / +19.9 pp."

**Where.** `README.md` (status line, pipeline diagram, §"Router V4 accuracy",
§"Current work"); `paper/micro-kiki.tex` (abstract, contributions, §4.1
architecture and footnote, §4.2 domain merging, §4.3 training, Table
`tab:router`, §6.1, conclusion); `scripts/sync_hf.py` (router model-card
template).

**What the committed artifacts say.** `results/router-v4-eval.json` (commit
`0f425d5`, 2026-04-21): `all-MiniLM-L6-v2` 384d, 34 domains, 5 377 validation
samples, top-1 0.4642, top-3 0.6956, per-domain accuracy and top confusions.
The published Hugging Face card `clemsail/micro-kiki-router-v4` (2026-04-22)
reports the same 0.464 / 0.696 on a 21 510 / 5 377 split, 34 domains.
`output/router-v4/meta.json` at HEAD (commit `7f2278b`): MiniLM-L6 384d, 35
outputs, `BCEWithLogitsLoss`, 30 epochs. The deployed server
(`src/serving/full_pipeline_server.py`) wraps that MiniLM router (34 outputs).

**Provenance of 70.7 / 89.5.** Commit `10c7b6e` (2026-04-24, "feat(router):
retrain with clean data") committed `all-mpnet-base-v2` 768d weights with 31
merged classes (cross-entropy, inverse-frequency class weights) and stated in
its message only: "57K train / 14K valid (cleaned + augmented) — top-1 70.7%,
top-3 89.5% (was 46.4%/69.6%)". No evaluation JSON, no confusion matrix and
no validation split were committed (`data/router-v4/` is gitignored). The same
day, merge `bf986ba` resolved `output/router-v4/` back to the MiniLM-L6
weights, so the mpnet weights survive only in history. Commit `ceda3d9`
(2026-04-24) then wrote 70.7 / 89.5 into README and paper as "V4" and
mislabelled the committed 46.4 / 69.6 as "V3". Both numbers come from V4
trainings; they differ by protocol (embedding, 34 vs 31 classes, 5 377 vs
~14 K validation examples drawn from a different, augmented dataset), not by
version. The paper's own footnote already admitted "we have not reconciled
them".

**Correction.** Every occurrence replaced by the artifact-backed values
46.4 % / 69.6 % (MiniLM-L6 384d, 34 domains, N = 5 377). The 70.7 % / 89.5 %
figure is withdrawn: it is not reproducible from the repository. The paper's
§4.2 now describes the merge map actually used by `10c7b6e` (components,
electronics, power → electronics-hw; kicad-dsl, kicad-pcb → kicad;
spice-sim → spice) as a withdrawn variant; the previously printed merge list
(electronics-design, spice-models) did not match the code. The ablation ratio
"2.2× over keyword matching" is unaffected: it comes from
`results/ablation-router/router-ablation.json` (29.4 % vs 13.2 %).

**Open.** To reinstate an mpnet router: retrain with
`scripts/train_router_v4.py`, commit `output/router-v4/` together with the
`--mode eval` JSON on a committed (or hashed) validation split, and report
both protocols side by side. MK-01 in the master table stays "non concluant
(provenance)" until then; the row's measured value is 46.4 / 69.6.
