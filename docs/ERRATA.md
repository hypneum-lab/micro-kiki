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

**Update 2026-10-28 (NH-81 measured).** On the macM1 GPU rail, a 26.5 M
BSK spiking LM at iso-quality with its dense twin (test gap 0.007 nats)
uses as much or more energy per token: R = J_dense/J_spiking = 0.91
(95 % CI 0.83-1.01) in generation, 0.73 in batched scoring. The energy
argument is withdrawn for GPU/M1-class hardware
(`docs/results/2026-10-25-nh81-joules.md`); NVML replication on the RTX
4090 pending.

**Addendum 2026-10-31 (correction to the root cause).** The sentence "No
ANN→SNN conversion of the 35B model was ever run" is wrong. A partial
conversion of Qwen3.6-35B-A3B was run on 2026-04-21
(`results/spikingkiki-35b-convert.json`, `"status": "ok"`, 8 543 of 31 070
layers converted; `docs/research/2026-04-21-spikingkiki-c2-moe-blocker.md`).
It was never evaluated: the repository holds only module-level sanity
checks (`results/spiking-e-validation*.json`), no quality or energy
measurement. Nothing ran on neuromorphic hardware, so the energy
withdrawal is unchanged. Concerned Hub repos: `clemsail/spikingkiki-35b-a3b-v4`
(partial upload, card not updated) and `hypneum-lab/spikingkiki-v4-lif-metadata`
(card updated on 2026-10-31). Found during an independent review of the
Hugging Face cards, 2026-10-31.

**Update 2026-10-28, evening (verdict status; KXKM replication).** PI
decision: the macM1 verdict reads **"withdraw, provisional (IOReport)"**
until the preregistered NVML replication on the KXKM RTX 4090
(`docs/prereg/2026-10-28-nh81-kxkm-replication.md`, tag
`prereg-nh81-kxkm-20261028-r2`) decides: agreement (KXKM R <= 1.1 in
generation, fused path) makes it definitive; divergence keeps it
provisional and is reported. First KXKM attempt (2026-10-28 16:22-19:24
CET): **not measured** — the start guard never passed (a crash-looping
`llama-server` user service, then N10-C freeze jobs of another session);
iso-quality reproduced on the GPU to 5e-5 nats
(`docs/results/2026-10-28-nh81-kxkm-nvml.md`). Second attempt 20:18
CET, after the service was stopped and N10-C ended: refused again; at
idle the RTX 4090 sits in P8 and NVML reports ~40 % utilisation from a
light constant background, above the 10 % guard. Status unchanged:
withdraw, provisional (IOReport); replication still pending, guard
definition for KXKM to be decided by the PI.

**Update 2026-10-30 (third platform, M3 Pro).** Preregistered
replication on macM3 (Apple M3 Pro, IOReport GPU rail, tag
`prereg-nh81-m3pro-20261030`): W1 generation, fused path, R = 0.670
(95 % CI 0.662-0.677), 12/12 valid pairs, iso-quality reproduced to
2e-5 nats; every block "withdraw" (`docs/results/2026-10-30-nh81-m3pro.md`).
Agreement with macM1, so no veto; fixed in advance as descriptive, it
cannot make the verdict definitive. Status: withdraw, provisional
(IOReport); KXKM NVML replication (amendment r4) still pending.
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

**Resolved 2026-11-08 (PR #39).** `results/c5-info-bound.json` and
`docs/paper-a/c5-bound-figure.pdf` were regenerated with the joint k-NN
estimator (code fix d25e701) from the C1 embedding cache
(`scripts/bench_info_bound.py`, unchanged). MI before -> after: PCA-4
2.04 -> 1.167 bits, PCA-16 -> 1.556, PCA-128 -> 0.351, d = 384
150.74 -> 1.426 bits; `bound_vqc_4qubit` 0.9110 -> 0.6356. All values are
now below H(Y) = 3.32 bits. PI decision of 2026-11-08 (refs
[#11](https://github.com/hypneum-lab/micro-kiki/issues/11)): MK-23 stays
**refuted**. The requalification concerned the published statement (Holevo+Fano
bound 0.911 for a 4-qubit VQC), whose derivation was invalid. The corrected
value (`bound_vqc_4qubit` 0.6356 at joint MI 1.167 bits on PCA-4) is a
different statement: a plain Fano/DPI bound, since the Holevo cap never binds
(N = 4 >= log2 K = 3.32 bits). This matches the hypothesis catalogue v3 of
hypneum-lab/.github (MK-23, "reformulated"). Catalogue v3 (reference since
2026-11-07) already carries the reformulation; the 2026-10-24 master table
(`synthese/01-table-maitresse-hypotheses.md`,
`inventaires/babybrain-microkiki-quantum.md`) is historical.

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

---

## E-2026-11-08-01 — Text-JEPA "97 % retention at 3x compression" (0.925 -> 0.900) (master table MK-20)

**Claim as published.** "3x embedding compression (384 -> 128 dims) retains
97 % of VQC routing accuracy (baseline 0.925, Text-JEPA 0.900) on a 10-domain
classification", with a 10-class VQC test set of 80 samples.

**Where.** `results/text-jepa-vqc.md` (lines 16-19) and
`results/text-jepa-vqc.json` (Task 14); `docs/papers/paper-a-draft-v1.md`
(lines 12, 36, 115, 165, 177-184, 221, 255, 275, 348) and its French twin
`docs/papers/paper-a-draft-v1-fr.md` (lines 14, 38, 117, 167, 179-186, 223,
257, 277, 350); `docs/papers/paper-a-reframe-aeon-ami.md` (lines 21, 29, 56,
78) and `-fr.md` (lines 23, 31, 58, 80); `docs/SESSION-2026-04-19-INDEX.md`
(lines 86, 96). Master table row MK-20 of the 2026-11-06 catalogue v3:
"retracted".

**What is true.** The 0.925 baseline is not reproducible: on identical code
and data a rigorous torch VQC loop gives about 0.20 on the reconstructed
Task 14 configuration and 0.08 +/- 0.03 at 4 qubits on 10 classes (chance is
0.10), so the ratio 0.900 / 0.925 compares Text-JEPA against an
irreproducible reference (`results/plan6-findings.md`, lines 10-12, 63-70).
Task 14's `n_params = 20` and `n_test = 80` match 4 classes x 100 samples
with no variational layer (`n_layers = 0`), not the 10 domains written in
the JSON; this reconstruction is an inference from the parameter count, not
a recorded setting. `results/plan6-findings.md` (2026-04-19) already
recommended retracting the Paper A section 4.4 numbers; that retraction
was never carried into the files above.

**Root cause.** Baseline missing: the headline ratio depended on a single
run whose configuration was not committed (the findings note states Task 14
"must have been run with uncommitted local modifications"; the scripts and
data are byte-identical to HEAD and do not reproduce it). The retraction
lived only in a findings note, not in the documents that carry the claim.

**Correction.** This entry. The claim "97 % retention at 3x compression"
is withdrawn; no Text-JEPA compression-retention figure is established by
this repository.

**Open.** The occurrences listed under "Where" still carry the figure and
must be amended (flag or remove) in a follow-up PR; this entry does not
edit them. A valid test needs a committed configuration, >= 5 seeds, a
bootstrap CI and the learned-projection baseline of
`results/plan6-findings.md`. Refs hypneum-lab/.github#10.

---

## E-2026-11-24-01 — Aeon negative note v1: "synthetic gains that do not survive real conversational streams" (master table MK-11)

**Claim as published.** Title: "Anti-collapse choices in a ~267K-parameter
latent predictor: synthetic gains that do not survive real conversational
streams". Abstract: the +20.6 % MRR obtained with shared centering on
synthetic streams "do[es] not transfer" to real 10-domain data; MK-11 is
"refuted (real data)". Section 5 explained the gap by the behavior of
centering on real embeddings (hypotheses H2, H3).

**Where.** `docs/notes/2026-11-08-aeon-anti-collapse-negative-note.md`,
version 1 (commit `3bf44c1`, PR #45), deposited unchanged on Zenodo on
2026-11-24 as 10.5281/zenodo.24496729 (concept DOI
10.5281/zenodo.24496728). Pointers added by PR #46 carried two of the
errors: "~2 MB float64" (`docs/papers/paper-a-draft-v1.md` line 15 and
`-fr.md` line 17; `docs/papers/paper-a-reframe-aeon-ami.md` line 83 and
`-fr.md` line 85) and "null == prédictif sur données réelles" attached
to the centering claim (`paper-a-reframe-aeon-ami.md` line 18, `-fr.md`
line 20). The other pointers of PR #46 ("null == predictive" for
condition D, "negative vs null" for the soft interleaved run, in
`docs/SESSION-2026-04-19-INDEX.md` and the same drafts) stand.

**What is wrong.** Same numbering as the "Changes from version 1"
section of the note. Items 1-6: independent review of 2026-11-24.
Items 7-9: found while preparing version 2. Items 10-11 and the parts
marked "(later review)": two further reviews, of drafts of version 2
(PR #48). Each item was checked against the code at commit `ba5b1f5`,
the result files or the cited source.

1. Centering was never run on real data. The four files
   `results/aeon-realdata.json`, `-within-topic`, `-soft-interleaved`,
   `-soft-withintopic` record `"use_centering": false,
   "use_layernorm_delta": true`. The title, the abstract and H2/H3
   compared a centering run with LayerNorm(delta) runs.
2. arXiv:2206.15331 is "GitHub Copilot AI pair programmer: Asset or
   Liability?" (Moradi Dakhel et al.), not LeCun (2022), which is an
   OpenReview position paper (forum id `BZ5a1r-kVsf`). The wrong id
   appeared four times, in the note only.
3. arXiv:2511.08544 (LeJEPA) is by Balestriero and LeCun, not "Xiong et
   al.".
4. Mechanisms misdescribed. Centering is subtracted from the output
   (`src/memory/aeon_predictor.py:141-162`), not from the input.
   LayerNorm normalizes the residual predicted by the MLP (lines
   134-140), not h_{t+1} - h_t. `backward_cosine` recomputes
   `h_hat = x + delta` without either (lines 184-186), and
   `_ln_gamma` / `_ln_beta` are never updated: neither mechanism enters
   training. `final_train_loss` is identical across C, D, F1, L1, L3
   (0.5667058005928993) and across F2, L2 (0.5203294903039932). The
   targets are fixed vectors, so the collapse that such devices guard
   against in joint-embedding training cannot occur here; "anti-collapse"
   was a misnomer. (Later review:) "stateless (no EMA teacher)" and
   "inspired by DINO/DINOv3 self-distillation": the running mean is an
   exponential-moving-average state, and DINO (arXiv:2104.14294, section
   3.1) centers and sharpens the teacher output inside the training
   loss and never centers the features that are evaluated, whereas here
   centering is applied only at evaluation. "Trained online": in these
   runs training is in batch, 50 or 300 epochs after ingestion.
5. Null predictor misdescribed as "random" / "wrong" / "incorrect" /
   "arbitrary" stack id, in six places. It is `stack_id = -1`
   (`scripts/eval_aeon_predictor.py:201-202`,
   `scripts/eval_aeon_realdata.py:283-284`), an all-zero stack input
   (`aeon_predictor.py:408-409`) that never occurs in training. With
   per-stack centering it also uses a separate running mean (line 147).
   `predict_next` goes through `forward` (line 389), which updates the
   running mean at every query, so the predictive and null calls share
   mutable state.
6. Real-data protocol misdescribed. The stream is made of the first
   user message of each of 100 records per domain, shuffled, then
   interleaved or concatenated (`scripts/eval_aeon_realdata.py:101-150`);
   consecutive items are not conversation turns. Exact-match MRR below
   0.005 is the chance level, not a sign that "the latent palace is too
   sparse". The soft within-topic baseline of 0.99 follows from the
   construction. All 1 000 items, queries included, are ingested and
   trained on (lines 226-235), although the docstring says "train on
   first 80%". (Later review:) the recommendation "N >= 10,000 turns"
   rested on that misreading; the next-domain baseline is "stay on the
   current domain" and is 0.00 because all 100 evaluated transitions
   change domain, not because "raw retrieval cannot classify
   next-domain by design".
7. Learning rate of F2 and L2 omitted: 5e-3, not only 300 epochs. The
   April drafts say so (`docs/papers/paper-a-draft-v1.md`, Table 2);
   the JSON files record neither. A re-run on 2026-11-24 (seed 0; 50
   epochs at 1e-3, or 300 epochs at 5e-3) reproduces all seven
   synthetic files to the last digit; 300 epochs at 1e-3 does not
   (`results/2026-11-24-aeon-centering-query-order.json`, key
   `training_settings_check`).
8. In the synthetic stream the stack id stored with turn i is the stack
   of the transition that produced turn i
   (`scripts/eval_aeon_predictor.py:150-156`), not of the transition to
   be predicted, which is drawn independently. The statement that
   condition L2 "genuinely uses stack information" is withdrawn.
9. The query turn is stored in the index and raw retrieval returns it
   first (100 of 100 queries), so the gold turn ranks second at best
   and the baseline MRR cannot exceed 0.5. The scores 0.413 and 0.498
   are readings on that scale: gold turn second on 48 queries (raw
   retrieval, where turn i-1 is second on the other 52) and on 99
   (condition D).
10. The gain of condition D comes from state updated during evaluation.
    Diagnostic `scripts/diag_aeon_centering_query_order.py`, output
    `results/2026-11-24-aeon-centering-query-order.json` (seed 0, same
    weights): predictive / null MRR is 0.498 / 0.498 as run, 0.410-0.430
    / 0.412-0.417 with the queries shuffled (five orders), 0.162 / 0.152
    in reverse order, 0.423 / 0.417 with the running mean frozen at its
    post-training value. Without any predictor, the raw query minus an
    exponential moving average of the previous queries gives 0.498, with
    the same rank of the gold turn on every query. Per-stack centering
    (F1, F2): the null's 0.495 and 0.498 fall to 0.410-0.425 with the
    queries shuffled or the running mean frozen. The explanation given in
    version 1 ("a regime effect where the centred embedding space
    saturates") is withdrawn.
11. "Module 7" is not a numbering of LeCun (2022), which describes six
    unnumbered modules; it comes from
    `docs/papers/paper-a-reframe-aeon-ami.md`.
12. Smaller points: "~2 MB in float64" (weights are float32, about
    1 MB); "HNSW store" (the index is `src/memory/atlas.py`, exhaustive
    search); the std-ratio rollback lives in `AeonSleep.sleep_cycle`, is
    not run by the evaluation scripts, and compares the output with the
    targets, not with h_t (later review); +20.6 % is relative to raw
    retrieval in the same run, not to a "no-centering baseline"; the
    power statement of section 7 (item 1) had no calculation behind it;
    internal references a reader could not resolve; the bullet "The
    null predictor is a meaningful upper bound"; the header cited the
    unrelated erratum E-2026-10-25-02 (later review).

**What is true.** In synthetic condition D, `predictive_mrr` equals
`null_stack_mrr` (0.4983 = 0.4983) and `win_rate_stack_vs_null` is 0.0:
the +20.6 % over raw retrieval (0.4133) is not attributable to stack
conditioning, and item 10 attributes it to a running mean updated by
the queries in stream order. In the four real-data pilots, which test
the LayerNorm(delta) variant only, the conditioned predictor shows no
benefit over the null that survives the limits of the protocol: the
null scores higher in the soft interleaved run (0.1765 vs 0.1628); the
conditioned predictor scores higher in the soft within-topic run
(0.1093 vs 0.0687), where both are far below a baseline of 0.99 that
the construction makes trivial. All runs are single-seed and
exploratory. Whether the synthetic centering gain transfers to real
data was not tested.

**Root cause.** The metric values of version 1
match the JSON files; the errors are in what the text says about them.
Recorded facts: PR #45 states that the numbers were copied from the
JSON keys, with no code change and no re-run; it was merged on
2026-11-08 with no review and no comment on GitHub; the JSON files
record neither seed, epochs nor learning rate; the evaluation-time
update of the running mean leaves no trace in any output. Version 1 was
drafted with LLM assistance and merged the same evening; its prose was
not checked against the code or the run configurations.

**Correction.** The note is replaced in place by version 2
(2026-11-24), which carries a "Changes from version 1" section, a new
title and abstract, and code-based descriptions of the mechanisms and
protocols. The values carried over in its Tables 1 and 2 are unchanged.
One diagnostic script, its smoke test and its JSON output are added
(item 10); the script re-scores existing conditions with the April
training settings, plus one training at 300 epochs and 1e-3 for the
settings check of item 7. Dated annotations are appended, nothing
deleted: three in `docs/SESSION-2026-04-19-INDEX.md` (real-data runs;
centering row; L2 row) and one after each of the six pointers listed
under "Where". No existing result file, script or source file is
modified.

**Open.** (a) Zenodo: record 10.5281/zenodo.24496729 still carries the
files of version 1; its description was edited on 2026-11-24 and lists
three known errors (items 1-3). Depositing version 2 as a new version
is for the PI. (b) Catalogue v3 (hypneum-lab/.github) labels MK-11
"réfuté (données réelles)" on the basis of `results/aeon-realdata.json`,
a run without centering; whether to relabel is a PI decision. (c) The
docstrings of `scripts/eval_aeon_realdata.py` ("train on first 80%") and
`scripts/eval_aeon_predictor.py` ("held-out indices", stack id of the
transition) do not match the code; not edited here. (d) The paper
drafts under `docs/papers/` still describe centering and
LayerNorm(delta) as training-time anti-collapse mechanisms and number
LeCun's modules; not edited here. (e) No valid test of either transform
exists; requirements are listed in section 8 of the note.
