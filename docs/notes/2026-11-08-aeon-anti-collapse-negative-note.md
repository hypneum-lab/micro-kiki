# Anti-collapse choices in a 100K-parameter latent predictor: synthetic gains that do not survive real conversational streams

**Date:** 2026-11-08
**Status:** Negative note — single seed throughout; all findings are exploratory
**Hypothesis catalogue:** hypneum-lab/.github `docs/research/2026-11-06-refresh-hypotheses/catalogue-v3.md`
**Related issues:** hypneum-lab/micro-kiki #13 (NH-85, Aeon vs raw RAG baseline, deferred post-M3)
**Errata:** E-2026-10-25-02 (MK-23, C5 bound), E-2026-11-08-01 (MK-20, Text-JEPA retracted)

> Résumé (FR) :
> En avril 2026, le prédictor latent Aeon (LatentMLP numpy, ~267 K paramètres) montrait +22 % MRR
> sur des flux synthétiques structurés avec centrage DinoV3 (hypothèse MK-11 du catalogue). Quatre
> évaluations sur données réelles (10 domaines, 1 000 tours) infirment ce gain : MRR exacts proches
> de zéro dans tous les cas, et dans le scénario le plus favorable (soft-domain interleaved) le null
> predictor dépasse le prédictor conditionné (MRR 0,177 vs 0,163). Le verdict catalogue de MK-11
> est « réfuté (données réelles) ». Toutes les expériences sont à graine unique et sans
> pré-enregistrement ; aucune affirmation ne dépasse le stade exploratoire.

---

## Abstract

A 100K-parameter numpy MLP (LatentMLP, `src/memory/aeon_predictor.py`) was trained
on synthetic conversational streams in April 2026. Under stack-structured streams with
shared DinoV3-style centering, predictive MRR rose from 0.413 to 0.498 (+20.6 %) relative
to a no-centering baseline, a result cited as evidence for hypothesis MK-11 and framed as
alignment with LeCun's Module 7 (AMI, arXiv:2206.15331). Four subsequent evaluations on
real 10-domain conversational data (N = 100 queries, 1,000 ingested turns, single seed)
show that the gains do not transfer: exact-match MRR lies below 0.005 for all retrieval
modes, and in the most permissive soft-domain interleaved condition the null predictor
outperforms the conditioned predictor (null MRR 0.177 vs predictive MRR 0.163). No result
in this repository clears the minimum conditions for a confident positive verdict: five
seeds with declared namespaces, a matched null passing the hypneum-nulls checklist, and
a raw-RAG baseline (NH-85, deferred). MK-11 is therefore classified as **refuted
(real data)** in catalogue v3 (hypneum-lab/.github).

---

## 1. Context

### 1.1 The Aeon predictor and why anti-collapse matters

Aeon is a lightweight working-memory module for a conversational assistant. Its core
component is `AeonPredictor` (`src/memory/aeon_predictor.py`), which wraps an `AeonSleep`
embedding palace (cosine-indexed HNSW store) with a small numpy MLP. The MLP is trained
online on successive latent states h\_t → h\_{t+1} drawn from the conversation trace. At
query time, instead of retrieving with the raw query vector h\_q, the predictor retrieves
with a *predicted successor* ĥ\_{t+1}. The rationale is that ĥ\_{t+1} should lie closer to
upcoming turns than h\_q itself.

The architecture (LatentMLP: 384 → 256 → 256 → 384 with a residual skip, one-hot stack
conditioning of size 16, numpy cosine loss, < 1 MB weights) is explicitly aligned with
LeCun's Autonomous Machine Intelligence framework (arXiv:2206.15331), specifically
Module 7 (Short-Term Memory). The April 2026 reframe plan (`docs/papers/paper-a-reframe-aeon-ami.md`)
also cited the LeJEPA / SIGReg line of work (arXiv:2511.08544) as framing for JEPA-style
latent prediction.

### 1.2 The April 2026 claims and the reframe

The reframe plan stated three "strong" empirical findings (§4):

1. Shared centering delivers +22 % MRR on structured streams (MK-11, condition D).
2. LayerNorm(δ) preserves stack-conditioning signal at 300 epochs: 59 % win\_stack
   (condition L2).
3. Text-JEPA compression retains 97 % VQC routing accuracy at 3× compression
   (MK-20). **This third claim is retracted; see E-2026-11-08-01.**

The present note focuses on claims 1 and 2 as they appear in the committed JSON artifacts.
Text-JEPA (MK-20) is not discussed further; the errata entry (E-2026-11-08-01) is the
authoritative record of its retraction.

---

## 2. Mechanisms under test

Three anti-collapse mechanisms were evaluated in April 2026, each implemented as a flag in
`PredictorConfig`:

**Shared centering** (`use_centering=True`, `per_stack_centering=False`): a running mean
c is maintained across all ingested latents and subtracted from each h\_t before it enters
the MLP. Inspired by DINO/DINOv3 self-distillation. Stateless (no EMA teacher). The intent
is to prevent representational collapse where ĥ converges to a constant.

**Per-stack centering** (`use_centering=True`, `per_stack_centering=True`): a separate
running mean is kept per stack\_id. This attempts to remove per-stack distributional shifts
while preserving cross-stack discriminability.

**LayerNorm(δ)** (`use_layernorm_delta=True`): instead of subtracting a running mean,
the per-sample residual δ = h\_{t+1} − h\_t is normalised via LayerNorm before entering
the MLP. The intent is to normalise transition magnitudes while preserving stack-specific
offset information that shared centering would remove.

A fourth control is the **std-ratio rollback**: if `std(ĥ) / std(h_t) < 0.1` the predictor
weights are reset to their last checkpoint (`detect_collapse`, `src/memory/aeon_predictor.py`).
This is a runtime safety mechanism, not an anti-collapse strategy in the training sense.

---

## 3. What the synthetic streams showed

All synthetic evaluations used `scripts/eval_aeon_predictor.py` with `--seed 0` (single
seed; default `default_rng(0)`). N = 100 queries from a stream of N = 1,000 turns (stack-
structured generator, 16 stacks, `--stream stack-structured`). No multi-seed sweep was run;
results are **exploratory**. The null predictor uses a random stack\_id at each query in
place of the true stack\_id.

**Table 1. Synthetic stream ablation (single seed = 0, n\_queries = 100 each).**

| Condition | Mechanism | Epochs | baseline\_mrr | predictive\_mrr | null\_mrr | win\_pred | win\_stack\_vs\_null | recall@5 (pred) | Source |
|---|---|---|---|---|---|---|---|---|---|
| C | No centering | 50 | 0.413 | 0.415 | 0.412 | 5 % | 5 % | 1.00 | `aeon-baseline-no-centering.json` |
| D | Shared centering | 50 | 0.413 | 0.498 | 0.498 | 51 % | 0 % | 1.00 | `aeon-baseline-shared-centering.json` |
| F1 | Per-stack centering | 50 | 0.413 | 0.413 | 0.495 | 9 % | 0 % | 1.00 | `aeon-poc-F1-per-stack.json` |
| F2 | Per-stack centering | 300 | 0.413 | 0.433 | 0.498 | 14 % | 0 % | 1.00 | `aeon-poc-F2-per-stack-long.json` |
| L1 | LayerNorm(δ) | 50 | 0.413 | 0.012 | 0.015 | 2 % | 3 % | 0.03 | `aeon-poc-L1-layernorm.json` |
| L2 | LayerNorm(δ) | 300 | 0.413 | 0.447 | 0.090 | 40 % | 59 % | 0.67 | `aeon-poc-L2-layernorm-long.json` |
| L3 | LN(δ) + centering | 50 | 0.413 | 0.005 | 0.012 | 1 % | 1 % | 0.01 | `aeon-poc-L3-ln-plus-cent.json` |

All files under `results/` prefix. All sourced from commit `804bf02` (conditions C, L1–L3)
or `2d2ca3e` (F1, F2), dated 2026-04-18. Metric definitions from `eval_aeon_predictor.py`:
`baseline_mrr` = MRR of raw retrieval; `predictive_mrr` = MRR of retrieval from ĥ;
`null_stack_mrr` = MRR of retrieval from ĥ using a *wrong* stack\_id; `win_rate_stack_vs_null`
= fraction of queries where predictive > null.

**Observations.** Condition D shows the strongest apparent gain: predictive\_mrr rises from
0.413 to 0.498 (+20.6 % relative). However, `null_mrr = 0.498` (identical to predictive),
and `win_rate_stack_vs_null = 0.00`: the predictor with the correct stack id never beats the
predictor with a *wrong* stack id. The centering gain is therefore not attributable to
stack conditioning; it is a regime effect where the centred embedding space saturates at the
top of the recall@5 distribution for both correct and incorrect stack ids. MK-11 describes
this as "+22 % MRR" but the null comparison is unfavourable to any stack-conditioning claim.

Condition L2 is the only one where `win_rate_stack_vs_null > 50 %`, suggesting LayerNorm(δ)
at 300 epochs genuinely uses stack information. However this is a single seed, the recall@5
drops to 0.67 (vs baseline 1.00), and there is no null distribution over permuted stacks.
Condition L3 (combining both mechanisms) causes catastrophic collapse: recall@5 = 0.01.

---

## 4. What the real-data streams showed

Evaluated with `scripts/eval_aeon_realdata.py` on a 10-domain real corpus (n\_domains = 10,
n\_samples\_ingested = 1,000, n\_queries = 100). Single seed throughout (`--seed 0`). The
"exact" matching mode requires the gold episode id to appear in the top-5 results; the
"soft" mode matches at the domain level (any embedding from the correct domain counts).

**Table 2. Real-data evaluations (single seed = 0, n\_queries = 100, dim = 384).**

| Scenario | Match | baseline\_mrr | predictive\_mrr | null\_mrr | win\_pred | win\_stack\_vs\_null | recall@5 (base / pred) | Source |
|---|---|---|---|---|---|---|---|---|
| Interleaved | Exact | 0.0045 | 0.0020 | 0.0020 | 1 % | 0 % | 0.02 / 0.01 | `aeon-realdata.json` |
| Within-topic | Exact | 0.0020 | 0.0020 | 0.0045 | 1 % | 0 % | 0.01 / 0.01 | `aeon-realdata-within-topic.json` |
| Interleaved | Soft | 0.0395 | 0.1628 | **0.1765** | 29 % | 12 % | 0.11 / 0.31 | `aeon-realdata-soft-interleaved.json` |
| Within-topic | Soft | **0.9900** | 0.1093 | 0.0687 | 1 % | 27 % | 0.99 / 0.47 | `aeon-realdata-soft-withintopic.json` |

All files at commit `a8cc1be` / `63e546c` / `88b5ed9` (2026-04-19). JSON keys used:
`baseline_mrr`, `predictive_mrr`, `null_stack_mrr`, `win_rate_predictive`,
`win_rate_stack_vs_null`, `baseline_recall_at_5`, `predictive_recall_at_5`. Bold values
flag the dominant metric per row.

**Domain-switch classification** (`results/aeon-next-domain.json`, commit `88b5ed9`):
`predictive_acc = 0.10` = chance level (1/10), `top3_predictive_acc = 0.33`. Switch
detection is perfect (`switch_detection_rate = 1.00`) but that reflects the construction of
the eval, not predictive power. `baseline_acc = 0.00` (raw retrieval cannot classify
next-domain by design). N = 100 queries = 100 switches, 10 domains, single seed.

**Observations.** Exact-match MRR is below 0.005 in all real-data conditions, confirming
that the latent palace is too sparse at N = 1,000 turns to support exact retrieval across
100 queries from a 10-domain vocabulary. Under soft-domain interleaved matching — the
most permissive condition — predictive MRR (0.163) rises above baseline (0.040), which
the April session index marked as a positive result. However, `null_mrr = 0.177 > 0.163`:
the null predictor, which uses an *arbitrary* stack id, outperforms the conditioned
predictor on this metric. `win_rate_stack_vs_null = 0.12` confirms the predictor with the
correct stack wins in only 12 % of queries. Under soft-domain within-topic, the raw
retrieval baseline reaches MRR = 0.99 (saturated), making any predictor performance
irrelevant by ceiling; predictive\_mrr = 0.109 is 9× below the baseline. These four
real-data runs jointly do not support MK-11.

---

## 5. Why the gap — hypotheses only

The following explanations are labelled as working hypotheses. None is tested or
pre-registered.

**H1 — Saturation regime.** The synthetic stream generator produces 16 deterministic
per-stack drift directions in a 384-dimensional space. With 1,000 turns and 50-epoch
training, the palace fills a smooth manifold where next-turn retrieval is near-perfect
(recall@5 = 1.00). Real conversational embeddings span a sparser, higher-variance manifold
where the same 1,000 turns leave most of the query space uncovered.

**H2 — Per-batch vs per-sample normalisation.** Shared centering subtracts a running
batch mean, which on the synthetic stream converges to the geometric centre of the drift
manifold. On real streams the mean drifts with topic, so centering de-correlates rather
than normalises. This hypothesis would predict that per-batch centering should hurt in
direct proportion to inter-topic variance, which the within-topic soft result (baseline
0.99) is consistent with but does not prove.

**H3 — Embedding geometry.** The 384-dimensional MiniLM-L6-v2 embedding space for
real conversational text is not isotropic: topic-specific attractors cluster tightly,
leaving the predictor's centring statistics dominated by the current topic. The soft
within-topic result (predictive\_mrr = 0.109 against baseline 0.99) is consistent with the
predictor moving the query vector *away* from its own topic cluster towards an adjacent one.

These hypotheses account for the direction of the gap but do not establish its cause.
Diagnostic runs isolating each factor would require pre-registered protocols.

---

## 6. What this note does not claim

This note does **not** claim that:

- The Aeon predictor architecture is fundamentally flawed. The code passes 33 unit tests
  (`uv run python -m pytest tests/memory/test_aeon_predictor.py`, 2026-11-08).
- LayerNorm(δ) is useless. Condition L2 shows a potentially informative pattern at 300
  epochs (win\_stack = 59 %) that warrants a seeded, controlled replication.
- The AMI Module 7 framing (arXiv:2206.15331) is incorrect. The framing is not being
  tested here.
- Real-data performance cannot improve with more data. The n = 1,000 turns is a
  pilot scale.
- The null predictor is a meaningful upper bound. The null predictor outperforming the
  conditioned predictor is a *lower bar* failure, not a proof of non-existence of signal.

---

## 7. What a valid test would need

A result capable of moving MK-11 from "refuted (real data)" to a positive verdict would
need at minimum:

1. **≥ 5 seeds with declared namespaces.** The hypneum-nulls policy (§5 hypneum-nulls
   SKILL) requires distinct pilot and confirmatory seed namespaces. With N = 100 queries
   and a sign test, power ≥ 80 % for effects ≥ 20 % requires at least 5 seeds; smaller
   effects are not decidable at this query count.

2. **A matched null.** The null predictor (wrong stack\_id) is the minimum control. It
   must be evaluated on the same queries, same stream, same seed. Shared-centering
   condition D in Table 1 already shows that predictive == null; any new claim must
   demonstrate predictive > null before interpreting a baseline–predictive difference.

3. **A raw-RAG baseline (NH-85).** Issue #13 (hypneum-lab/micro-kiki) defers the
   comparison between Aeon and a simple raw-RAG retrieval baseline to after M3. Without
   this, it is unknown whether the entire Aeon architecture (latent indexing + prediction)
   adds anything beyond a standard BM25 or dense retrieval system on the same corpus.

4. **Pre-registration (L0–L3 Hypneum roadmap).** Per the ROADMAP levels, a confirmatory
   claim at L2 or higher requires a time-stamped hypothesis and analysis plan filed before
   data collection. The April 2026 PoC was exploratory (pilot, not confirmatory). Any
   replication targeting a positive verdict on MK-11 must file an L2 pre-registration.

5. **Corpus scale.** The soft-within-topic result at N = 1,000 turns and baseline MRR = 0.99
   suggests retrieval saturation. A valid test should target a scale where baseline MRR
   is well below ceiling, e.g. N ≥ 10,000 turns or a genuinely interleaved corpus where
   the gold answer is not trivially retrievable by the raw embedding.

---

## References

- LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. OpenReview.
  arXiv:2206.15331. (AMI Module 7 framing.)
- Xiong et al. (2025). LeJEPA / SIGReg. arXiv:2511.08544. (JEPA vision context.)
- Hypothesis catalogue v3: hypneum-lab/.github,
  `docs/research/2026-11-06-refresh-hypotheses/catalogue-v3.md`.
  Row MK-11: "centering +22 % MRR — réfuté (données réelles)".
  Row MK-20: "Text-JEPA 97 % retention — retracted (E-2026-11-08-01)".
- NH-85 / issue #13: Aeon vs raw RAG baseline — deferred post-M3.
- E-2026-11-08-01 (this repository, `docs/ERRATA.md`): Text-JEPA VQC
  "97 % retention at 3× compression" retracted.

---

## Provenance

Commits of result files (via `git log --format='%h %ad' --diff-filter=A`):

| File | Commit | Date |
|---|---|---|
| `results/2026-04-19-aeon-baseline-no-centering.json` | `850048d` | 2026-04-18 |
| `results/2026-04-19-aeon-baseline-shared-centering.json` | `850048d` | 2026-04-18 |
| `results/2026-04-19-aeon-poc-F1-per-stack.json` | `2d2ca3e` | 2026-04-18 |
| `results/2026-04-19-aeon-poc-F2-per-stack-long.json` | `2d2ca3e` | 2026-04-18 |
| `results/2026-04-19-aeon-poc-L1-layernorm.json` | `804bf02` | 2026-04-18 |
| `results/2026-04-19-aeon-poc-L2-layernorm-long.json` | `804bf02` | 2026-04-18 |
| `results/2026-04-19-aeon-poc-L3-ln-plus-cent.json` | `804bf02` | 2026-04-18 |
| `results/aeon-realdata.json` | `63e546c` | 2026-04-19 |
| `results/aeon-realdata-within-topic.json` | `88b5ed9` | 2026-04-19 |
| `results/aeon-realdata-soft-interleaved.json` | `a8cc1be` | 2026-04-19 |
| `results/aeon-realdata-soft-withintopic.json` | `a8cc1be` | 2026-04-19 |
| `results/aeon-next-domain.json` | `88b5ed9` | 2026-04-19 |
