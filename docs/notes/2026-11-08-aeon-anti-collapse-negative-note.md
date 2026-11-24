# Output centering and LayerNorm(δ) in a ~267K-parameter latent predictor: a synthetic MRR gain matched by the null predictor, and real-data pilots that did not test centering

**Date:** 2026-11-08
**Version:** 2 (2026-11-24)
**Supersedes** the text deposited as 10.5281/zenodo.24496729; see erratum E-2026-11-24-01 (`docs/ERRATA.md`).
**Status:** Negative note — single seed throughout; all findings are exploratory
**Hypothesis catalogue:** hypneum-lab/.github `docs/research/2026-11-06-refresh-hypotheses/catalogue-v3.md` (row MK-11)
**Related issues:** hypneum-lab/micro-kiki #13 (NH-85, Aeon vs raw RAG baseline, deferred)
**Errata:** E-2026-11-24-01 (this note, version 1), E-2026-11-08-01 (MK-20, Text-JEPA retracted)

> Résumé (FR) :
> En avril 2026, le prédicteur latent Aeon (MLP numpy, ~267 K paramètres,
> `src/memory/aeon_predictor.py`) affichait +20,6 % de MRR sur un flux synthétique avec
> centrage partagé de la sortie (condition D, MRR 0,413 → 0,498 ; ligne MK-11 du catalogue).
> Dans ce même essai, le prédicteur nul obtient exactement le même MRR (0,498) et le
> prédicteur conditionné ne le bat sur aucune requête : le gain n'est pas attribuable au
> conditionnement par pile. Les quatre évaluations sur textes réels (10 domaines,
> 1 000 messages) ont toutes été lancées avec LayerNorm(δ) et sans centrage : le centrage
> n'a jamais été testé sur données réelles. Ces pilotes ne montrent aucun avantage du
> prédicteur conditionné sur le prédicteur nul (MRR 0,163 contre 0,177 en appariement par
> domaine, flux entrelacé). Leur protocole limite ce qu'ils peuvent établir : les
> « tours » sont les premiers messages d'enregistrements indépendants, les requêtes font
> partie des données d'entraînement, et le prédicteur nul reçoit une entrée jamais vue à
> l'entraînement. Dans le code, ni le centrage ni LayerNorm(δ) n'entrent dans la perte
> d'entraînement : ce sont des transformations de la sortie appliquées à l'inférence.
> Tous les essais sont à graine unique, sans préenregistrement ; rien ne dépasse le stade
> exploratoire. La version 1 de cette note concluait à tort que le gain du centrage ne
> se transférait pas aux données réelles ; voir l'erratum E-2026-11-24-01.

---

## Changes from version 1

Version 1 (2026-11-08, deposited unchanged on Zenodo as 10.5281/zenodo.24496729) contained
the errors below. They were found by an independent review on 2026-11-24 and checked
against the code and the result files before this revision. Items 10 and 11 were found
during the revision itself.

1. **Centering was never run on real data.** All four real-data result files record
   `"use_centering": false, "use_layernorm_delta": true`. The title, the abstract and
   the hypotheses H2 and H3 of version 1, which said that the centering gain "does not
   survive" real streams and explained the gap by centering, were not supported. They
   are withdrawn.
2. **Wrong identifier for LeCun (2022).** arXiv:2206.15331 is "GitHub Copilot AI pair
   programmer: Asset or Liability?" (Moradi Dakhel et al.). "A Path Towards Autonomous
   Machine Intelligence" is an OpenReview position paper with no arXiv version. Corrected
   in four places.
3. **Wrong authors for LeJEPA.** arXiv:2511.08544 is by Balestriero and LeCun, not
   "Xiong et al.".
4. **Mechanisms misdescribed.** Centering is subtracted from the predictor's output, not
   from its input. LayerNorm(δ) normalises the residual predicted by the MLP, not the
   observed difference h\_{t+1} − h\_t. Neither enters the training loss, so version 1
   was wrong to present them as choices made "in training" (§2).
5. **Null predictor misdescribed.** The null predictor uses `stack_id = -1`, an all-zero
   stack input that never occurs in training. Version 1 called it "random", "wrong" and
   "arbitrary" in three places.
6. **Real-data protocol misdescribed.** The "conversational streams" are first user
   messages of separate records, not consecutive turns of conversations. Version 1 read
   exact-match MRR below 0.005 as a sign that "the latent palace is too sparse"; it is
   the level expected by chance. The soft within-topic baseline of 0.99 follows from the
   construction. The query turns are part of the training data.
7. **Learning rate of the 300-epoch runs.** F2 and L2 used a learning rate of 5e-3, not
   the 1e-3 of the 50-epoch runs. Version 1 presented them as differing by epochs only.
8. **Unsupported power statement removed** (§7 of version 1: "power ≥ 80 % … requires at
   least 5 seeds"). No calculation supported it.
9. **Smaller corrections.** The weights are float32 (about 1 MB), not float64. The
   vector index is the page-clustered Atlas index, not HNSW. The std-ratio rollback is
   not exercised by the evaluation scripts. The +20.6 % is relative to raw retrieval in
   the same run. Internal references a reader could not resolve are replaced by file
   paths or removed. The bullet "The null predictor is a meaningful upper bound" is
   removed.
10. **Stack id alignment in the synthetic stream** (found during revision). The stack id
    given with a query identifies the transition that produced the query turn, not the
    transition to be predicted (§3.2). Version 1's reading of condition L2 as showing
    that the predictor "genuinely uses stack information" is withdrawn.
11. **Ceiling of the synthetic metric** (found during revision). The query turn is
    itself stored in the index, so the gold successor can rank second at best and MRR
    cannot exceed 0.5 for the raw baseline (§3.2).

The numbers in Tables 1 and 2 are unchanged; they were re-read from the JSON files.

---

## Abstract

A ~267K-parameter numpy MLP (`LatentMLP`, `src/memory/aeon_predictor.py`) predicts the
next latent state of a conversation and is used to shift retrieval queries in the Aeon
memory module. In April 2026 it was evaluated on a synthetic stream under two output
transforms, a running-mean centering and a LayerNorm applied to the predicted residual.
With shared centering (condition D), predictive MRR was 0.498 against 0.413 for raw
retrieval (+20.6 %), a figure later cited as hypothesis MK-11. In the same run the null
predictor also reaches 0.498 and the conditioned predictor beats it on none of the 100
queries, so this gain is not attributable to stack conditioning. Four pilot evaluations
on real text (10 domains, 1,000 ingested messages, 100 queries) all used LayerNorm(δ)
without centering: centering was never run on real data, and nothing here says whether
its synthetic gain transfers. In these pilots the conditioned predictor shows no benefit
over the null predictor (soft-domain interleaved: MRR 0.163 vs 0.177). Their protocol
limits what they can show: the "turns" are first user messages of independent records,
the queries are part of the training data, and the null predictor receives a stack input
never seen in training. In the code, neither transform enters the training loss; both
act only on the output at inference. All runs are single-seed and exploratory. The
mechanisms have not been validly tested, in either direction.

---

## 1. Context

### 1.1 The Aeon predictor

Aeon is a working-memory module for a conversational assistant. `AeonPredictor`
(`src/memory/aeon_predictor.py`) wraps an `AeonSleep` store, whose vector index is the
page-clustered Atlas index with cosine similarity (`src/memory/atlas.py`), with a small
numpy MLP. The MLP is trained on successive latent states h\_t → h\_{t+1}. At query
time, retrieval uses a predicted successor ĥ\_{t+1} instead of the raw query vector
h\_q. The intent is that ĥ\_{t+1} lies closer to upcoming turns than h\_q does.

The network (`LatentMLP`) maps 384 + 16 inputs through two hidden layers of 256 units to
a 384-dimensional residual that is added to the input: ĥ = x + δ. The 16 extra inputs
are a scaled one-hot code of a "stack" identifier (in micro-kiki, the active LoRA
domain). It has 267,136 parameters stored in float32 (about 1 MB) and is trained with a
cosine loss.

The April 2026 reframing plan (`docs/papers/paper-a-reframe-aeon-ami.md`) presented this
predictor as an instance of the short-term memory module (Module 7) of LeCun's
"A Path Towards Autonomous Machine Intelligence" (OpenReview, 2022) and cited LeJEPA
(Balestriero and LeCun, arXiv:2511.08544) as context for latent prediction. That framing
is not tested here.

### 1.2 The April 2026 claims

The reframing plan listed three "strong" empirical findings (its §4):

1. Shared centering gives +20.6 % MRR on structured streams (MK-11, condition D;
   originally reported as "+22 %").
2. LayerNorm(δ) preserves the stack-conditioning signal at 300 epochs: 59 % of queries
   won against the null predictor (condition L2).
3. Text-JEPA compression retains 97 % of VQC routing accuracy at 3× compression (MK-20).
   **This third claim is retracted; see E-2026-11-08-01.**

This note covers claims 1 and 2 as they appear in the committed JSON files.

---

## 2. What the code does

Line numbers refer to `src/memory/aeon_predictor.py` at commit `ba5b1f5`.

**Shared centering** (`use_centering=True`). After the forward pass computes
`out = x + delta` (line 141), the mean of the outputs in the batch updates an
exponential running mean (momentum 0.9), and that running mean is subtracted from the
output (lines 157–162). The input to the MLP is not centered.

**Per-stack centering** (`per_stack_centering=True`). The same operation with one
running mean per stack id, plus a separate slot for `stack_id = -1` (lines 144–154).

**LayerNorm(δ)** (`use_layernorm_delta=True`). The residual predicted by the MLP is
standardised across its 384 components for each sample, then added to the input (lines
136–141). The gain and bias (`_ln_gamma`, `_ln_beta`) are fixed at 1 and 0 and are never
updated. A standardised 384-dimensional residual has norm √384 ≈ 19.6, while the inputs
are unit vectors, so with this flag the direction of the output is set almost entirely
by the residual.

**None of these enters training.** `backward_cosine` recomputes ĥ = x + δ from the
cached activations without LayerNorm and without centering (lines 184–186), and the
loss and gradients are taken on that quantity. For a given seed, number of epochs and
learning rate, the trained weights are therefore the same whichever flag is set. The
result files agree: `final_train_loss` is 0.5667058005928993 in conditions C, D, F1, L1
and L3, and 0.5203294903039932 in F2 and L2. The conditions of Table 1 differ only in
the transform applied to the output when the predictor is queried. Describing them as
anti-collapse mechanisms of the training procedure, as version 1 did, is inaccurate.

**The running mean is mutable at query time.** `predict_next` calls the same `forward`
(line 389), which updates the running mean on every call (lines 150–153 and 158–161),
with a batch of one. The evaluation scripts call the predictor twice per query, first
with the true stack id and then for the null predictor, so with centering the two calls
share and modify the same state.

**Collapse rollback.** `detect_collapse` flags a predictor whose output standard
deviation falls below 0.1 times that of the targets. It is called from
`AeonSleep.sleep_cycle` (`src/memory/aeonsleep.py:327-346`), which restores the previous
weights when the flag is raised. The evaluation scripts call `fit_on_buffer` directly,
so this rollback plays no part in the results below.

---

## 3. Synthetic stream

### 3.1 Protocol and results

`scripts/eval_aeon_predictor.py --stream stack-structured`: 1,000 unit vectors in 384
dimensions. Each transition draws one of 16 stacks at random and adds that stack's fixed
drift direction plus noise. All 1,000 turns are ingested and the predictor is trained on
all consecutive pairs. The 100 queries are the turns 899 to 998; the gold answer for
turn i is turn i+1. Each query is scored three ways with top-5 retrieval:

- `baseline`: retrieval with the raw query vector;
- `predictive`: retrieval with ĥ computed with the stack id stored with the query turn;
- `null`: retrieval with ĥ computed with `stack_id = -1`, which gives an all-zero stack
  input (script lines 201–202; predictor lines 408–409).

MRR is the mean reciprocal rank of the gold turn within the top five (0 if absent).
`win_pred` is the fraction of queries where predictive beats baseline; `win_vs_null` the
fraction where predictive has a strictly higher reciprocal rank than null.

**Table 1. Synthetic stream, single seed (0), 100 queries per condition.**

| Condition | Output transform | Epochs, lr | baseline MRR | predictive MRR | null MRR | win\_pred | win\_vs\_null | recall@5 pred / null |
|---|---|---|---|---|---|---|---|---|
| C | none | 50, 1e-3 | 0.413 | 0.415 | 0.412 | 5 % | 5 % | 1.00 / 1.00 |
| D | shared centering | 50, 1e-3 | 0.413 | 0.498 | 0.498 | 51 % | 0 % | 1.00 / 1.00 |
| F1 | per-stack centering | 50, 1e-3 | 0.413 | 0.413 | 0.495 | 9 % | 0 % | 1.00 / 1.00 |
| F2 | per-stack centering | 300, 5e-3 | 0.413 | 0.433 | 0.498 | 14 % | 0 % | 1.00 / 1.00 |
| L1 | LayerNorm(δ) | 50, 1e-3 | 0.413 | 0.012 | 0.015 | 2 % | 3 % | 0.03 / 0.02 |
| L2 | LayerNorm(δ) | 300, 5e-3 | 0.413 | 0.447 | 0.090 | 40 % | 59 % | 0.67 / 0.17 |
| L3 | LayerNorm(δ) + shared centering | 50, 1e-3 | 0.413 | 0.005 | 0.012 | 1 % | 1 % | 0.01 / 0.02 |

Source files, all under `results/`: `2026-04-19-aeon-baseline-no-centering.json` (C),
`2026-04-19-aeon-baseline-shared-centering.json` (D), `2026-04-19-aeon-poc-F1-per-stack.json`,
`2026-04-19-aeon-poc-F2-per-stack-long.json`, `2026-04-19-aeon-poc-L1-layernorm.json`,
`2026-04-19-aeon-poc-L2-layernorm-long.json`, `2026-04-19-aeon-poc-L3-ln-plus-cent.json`.
JSON keys: `baseline_mrr`, `predictive_mrr`, `null_stack_mrr`, `win_rate_predictive`,
`win_rate_stack_vs_null`, `predictive_recall_at_5`, `null_stack_recall_at_5`. Baseline
recall@5 is 1.00 in every condition.

The JSON files do not record the seed, the number of epochs or the learning rate, and
the files for C, D, F1, F2 and L1 do not contain the `use_layernorm_delta` key. The settings in
the table were recovered by re-running the script on 2026-11-24: with seed 0 they
reproduce every committed value of the seven files, including `final_train_loss`, to the
last digit. The re-run outputs are not committed and add no new number.

### 3.2 What the protocol can and cannot show

These points follow from the script and bear on every row of Table 1.

- **The query is in the index.** Query turn i was itself ingested. Raw retrieval returns
  it first, so the gold turn i+1 ranks second at best and baseline MRR cannot exceed
  0.5. A baseline of 0.413 and a predictive MRR of 0.498 are both readings on this
  compressed scale: the "gain" of condition D is the gold turn moving from third to
  second place on about half of the queries.
- **No held-out data.** The query turns and their successors are part of the training
  pairs. The script comment "held-out indices" (line 173) does not match the code.
- **The stack id describes the previous transition.** The stream is ingested with
  `stream_stack_ids = [0] + transition_stacks` (lines 150–156): the id stored with turn
  i is the stack that produced turn i from turn i−1. The transition to be predicted,
  from i to i+1, draws its stack independently. Training pairs and queries both use the
  id of turn i. The stack input therefore carries no information about the drift of the
  transition being predicted. It does carry information about where turn i−1 lies, and
  turn i−1 is the competitor of the gold turn for second place in the ranking.
- **The null input is outside the training distribution.** Every training pair has a
  stack id between 0 and 15. The all-zero stack input of the null predictor is never
  seen in training. A difference between predictive and null therefore mixes "the stack
  id is informative" with "the network is being evaluated on an input pattern it was
  not trained on". A null using a permuted valid stack id was not run.
- **With per-stack centering the null uses different statistics.** `stack_id = -1`
  selects a running-mean slot that training never updates; it starts at zero and is
  updated only by the null queries themselves. In F1 and F2 the predictive and null
  outputs differ in their centering state as well as in their stack input.

### 3.3 Observations

**Condition D.** Predictive MRR is 0.498 against a raw-retrieval baseline of 0.413
(+20.6 %; +20.1 % against the predictive MRR of condition C, 0.415). The null predictor
obtains the same 0.498 and `win_rate_stack_vs_null` is 0: on no query does the true
stack id do better than the all-zero input. The gain is produced by applying shared
centering to the output and does not depend on the stack input. It is not evidence for
stack conditioning. Since the trained weights are those of condition C, it is not
evidence about training either. Why subtracting the running mean improves the rank of
the gold turn in this stream was not investigated.

**Conditions F1 and F2.** The null predictor scores higher than the conditioned one
(0.495 vs 0.413; 0.498 vs 0.433) and `win_rate_stack_vs_null` is 0 in both. Because the
null uses a running-mean slot filled only by the null queries themselves (§3.2), this
comparison does not isolate the stack input, and these runs do not show that per-stack
centering is harmful in general. They show that, as implemented and evaluated, it gave
no gain over the null.

**Conditions L1, L2, L3.** With LayerNorm(δ) at 50 epochs (L1) and with both transforms
(L3), retrieval fails almost entirely (recall@5 of 0.03 and 0.01, against 1.00 for the
baseline). At 300 epochs and a learning rate of 5e-3 (L2), predictive MRR is 0.447,
recall@5 is 0.67 (below the baseline's 1.00), and the conditioned predictor beats the
null on 59 % of queries. L2 and L1 differ in both epochs and learning rate. Given the
points of §3.2, the 59 % cannot be read as the predictor using stack information about
the next turn; it may reflect the all-zero null input being out of distribution for an
output that depends almost entirely on the residual. This was not tested.

---

## 4. Real-data pilots

### 4.1 Protocol and results

`scripts/eval_aeon_realdata.py` reads `data/final/<domain>/train.jsonl` for 10 technical
domains. `data/` is not tracked by git, so the description below comes from the script.
For each domain it takes the first user message of each of the first 100 records (lines
101–119) and embeds it with a 384-dimensional sentence-embedding model loaded from
`models/niche-embeddings`. It then builds one stream of 1,000 vectors in one of two
ways:

- `interleaved`: the domains are visited round-robin, in a reshuffled order at each
  round (lines 139–150);
- `within-topic`: the 100 messages of each domain are placed contiguously (lines
  127–136).

The stack id is the domain index. All 1,000 vectors are ingested and the predictor is
trained on all consecutive pairs (lines 226–235); the 100 queries are the turns 899 to
998 (lines 240–248). The script docstring says "train on first 80%, evaluate on last
20%"; the code does not hold anything out. "Exact" scoring asks for the id of turn i+1
in the top five; "soft" scoring accepts any stored turn of the same domain as turn i+1.

**All four runs used LayerNorm(δ) without centering** (`"use_centering": false,
"use_layernorm_delta": true` in each file). No real-data run with centering exists in
this repository.

**Table 2. Real-data pilots, LayerNorm(δ), no centering, 100 queries each.**

| Stream | Scoring | baseline MRR | predictive MRR | null MRR | win\_pred | win\_vs\_null | recall@5 base / pred / null | Source |
|---|---|---|---|---|---|---|---|---|
| Interleaved | exact | 0.0045 | 0.0020 | 0.0020 | 1 % | 0 % | 0.02 / 0.01 / 0.01 | `aeon-realdata.json` |
| Within-topic | exact | 0.0020 | 0.0020 | 0.0045 | 1 % | 0 % | 0.01 / 0.01 / 0.02 | `aeon-realdata-within-topic.json` |
| Interleaved | soft | 0.0395 | 0.1628 | 0.1765 | 29 % | 12 % | 0.11 / 0.31 / 0.29 | `aeon-realdata-soft-interleaved.json` |
| Within-topic | soft | 0.9900 | 0.1093 | 0.0687 | 1 % | 27 % | 0.99 / 0.47 / 0.28 | `aeon-realdata-soft-withintopic.json` |

All files under `results/`. The files do not record the seed, the epochs or the learning
rate; the script defaults are seed 0, 50 epochs and 1e-3. These runs could not be
reproduced for this revision because the data are not in the repository. The two
interleaved rows share the same `final_train_loss` (0.2350528948009014), as do the two
within-topic rows (0.17055095545947552): each pair is one training scored two ways.

**Next-domain classification** (`results/aeon-next-domain.json`,
`scripts/eval_aeon_next_domain.py`, interleaved stream, LayerNorm(δ), no centering
option in the script). The predicted vector is assigned to the nearest domain centroid.
Accuracy is 0.10 for 10 domains and top-3 accuracy 0.33; with ten equally frequent
domains, guessing gives 0.10 and 0.30. All 100 evaluated transitions are domain
switches, so the "stay on topic" baseline scores 0.00 and the switch-detection rate of
1.00 only says that the predicted domain always differed from the current one. Unlike
the two scripts above, this one trains only on the turns before the query window.

### 4.2 What the protocol can and cannot show

- **The stream is not a conversation.** Consecutive vectors are first messages of
  different records. Turn i+1 is a draw from another domain (interleaved) or from the
  same domain (within-topic), not a reply to or continuation of turn i. Nothing ties a
  specific turn i+1 to turn i beyond its domain.
- **Exact scoring is at chance level by construction.** With 1,000 stored turns and
  five retrieved, a ranking unrelated to the gold turn gives recall@5 of about 0.005
  and MRR of about 0.002 (analytical estimate, not a measurement). The observed values
  are one or two hits in 100 queries. Version 1 attributed them to a memory "too
  sparse at N = 1,000"; they say nothing about density.
- **The soft within-topic baseline is trivial.** The query is in the index and belongs
  to the same domain as its successor except at the single block boundary inside the
  query window, so raw retrieval scores 99 out of 100 without predicting anything. Any
  transform that moves the query away from itself can only lower this score.
- **The soft interleaved scores are below random retrieval.** In this stream the next
  turn almost always belongs to a different domain from the query. Five stored turns
  drawn at random would contain the gold domain with probability 1 − 0.9⁵ ≈ 0.41
  (analytical estimate, not a measurement). The baseline's recall@5 of 0.11 is well
  below that, as expected if the neighbours of a query come mostly from its own domain
  (not checked: the data are not in the repository). The predictive and null recall@5
  of 0.31 and 0.29 are also below it, so the rise from 0.11 is not evidence of
  prediction.
- **Same limits as the synthetic protocol:** queries are in the training pairs, the null
  input is never seen in training, and there is one seed.

### 4.3 Observations

In none of the four rows does the conditioned predictor show a benefit over the null
predictor that survives these limits. In the soft interleaved row, which
`docs/SESSION-2026-04-19-INDEX.md` recorded in April as a positive result, the null MRR
is higher than the predictive MRR (0.1765 vs 0.1628) and the conditioned predictor wins
on 12 % of queries. In the soft within-topic row the predictive MRR is higher than the
null (0.1093 vs 0.0687) but both are far below a baseline that is trivially 0.99. The
exact rows are at chance. The next-domain accuracy is at chance.

These pilots concern the LayerNorm(δ) variant only. They do not test centering, and
because of the protocol they are weak evidence even about LayerNorm(δ): they show no
benefit, and they could not have shown one for next-turn prediction since the stream
has no turn-to-turn structure beyond the domain.

---

## 5. Why the synthetic and real-data results cannot be compared

Version 1 set the +20.6 % of condition D against the real-data pilots and offered three
hypotheses for the "gap". Two of them (H2, H3) explained it by the behaviour of
centering on real embeddings. They are withdrawn: no real-data run used centering. The
third (H1, saturation of the synthetic stream) is replaced by the description of the
protocol in §3.2.

The two sets of runs differ in the output transform (centering in D, LayerNorm(δ) in
all real-data runs), in the stream (a drift process with turn-to-turn structure, against
independent draws within or across domains), and in what the baseline measures. A
difference between them cannot be assigned to any one of these. No explanation of the
difference is proposed here.

---

## 6. What this note does not claim

- It does not claim that centering or LayerNorm(δ) is useless or harmful. Neither was
  validly tested: they are not part of training, the synthetic null is confounded, and
  centering was never run on real data.
- It does not claim that the synthetic gain of condition D fails to transfer to real
  data. That transfer was not tested.
- It does not claim that the predictor code is faulty as software. Its 33 unit tests
  pass (`tests/memory/test_aeon_predictor.py`, re-run 2026-11-24). They test shapes,
  training and state handling, not the scientific claims.
- It does not claim that condition L2 shows a use of stack information, nor that it
  shows the absence of one.
- It does not assess the framing of the predictor as LeCun's Module 7.
- It does not claim that results would be the same at larger scale or on real
  multi-turn conversations.

What the files support is narrower: (a) in condition D the predictive and null MRR are
equal, so the +20.6 % is not attributable to stack conditioning; (b) in the real-data
pilots of the LayerNorm(δ) variant, the conditioned predictor shows no benefit over the
null; (c) every run is single-seed and exploratory.

---

## 7. Status of MK-11

Catalogue v3 records MK-11 ("centering +22 % MRR on structured streams") as "réfuté
(données réelles)" and cites `results/aeon-realdata.json`. That file is a LayerNorm(δ)
run without centering. On the evidence in this repository, the claim that centering
improves MRR through stack conditioning is unsupported on synthetic data (null equal to
predictive), and untested on real data. Whether the catalogue label should change is a
decision for the principal investigator; this note does not change it.

---

## 8. What a valid test would need

1. **Centering actually run on real data**, with the flags recorded in the result file
   together with seed, epochs and learning rate.
2. **Real consecutive turns.** A stream built from multi-turn conversations, so that
   turn i+1 depends on turn i.
3. **Held-out queries.** Queries and their gold successors excluded from training and,
   for the baseline to be meaningful, the query itself excluded from retrieval.
4. **A matched null.** The null predictor should receive a valid stack id drawn at
   random or permuted, not a pattern absent from training, and both calls should run
   with the running means frozen.
5. **A stack id that refers to the transition being predicted**, or an explicit
   statement of what the stack id is expected to inform.
6. **Mechanisms inside the training loop** if they are to be called anti-collapse
   mechanisms: a loss computed on the transformed output, and an actual measure of
   collapse.
7. **Several seeds.** The lab's working rule is at least five seeds with separate pilot
   and confirmatory seed sets; this is a rule of thumb, not the result of a power
   calculation.
8. **Pre-registration** of the hypothesis and analysis plan before the runs. The April
   2026 runs were pilots.
9. **A raw-retrieval (RAG) baseline** on the same corpus (issue #13, NH-85, deferred).

---

## References

- LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. Version 0.9.2,
  27 June 2022. OpenReview position paper (no arXiv version).
  https://openreview.net/forum?id=BZ5a1r-kVsf
- Balestriero, R., LeCun, Y. (2025). LeJEPA: Provable and Scalable Self-Supervised
  Learning Without the Heuristics. arXiv:2511.08544.
- Hypothesis catalogue v3: hypneum-lab/.github,
  `docs/research/2026-11-06-refresh-hypotheses/catalogue-v3.md`.
  Row MK-11: "centering +22 % MRR on structured streams — réfuté (données réelles)".
  Row MK-20: retracted (E-2026-11-08-01).
- NH-85 / issue #13 (hypneum-lab/micro-kiki): Aeon vs raw RAG baseline, deferred.
- `docs/ERRATA.md` (this repository): E-2026-11-24-01 (version 1 of this note);
  E-2026-11-08-01 (Text-JEPA "97 % retention at 3× compression" retracted).
- Version 1 of this note: commit `3bf44c1`; Zenodo 10.5281/zenodo.24496729 (concept DOI
  10.5281/zenodo.24496728).

---

## Provenance

Commits that added the result files (`git log --diff-filter=A --format='%h %ad'`):

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

Code read for this revision, at commit `ba5b1f5`: `src/memory/aeon_predictor.py`,
`src/memory/aeonsleep.py`, `scripts/eval_aeon_predictor.py`,
`scripts/eval_aeon_realdata.py`, `scripts/eval_aeon_next_domain.py`.
