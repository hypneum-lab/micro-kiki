# Test-time output centering in a small latent next-state predictor: a synthetic retrieval gain that needs no predictor, and real-data pilots that did not test it

**Date:** 2026-11-08
**Version:** 2 (2026-11-24)
**Supersedes** the text deposited as 10.5281/zenodo.24496729; see erratum E-2026-11-24-01 (`docs/ERRATA.md`).
**Status:** Negative note — single seed throughout; all findings are exploratory
**File path:** kept from version 1, although `docs/CLAUDE.md` prefers a new dated note
for a revision: the erratum, `docs/SESSION-2026-04-19-INDEX.md` and the paper drafts
cite this path, and the history then shows the change. The slug ("anti-collapse") no
longer matches the content (§2).
**Hypothesis catalogue:** hypneum-lab/.github `docs/research/2026-11-06-refresh-hypotheses/catalogue-v3.md` (row MK-11)
**Related issues:** hypneum-lab/micro-kiki #13 (NH-85, Aeon vs raw RAG baseline, deferred)
**Errata:** E-2026-11-24-01 (this note, version 1), E-2026-11-08-01 (MK-20, Text-JEPA retracted)

> Résumé (FR) :
> Nous réexaminons des résultats exploratoires d'avril 2026 sur un petit prédicteur
> d'état suivant utilisé pour la recherche en mémoire d'un assistant conversationnel
> (micro-kiki). Un MLP numpy de 267 136 paramètres associe à un vecteur latent h_t de
> dimension 384 (un plongement de phrase dans les essais sur texte réel) et à un
> identifiant de « pile » (domaine actif) codé en one-hot un successeur prédit ĥ_{t+1},
> qui remplace h_t comme requête. Deux transformations étaient appliquées à la sortie
> du prédicteur : la soustraction d'une moyenne glissante (« centrage ») et la
> standardisation du résidu prédit (« LayerNorm(δ) »). Aucune n'entre dans la perte
> d'entraînement. Sur un flux synthétique à dérive, le centrage fait passer le MRR@5
> du véritable élément suivant de 0,413 (recherche brute) à 0,498 (+20,6 %) ; la
> requête étant elle-même indexée et retrouvée en premier, le plafond est 0,5, et le
> gain fait passer l'élément suivant de la troisième à la deuxième place (deuxième sur
> 48 requêtes sur 100 en recherche brute, sur 99 avec centrage). Un contrôle nul, qui
> retire l'identifiant de pile, obtient le même 0,498 et n'est battu sur aucune des
> 100 requêtes : le gain n'est pas attribuable au conditionnement par l'identifiant de
> pile. Un diagnostic ajouté dans cette version montre que le gain tient à la mise à
> jour de la moyenne glissante par les requêtes elles-mêmes, prises dans l'ordre du
> flux : avec les requêtes mélangées (cinq ordres) le MRR vaut 0,410 à 0,430, avec la
> moyenne figée après l'entraînement 0,423, et la soustraction d'une moyenne mobile
> des requêtes récentes à la requête brute, sans aucun prédicteur, redonne 0,498.
> Avec LayerNorm(δ), la recherche synthétique échoue après 50 époques (recall@5 de
> 0,03 contre 1,00 en recherche brute) ; après 300 époques à un taux d'apprentissage
> plus élevé, le prédicteur conditionné bat le nul sur 59 % des requêtes, mais ce nul
> est une entrée jamais vue à l'entraînement. Quatre pilotes sur texte réel (premiers
> messages utilisateur de 10 domaines techniques, 1 000 éléments, 100 requêtes) n'ont
> utilisé que la normalisation du résidu : le centrage n'a jamais été lancé sur
> données réelles, et l'on ignore si son gain synthétique s'y transfère. Dans ces
> pilotes, le prédicteur conditionné ne montre, face au nul, aucun avantage qui résiste
> aux limites du protocole : en évaluation souple au niveau du domaine sur le flux
> entrelacé, le nul fait mieux (MRR 0,177 contre 0,163) ; dans le seul essai où le
> prédicteur conditionné fait mieux (évaluation souple, flux par domaine, 0,109 contre
> 0,069), les deux sont très loin d'une recherche brute à 0,99 que la construction rend
> triviale. Les éléments sont des premiers messages d'enregistrements distincts, en
> ordre mélangé, et non des tours consécutifs de conversation ; les requêtes font
> partie des données d'entraînement ; l'entrée du nul n'apparaît jamais à
> l'entraînement. Tous les essais sont à graine unique et exploratoires ; aucune des
> deux transformations n'a été testée dans des conditions qui permettraient de montrer
> un apport du prédicteur. Cette version corrige la version 1, qui concluait à tort
> que le gain du centrage ne se transférait pas aux données réelles (erratum
> E-2026-11-24-01).

---

## Changes from version 1

Version 1 (2026-11-08, commit `3bf44c1`, deposited unchanged on Zenodo as
10.5281/zenodo.24496729) contained the errors below. The numbering is the same as in
erratum E-2026-11-24-01. Items 1 to 6 were reported by an independent review on
2026-11-24. Items 7 to 9 were found while preparing this version. Items 10 and 11, and
the additions marked "(later review)" in items 4, 6 and 12, come from two further
reviews, of drafts of this version. Every item was checked against the code, the
result files or the cited source before being written here.

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
   from its input. LayerNorm(δ) normalizes the residual predicted by the MLP, not the
   observed difference h\_{t+1} − h\_t. Neither enters the training loss, so version 1
   was wrong to present them as anti-collapse choices of training (§2). (Later review:)
   version 1 called centering "stateless (no EMA teacher)" and "inspired by DINO/DINOv3
   self-distillation". The running mean is itself an exponential-moving-average state.
   In DINO (arXiv:2104.14294, Section 3.1 of Caron et al.) centering is applied,
   together with sharpening, to the teacher's output inside the training loss, and
   never to the features that are evaluated; here it is applied only at evaluation.
   The analogy is dropped. Version 1 also said the MLP is "trained online"; in these
   runs it is trained in batch, for 50 or 300 epochs, after the whole stream has been
   ingested.
5. **Null predictor misdescribed.** The null predictor uses `stack_id = -1`, an all-zero
   stack input that never occurs in training. Version 1 called it "random", "wrong",
   "incorrect" or "arbitrary" in six places.
6. **Real-data protocol misdescribed.** The "conversational streams" are first user
   messages of separate records in shuffled order, not consecutive turns of
   conversations. Version 1 read exact-match MRR below 0.005 as a sign that "the latent
   palace is too sparse"; it is the level expected by chance. The soft within-topic
   baseline of 0.99 follows from the construction. The query turns are part of the
   training data. (Later review:) the recommendation of version 1 to test at
   "N ≥ 10,000 turns" rested on that misreading and is withdrawn. Version 1 also said
   that in the next-domain run "raw retrieval cannot classify next-domain by design";
   the baseline there is "stay on the current domain", and it scores 0.00 because all
   100 evaluated transitions change domain.
7. **Learning rate of the 300-epoch runs omitted.** F2 and L2 used a learning rate of
   5e-3, not the 1e-3 of the 50-epoch runs. The April drafts say so
   (`docs/papers/paper-a-draft-v1.md`, Table 2); version 1 presented these runs as
   differing by epochs only.
8. **Stack id alignment in the synthetic stream.** The stack id given with a query
   identifies the transition that produced the query turn, not the transition to be
   predicted (§3.2). Version 1's reading of condition L2 as showing that the predictor
   "genuinely uses stack information" is withdrawn.
9. **Ceiling of the synthetic metric.** The query turn is itself stored in the index
   and raw retrieval returns it first, so the gold successor ranks second at best and
   the baseline MRR cannot exceed 0.5 (§3.2).
10. **The centering gain comes from state updated during evaluation.** A diagnostic
    added in this version (§3.4) shows that the 0.498 of condition D depends on the
    running mean being updated by the queries in stream order, and is reproduced
    without any predictor. Version 1 explained the gain by "a regime effect where the
    centred embedding space saturates"; that explanation had no support and is
    withdrawn.
11. **"Module 7" is not a numbering of LeCun (2022).** The paper describes six
    unnumbered modules (configurator, perception, world model, cost, short-term memory,
    actor). The numbering comes from `docs/papers/paper-a-reframe-aeon-ami.md`.
12. **Smaller corrections.** The weights are float32 (about 1 MB), not float64. The
    vector index is the page-clustered Atlas index with exhaustive search, not HNSW.
    The std-ratio rollback is not exercised by the evaluation scripts, and it compares
    the predictor's output with the targets, not with h\_t (later review). The +20.6 %
    is relative to raw retrieval in the same run, not to a "no-centering baseline" run.
    The power statement in §7, item 1, of version 1 ("power ≥ 80 % … requires at least
    5 seeds") had no calculation behind it and is removed. Internal references a reader
    could not resolve are replaced by file paths or removed. The bullet "The null
    predictor is a meaningful upper bound" is removed. The header cited an unrelated
    erratum (E-2026-10-25-02) (later review).

The values carried over from version 1 in Tables 1 and 2 are unchanged; they were
re-read from the JSON files. The null recall@5 columns were added from the same files,
and the epochs / learning-rate column from the re-run described in §3.1. Table 3 and the
file behind it are new. The real-data scores are given to four decimals in Table 2 and
§4.3 and to three in the abstract and the French summary; 0.1765 is the value in the
file and is written 0.177 there.

---

## Abstract

We re-examine exploratory results from April 2026 on a small next-state predictor used
for memory retrieval in a conversational assistant (micro-kiki). A numpy MLP with
267,136 parameters maps a 384-dimensional latent vector h\_t (a sentence embedding in
the real-text runs), together with a one-hot "stack" (domain) identifier, to a predicted
successor ĥ\_{t+1}; retrieval then uses ĥ\_{t+1} instead of h\_t. Two transforms were
applied to the predictor's output: subtraction of a running mean ("centering") and
standardization of the predicted residual ("LayerNorm(δ)"). Neither enters the training
loss. On a synthetic drift stream, centering raised the MRR@5 of the true next item from
0.413 (raw retrieval) to 0.498 (+20.6 %); because the query is itself stored in the
index and retrieved first, 0.5 is the ceiling, and the gain moves the true next item
from third to second place (second on 48 of 100 queries with raw retrieval, on 99 with
centering). A null control that removes the domain identifier obtains the same 0.498
and is beaten on none of the 100 queries, so the gain is not attributable to domain
conditioning. A diagnostic added in this version shows that the gain depends on the
running mean being updated by the queries themselves, in stream order: with the queries
shuffled (five orders) the MRR is 0.410 to 0.430, with the running mean frozen after
training it is 0.423, and subtracting a moving average of the recent queries from the
raw query, without any predictor, gives the same 0.498. With LayerNorm(δ), synthetic
retrieval fails after 50 epochs (recall@5 0.03, against 1.00 for raw retrieval); after
300 epochs at a higher learning rate the conditioned predictor beats the null on 59 %
of queries, but that null is an input never seen in training. Four pilots on real text
(first user messages from 10 technical domains, 1,000 items, 100 queries) used the
residual normalization only: centering was never run on real data, so whether its
synthetic gain transfers is unknown. In these pilots the conditioned predictor shows no
benefit over the null that survives the protocol's limits: with domain-level scoring on
the interleaved stream the null scores higher (MRR 0.177 vs 0.163), and in the one run
where the conditioned predictor scores higher (domain-level scoring, within-topic
stream, 0.109 vs 0.069) both are far below a raw-retrieval baseline of 0.99 that the
construction makes trivial. The items are first messages of separate records in
shuffled order rather than consecutive conversation turns, queries are part of the
training data, and the null input never occurs in training. All runs are single-seed
and exploratory; neither transform has been tested in a setting that could show a
benefit of the predictor. This version corrects version 1, which wrongly concluded that
the centering gain does not transfer to real data (erratum E-2026-11-24-01).

---

## 1. Context

### 1.1 The predictor

micro-kiki's working-memory module is called Aeon. The name, and those of its two
stores (Atlas, a page-clustered vector index, and Trace, an episodic graph), come from
Arslan, "Aeon: High-Performance Neuro-Symbolic Memory Management for Long-Horizon LLM
Agents" (arXiv:2601.15311), which `src/memory/atlas.py` cites as its source. The latent
predictor studied here is the code of `src/memory/aeon_predictor.py`; it was not
compared with that paper.

`AeonPredictor` (`src/memory/aeon_predictor.py`) wraps an `AeonSleep` store with a small
numpy MLP. The store's vector index (`src/memory/atlas.py`) ranks by cosine similarity
with an exhaustive search over all stored vectors, so retrieval is exact. The MLP is
trained on successive latent states h\_t → h\_{t+1}. At query time, retrieval uses a
predicted successor ĥ\_{t+1} instead of the raw query vector h\_q. The intent is that
ĥ\_{t+1} lies closer to upcoming turns than h\_q does.

The network (`LatentMLP`) maps 384 + 16 inputs through two hidden layers of 256 units to
a 384-dimensional residual that is added to the input: ĥ = x + δ. The 16 extra inputs
are a scaled one-hot code of a "stack" identifier (in micro-kiki, the active LoRA
domain). It has 267,136 parameters stored in float32 (about 1 MB) and is trained with a
cosine loss. In the runs reported here, training is done in batch (50 or 300 epochs of
SGD) after the whole stream has been ingested.

The April 2026 reframing plan (`docs/papers/paper-a-reframe-aeon-ami.md`) presented this
predictor as an instance of the short-term memory module of LeCun's "A Path Towards
Autonomous Machine Intelligence" (OpenReview, 2022; the plan calls it "Module 7", a
numbering the paper does not use) and cited LeJEPA (Balestriero and LeCun,
arXiv:2511.08544) as context for latent prediction. That framing is not tested here.

### 1.2 The April 2026 claims

The reframing plan rated as "Strong", among others, three empirical findings (its
"Empirical scorecard", the second section numbered 4):

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
standardized across its 384 components for each sample, then added to the input (lines
136–141). The gain and bias (`_ln_gamma`, `_ln_beta`) are fixed at 1 and 0 and are never
updated. A standardized 384-dimensional residual has norm √384 ≈ 19.6, while the inputs
are unit vectors, so with this flag the direction of the output is set almost entirely
by the residual.

**None of these enters training.** `backward_cosine` recomputes ĥ = x + δ from the
cached activations without LayerNorm and without centering (lines 184–186), and the
loss and gradients are taken on that quantity. For a given seed, number of epochs and
learning rate, the trained weights are therefore the same whichever flag is set. The
result files agree: `final_train_loss` is 0.5667058005928993 in conditions C, D, F1, L1
and L3, and 0.5203294903039932 in F2 and L2. Within each training setting (50 epochs
at 1e-3: C, D, F1, L1, L3; 300 epochs at 5e-3: F2, L2), the conditions of Table 1
differ only in the transform applied to the output when the predictor is queried.

**"Anti-collapse" does not apply.** The targets are fixed vectors (embeddings from a
frozen encoder, or fixed synthetic vectors), not the output of a second trained
network. The trivial solution that anti-collapse devices guard against in
joint-embedding training, where both sides of the loss can converge to the same
constant, does not exist here: a constant output is not a minimizer of a cosine loss to
fixed, varied targets. Version 1 was wrong to describe these transforms as anti-collapse
choices; they are transforms of the query vector at retrieval time.

**The running mean is mutable at query time.** `predict_next` calls the same `forward`
(line 389), which updates the running mean on every call (lines 150–153 and 158–161),
with a batch of one. The evaluation scripts call the predictor twice per query, first
with the true stack id and then for the null predictor, so with centering the two calls
share and modify the same state. With momentum 0.9 and two updates per query, the
running mean during evaluation is dominated by the last few queries. §3.4 shows that
this is where the gain of condition D comes from.

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
all consecutive pairs. The 100 queries are the turns 899 to 998, taken in that order;
the gold answer for turn i is turn i+1. Each query is scored three ways with top-5
retrieval:

- `baseline`: retrieval with the raw query vector;
- `predictive`: retrieval with ĥ computed with the stack id stored with the query turn;
- `null`: retrieval with ĥ computed with `stack_id = -1`, which gives an all-zero stack
  input (script lines 201–202; predictor lines 408–409).

MRR is the mean reciprocal rank of the gold turn within the top five (0 if absent);
the abstract writes it MRR@5. `win_pred` is the fraction of queries where predictive
beats baseline; `win_vs_null` the fraction where predictive has a strictly higher
reciprocal rank than null.

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

**How the settings were established.** The JSON files do not record the seed, the
number of epochs or the learning rate, and the files for C, D, F1, F2 and L1 do not
contain the `use_layernorm_delta` key; the label of L1 therefore rests on its file name
and on the reproduction below. The settings in the table were inferred from an exact
re-run on 2026-11-24: with seed 0, the commands below reproduce every value of the
seven committed files (elapsed time excepted), `final_train_loss` included, to the last
digit.

```bash
E() { uv run python scripts/eval_aeon_predictor.py --stream stack-structured "$@"; }
E                                                               # C
E --use-centering                                               # D
E --use-centering --per-stack-centering                         # F1
E --use-centering --per-stack-centering --epochs 300 --lr 5e-3  # F2
E --use-layernorm-delta                                         # L1
E --use-layernorm-delta --epochs 300 --lr 5e-3                  # L2
E --use-layernorm-delta --use-centering                         # L3
```

The settings agree with the April draft (`docs/papers/paper-a-draft-v1.md`, Table 2:
"L2 … (300 ep, lr=5e-3)"; its Table 1, condition E, "300 / 5e-3", loss 0.520). With 300
epochs at 1e-3 the committed loss is not reproduced
(`results/2026-11-24-aeon-centering-query-order.json`, key `training_settings_check`).
The outputs of the re-run of L1, L2 and L3 are not committed; C, D, F1 and F2 are
re-scored by the diagnostic of §3.4, whose output file records that they match. The
command block was run as written under zsh, with `--out` added to each line.

### 3.2 What the protocol can and cannot show

These points follow from the script and bear on every row of Table 1.

- **The query is in the index.** Query turn i was itself ingested. Raw retrieval returns
  it first for all 100 queries, so the gold turn i+1 ranks second at best and the
  baseline MRR cannot exceed 0.5. For the predictive score this ceiling is not
  structural, but it holds in practice in conditions C, D, F1 and F2, where the query
  turn is also ranked first in all 100 predictive retrievals (not checked for L1 to
  L3). A baseline of 0.413 and a predictive MRR of 0.498 are both readings on this
  compressed scale: for raw retrieval the gold turn is second on 48 queries and third
  on 52, behind turn i−1; in condition D it is second on 99 and third on 1
  (`results/2026-11-24-aeon-centering-query-order.json`).
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
  selects a running-mean slot that training never updates; it is still zero when
  evaluation starts and is then updated only by the successive null queries.
- **With centering, the score depends on the order of the queries** (§3.4; shown for
  D, F1 and F2; L3 was not re-scored).

### 3.3 Observations

**Condition D.** Predictive MRR is 0.498 against a raw-retrieval baseline of 0.413
(+20.6 %; +20.1 % against the predictive MRR of condition C, 0.415). The null predictor
obtains the same 0.498 and `win_rate_stack_vs_null` is 0: on no query does the true
stack id do better than the all-zero input. The gain does not depend on the stack
input and is not evidence for stack conditioning. Since the trained weights are those
of condition C, it is not evidence about training either. §3.4 shows where it comes
from.

**Conditions F1 and F2.** The null predictor scores higher than the conditioned one
(0.495 vs 0.413; 0.498 vs 0.433) and `win_rate_stack_vs_null` is 0 in both. This
comparison does not isolate the stack input: the null's running-mean slot is filled
only by the successive null queries, and §3.4 shows that the null's advantage comes
from that. These runs do not show that per-stack centering is harmful in general. They
show that, as implemented and evaluated, it gave the conditioned predictor little or no
gain (0.413 and 0.433, against 0.413 for raw retrieval), while the null gained more for
the reason given in §3.4.

**Conditions L1, L2, L3.** With LayerNorm(δ) at 50 epochs (L1) and with both transforms
(L3), retrieval fails almost entirely (recall@5 of 0.03 and 0.01, against 1.00 for the
baseline). At 300 epochs and a learning rate of 5e-3 (L2), predictive MRR is 0.447,
recall@5 is 0.67 (below the baseline's 1.00), and the conditioned predictor beats the
null on 59 % of queries. L2 and L1 differ in both epochs and learning rate. Given the
points of §3.2, the 59 % cannot be read as the predictor using stack information about
the next turn. It may reflect the all-zero null input being out of distribution for an
output that depends almost entirely on the residual, or the stack id informing the
predictor about turn i−1, the competitor for second place. Neither was tested.

### 3.4 Diagnostic added in version 2: the centering gain depends on evaluation-time state

`scripts/diag_aeon_centering_query_order.py` re-scores conditions C, D, F1 and F2 on
the same trained weights and changes only how the centering state evolves during
evaluation. Output: `results/2026-11-24-aeon-centering-query-order.json`. Single seed
(0), 100 queries, exploratory, same limits as §3.2. The "as run" variant reproduces the
committed April files exactly (`as_run_matches_committed_file` is true for the four
conditions).

**Table 3. Predictive MRR / null MRR under changes of the evaluation-time state.**

| Variant | C (no centering) | D (shared) | F1 (per-stack) | F2 (per-stack, 300 ep) |
|---|---|---|---|---|
| As run: stream order, predictive call then null call | 0.415 / 0.412 | 0.498 / 0.498 | 0.413 / 0.495 | 0.433 / 0.498 |
| Null call before predictive call | 0.415 / 0.412 | 0.498 / 0.498 | 0.413 / 0.495 | 0.433 / 0.498 |
| Queries in reverse stream order | 0.415 / 0.412 | 0.162 / 0.152 | 0.402 / 0.327 | 0.408 / 0.328 |
| Queries shuffled, five orders (min–max) | 0.415 / 0.412 | 0.410–0.430 / 0.412–0.417 | 0.402–0.412 / 0.410–0.420 | 0.420–0.425 / 0.418–0.425 |
| Running mean restored to its post-training value before every call | 0.415 / 0.412 | 0.423 / 0.417 | 0.407 / 0.412 | 0.422 / 0.420 |

JSON keys: `conditions.<C|D|F1|F2>.variants.<as_run|null_first|reversed|shuffled.<seed>|frozen>`,
fields `predictive_mrr` and `null_stack_mrr`. Shuffle seeds 1 to 5.

A control without any predictor (key `conditions.D.no_mlp_ema`): the raw query minus an
exponential moving average (momentum 0.9) of the raw queries seen so far, in stream
order, is used for retrieval. MRR is 0.498 with one or two updates per query, starting
from zero or from the post-training running mean; in each of the four settings the rank
of the gold turn is the same as in condition D for every one of the 100 queries (key
`gold_ranks_identical_to_as_run_predictive`).

The file also records, for each query, which turn holds second place (keys
`*_per_query.second_place`). With raw retrieval it is turn i−1 on 52 queries and the
gold turn i+1 on 48. In condition D as run it is turn i−1 on 1 query and the gold turn
on 99; with the running mean frozen, 46 and 54; in reverse order, turn i−1 on 99.

**What this shows.**

- Condition C, which has no centering state, is unaffected by any variant. The effects
  in the other columns come from the centering state.
- In condition D, the 0.498 is obtained only when the queries are scored in stream
  order with the running mean updated along the way. With the queries shuffled the
  predictive MRR is 0.410 to 0.430, and with the running mean frozen at its
  post-training value it is 0.423, against 0.413 for raw retrieval and 0.415 for
  condition C. In reverse order it falls to 0.162, below raw retrieval: the running
  mean is then built from turns i+1, i+2, …, so subtracting it demotes the gold turn
  itself.
- The predictor is not needed to obtain 0.498: subtracting a moving average of the
  recent queries from the raw query gives the same score and the same rank of the gold
  turn on every query.
- In F1 and F2 the same holds for the null: its 0.495 and 0.498 fall to 0.410–0.425
  when the queries are shuffled or the state is frozen, and the null then no longer
  differs from the conditioned predictor by more than 0.013.

**Reading.** During evaluation the running mean is an average of the last few queries,
which are the turns just before the current one on the drift path. Subtracting it
pushes the query away from where the stream has just been, which demotes turn i−1 (the
second-place counts above) and leaves turn i+1 in second place. This is a property of
scoring consecutive turns of a drift process in order with a state that the scoring
itself updates. It is not a property of the trained predictor, of the stack input, or
of centering as a remedy for collapse. Whether such a moving-average query transform
would help on a real conversation was not tested. Only momentum 0.9 was run; condition
L3, which also uses centering, was not re-scored.

---

## 4. Real-data pilots

### 4.1 Protocol and results

`scripts/eval_aeon_realdata.py` reads `data/final/<domain>/train.jsonl` for 10 domains
(dsp, electronics, emc, embedded, freecad, kicad-dsl, platformio, power, spice, stm32).
`data/` is not tracked by git, so the description below comes from the script; the
records themselves were not read. For each domain it takes the first user message of
each record until it has 100 (lines 101–119) and embeds them with a 384-dimensional
sentence-embedding model loaded from `models/niche-embeddings`. The messages of each
domain are shuffled (lines 132–133 and 141–142). The script then builds one stream of
1,000 vectors in one of two ways:

- `interleaved`: the domains are visited round-robin, in a reshuffled order at each
  round (lines 139–150);
- `within-topic`: the 100 messages of each domain are placed contiguously (lines
  127–136).

The stack id is the domain index. All 1,000 vectors are ingested and the predictor is
trained on all consecutive pairs (lines 226–235); the 100 queries are the turns 899 to
998 (lines 240–248). The script docstring says "train on first 80%, evaluate on last
20%"; the code does not hold anything out. "Exact" scoring asks for the id of turn i+1
in the top five; "soft" scoring accepts any stored turn of the same domain as turn i+1.

Line numbers are those of the script at commit `ba5b1f5`. The two exact-scoring files
were committed before the script's last revision (`32fb472`; the first of them also
before `89c6bf3`), so they were produced by earlier revisions of it.

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
within-topic rows (0.17055095545947552): each pair has an identical training (the runs
are deterministic), scored two ways in separate runs.

**Next-domain classification** (`results/aeon-next-domain.json`,
`scripts/eval_aeon_next_domain.py`, interleaved stream, LayerNorm(δ), no centering
option in the script). The predicted vector is assigned to the nearest domain centroid.
Accuracy is 0.10 for 10 domains and top-3 accuracy 0.33; with ten equally frequent
domains, guessing gives 0.10 and 0.30, and guessing among the nine domains other than
the current one about 0.11 and 0.33. All 100 evaluated transitions are domain
switches, so the "stay on the current domain" baseline scores 0.00 and the
switch-detection rate of 1.00 only says that the predicted domain always differed from
the current one. Unlike `scripts/eval_aeon_predictor.py` and
`scripts/eval_aeon_realdata.py`, this script trains only on the turns before the query
window.

### 4.2 What the protocol can and cannot show

- **The stream is not a conversation.** Consecutive vectors are first messages of
  separate records, in shuffled order. Turn i+1 is a draw from another domain
  (interleaved) or from the same domain (within-topic), not a reply to or continuation
  of turn i. Nothing in the construction ties a specific turn i+1 to turn i beyond its
  domain.
- **Exact scoring is at chance level by construction.** Since turn i+1 is a shuffled
  draw from its domain, even a predictor that knew the next domain could reach only
  about 5/100 recall@5; uniform random retrieval gives about 0.005, or about 0.004 with
  the query itself in first place (analytical estimates, not measurements). The
  observed values are one or two hits in 100 queries. Version 1 attributed them to a
  memory "too sparse at N = 1,000"; they say nothing about density.
- **The soft within-topic baseline is trivial.** The query is in the index and belongs
  to the same domain as its successor except at the single block boundary inside the
  query window, so raw retrieval scores 99 out of 100 without predicting anything. A
  transform of the query can raise this score by at most 0.01.
- **The soft interleaved scores are at or below random retrieval.** If, as the file
  `results/aeon-next-domain.json` reports for a stream built the same way
  (`n_switches_in_eval` = 100; same construction, and same seed if both runs used the
  script defaults), the next turn belongs to a different domain from the query in all
  100 query transitions, then five stored turns drawn at random, each with a 10 %
  chance of being of the gold domain, would contain it with probability
  1 − 0.9⁵ ≈ 0.41, or about 0.34 if the query itself takes the first place (analytical
  estimates, not measurements). The baseline's recall@5 of 0.11 is well below that, as
  expected if the neighbours of a query come mostly from its own domain (not checked:
  the data are not in the repository). The predictive and null recall@5 of 0.31 and
  0.29 are at or below it, so the rise from 0.11 is not evidence of prediction.
- **Same limits as the synthetic protocol:** queries are in the training pairs, the null
  input is never seen in training, and there is one seed.

### 4.3 Observations

In none of the four rows does the conditioned predictor show a benefit over the null
predictor that survives these limits. In the soft interleaved row, which
`docs/SESSION-2026-04-19-INDEX.md` recorded in April as a positive result, the null MRR
is higher than the predictive MRR (0.1765 vs 0.1628) and the conditioned predictor wins
on 12 % of queries. In the soft within-topic row the predictive MRR is higher than the
null (0.1093 vs 0.0687, wins on 27 % of queries), but both are far below a baseline
that is trivially 0.99. The exact rows are at chance. The next-domain accuracy is at
chance.

These pilots concern the LayerNorm(δ) variant only. They do not test centering, and
because of the protocol they are weak evidence even about LayerNorm(δ): they could not
have shown a benefit for next-turn prediction, since the stream has no turn-to-turn
structure beyond the domain.

---

## 5. Why the synthetic and real-data results cannot be compared

Version 1 set the +20.6 % of condition D against the real-data pilots and offered three
hypotheses for the "gap". Two of them (H2, H3) explained it by the behavior of
centering on real embeddings. They are withdrawn: no real-data run used centering. The
third (H1, saturation of the synthetic stream) is replaced by the description of the
protocol in §3.2 and by the diagnostic of §3.4.

The two sets of runs differ in the output transform (centering in D, LayerNorm(δ) in
all real-data runs), in the stream (a drift process with turn-to-turn structure, against
shuffled draws within or across domains), and in what the baseline measures. A
difference between them cannot be assigned to any one of these.

---

## 6. What this note does not claim

- It does not claim that centering or LayerNorm(δ) is useless as a query transform in
  general. As implemented, on the synthetic stream, LayerNorm(δ) at 50 epochs made
  retrieval fail (Table 1, L1 and L3) and the centering gain is reproduced without the
  predictor (§3.4); neither was tested on real conversations, and centering was never
  run on real data.
- It does not claim that the synthetic gain of condition D fails to transfer to real
  data. That transfer was not tested.
- It does not claim that the predictor code is faulty as software. Its 33 unit tests
  pass (`tests/memory/test_aeon_predictor.py`, re-run 2026-11-24). They test shapes,
  training and state handling, not the scientific claims.
- It does not claim that condition L2 shows a use of stack information, nor that it
  shows the absence of one.
- It does not assess the framing of the predictor as the short-term memory module of
  LeCun (2022).
- It does not claim that results would be the same at larger scale or on real
  multi-turn conversations.

What the files support is narrower: (a) in condition D the predictive and null MRR are
equal, so the +20.6 % is not attributable to stack conditioning, and the diagnostic of
§3.4 attributes it to a running mean updated by the queries in stream order; (b) in the
real-data pilots of the LayerNorm(δ) variant, the conditioned predictor shows no benefit
over the null that survives the protocol's limits (the null scores higher in the soft
interleaved run; the conditioned predictor scores higher in the soft within-topic run,
where both are far below a trivial baseline); (c) every run is single-seed and
exploratory.

---

## 7. Status of MK-11

Catalogue v3 records MK-11 ("centering +22 % MRR on structured streams") as "réfuté
(données réelles)" and cites `results/aeon-realdata.json`. That file is a LayerNorm(δ)
run without centering.

On the evidence in this repository, the descriptive figure of MK-11 is reproduced on
the synthetic stream (0.413 → 0.498, +20.6 %, not +22 %), but it is not attributable to
stack conditioning (null equal to predictive) nor to training (weights identical to
condition C). It is obtained only when the queries are scored in stream order with the
running mean updated during evaluation (0.410 to 0.430 with shuffled queries, 0.423
with the running mean frozen), and it is reproduced without the predictor (§3.4). It is
untested on real data. The label "réfuté (données réelles)" rests on a run without
centering. Whether the catalogue label should change is a decision for the principal
investigator; this note does not change it.

---

## 8. What a valid test would need

1. **Centering actually run on real data**, with the flags recorded in the result file
   together with seed, epochs and learning rate.
2. **Real consecutive turns.** A stream built from multi-turn conversations, so that
   turn i+1 depends on turn i.
3. **Held-out queries.** Queries and their gold successors excluded from training and,
   for the baseline to be meaningful, the query itself excluded from retrieval.
4. **A matched null.** The null predictor should receive a valid stack id drawn at
   random or permuted, not a pattern absent from training.
5. **Controlled evaluation-time state.** Either the running means are frozen during
   evaluation, or the transform is declared as a test-time moving-average transform and
   compared with the same transform applied to the raw query (the control of §3.4),
   with the query order stated.
6. **A stack id that refers to the transition being predicted**, or an explicit
   statement of what the stack id is expected to inform.
7. **No anti-collapse claim.** If kept, the transforms should be evaluated as test-time
   query transforms (§2).
8. **Several seeds.** The lab's working rule is at least five seeds with separate pilot
   and confirmatory seed sets; this is a rule of thumb, not the result of a power
   calculation.
9. **Pre-registration** of the hypothesis and analysis plan before the runs. The April
   2026 runs were pilots.
10. **A raw-retrieval (RAG) baseline** on the same corpus (issue #13, NH-85, deferred).

---

## References

- Arslan, M. (2026). Aeon: High-Performance Neuro-Symbolic Memory Management for
  Long-Horizon LLM Agents. arXiv:2601.15311.
- Balestriero, R., LeCun, Y. (2025). LeJEPA: Provable and Scalable Self-Supervised
  Learning Without the Heuristics. arXiv:2511.08544.
- Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.
  (2021). Emerging Properties in Self-Supervised Vision Transformers (DINO).
  arXiv:2104.14294.
- LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. Version 0.9.2,
  27 June 2022. OpenReview position paper (no arXiv version).
  https://openreview.net/forum?id=BZ5a1r-kVsf
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
| `results/2026-11-24-aeon-centering-query-order.json` | added with version 2 | 2026-11-24 |

The last file is produced by `scripts/diag_aeon_centering_query_order.py` (added with
version 2); three consecutive runs gave byte-identical output.

Code read for this revision, at commit `ba5b1f5`: `src/memory/aeon_predictor.py`,
`src/memory/aeonsleep.py`, `src/memory/atlas.py`, `scripts/eval_aeon_predictor.py`,
`scripts/eval_aeon_realdata.py`, `scripts/eval_aeon_next_domain.py`.
