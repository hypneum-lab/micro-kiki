# Stack-Conditioned Prediction under Centering Regularization: A Case Study in Latent Predictors

## Prédiction conditionnée par le stack sous régularisation par centrage : une étude de cas sur les prédicteurs latents

**Auteurs** : équipe micro-kiki
**Date** : 2026-04-19
**Statut** : rapport technique (non peer-reviewed)
**Papier companion** : Paper A — *Aeon as a Candidate Short-Term Memory Module for AMI-class Systems* (en préparation ; sera référencé une fois soumis)

---

## Abstract

Nous rapportons un résultat mixte : quatre résolutions proposées à une incompatibilité centrage–stack-conditioning, dont deux testées empiriquement. Le stack-conditioning (encodeur one-hot) fonctionne en isolation (23 % de `win_stack` sur flux random-walk) mais s'effondre sous le centrage par moyenne mobile à la DinoV3/JEPA sur flux structurés (0 % en D, 1 % en E). Le centrage per-stack (proposition Section 6.1) NE récupère PAS le signal (0 % win_stack à 50 comme à 300 epochs), indiquant que le problème est plus profond qu'une interférence de moyenne partagée. La LayerNorm per-sample du delta résiduel (proposition Section 6.5) RÉUSSIT : la condition L2 (300 epochs, lr=5e-3) atteint 59 % de `win_stack` avec predictive_mrr à 0.447 vs null_mrr à 0.090. Cela identifie LayerNorm(delta) comme un mécanisme anti-collapse compatible avec le centrage qui préserve le stack-conditioning discret — parce que la normalisation au niveau per-sample préserve les offsets per-sample que le centrage par batch efface. Combiner LayerNorm(delta) avec le centrage est catastrophique (L3 : 1 % `win_stack`). Le centrage apporte sa propre amélioration de +22 % MRR indépendamment du conditioning ; c'est le focus du Paper A. Nous clarifions l'interaction régulariseur-conditionneur et validons empiriquement une mitigation.

---

## 1. Introduction

**Contexte.** Les stacks de serving LLM modernes entourent le modèle de base d'une couche mémoire à long contexte — RAG, memory palaces (Aeon), approches virtual-OS (MemGPT). Une étape naturelle suivante, préfigurée par les world models JEPA, est un *prédicteur* qui anticipe les items mémoire pertinents au tour suivant pour prefetch, pre-ranking ou decoding spéculatif. Notre déploiement cible est micro-kiki, une stack LoRA 35 domaines sur Qwen3.5-35B-A3B où chaque tour est routé vers un expert par un classifieur appris.

**La claim du stack-conditioning.** Dans un pipeline activant le routage, chaque tour est pris en charge par l'un des N experts (N=16 dans le sous-ensemble PoC de 35). Puisque chaque stack a sa propre signature distributionnelle, le prédicteur devrait bénéficier de savoir quel stack a produit `h_t`. Le Paper A v1 encodait cela à la manière MTL : un identifiant de stack one-hot concaténé à `h_t` avant le MLP.

**Ce que montre ce case study.** La voie one-hot fonctionne en isolation — la condition A donne 23 % de `win_stack`. Avec le centrage activé, le signal disparaît : D donne 0 %, E 1 % malgré 6× plus d'entraînement. La soustraction par la moyenne mobile calcule l'espérance *globale* de la sortie et la retire ; la variance inter-stacks s'effondre. Les deux mécanismes sont, sous leur forme actuelle, mutuellement exclusifs.

---

## 2. Background

### 2.1 Prédicteurs latents dans les systèmes à mémoire augmentée

La famille JEPA — I-JEPA [1], V-JEPA 2 [2] — argumente pour la prédiction dans l'espace *latent* plutôt que pixel/token. Les world models à espace latent — DreamerV3 [3], TD-MPC2 [4] — couplent dynamique apprise et planification. Les LMs à mémoire augmentée comme MemGPT [5] fournissent le substrat de retrieval mais n'incluent pas encore par défaut de prédicteurs next-state à la JEPA. Notre design est un MLP numpy d'environ 100K paramètres au-dessus du substrat Atlas (vecteurs denses) et Trace (graph temporel) existant.

### 2.2 Prévention du collapse

I-JEPA [1] utilise un encodeur target EMA ; DINO [6] et DINOv3 [7] combinent centrage et sharpening ; LeJEPA [8] introduit SIGReg, un régulariseur de projection Cramér-Wold remplaçant les heuristiques EMA. Notre design suit la lignée DINOv3 — une moyenne mobile des sorties prédites, soustraite avant la loss cosinus — mais stateless (pas de teacher, pas de sharpening), appariée avec un tripwire std-ratio qui rollback les poids quand `std(ĥ)/std(h) < 0.1`. Centrage + rollback est le sujet du papier companion.

### 2.3 Mécanismes de conditioning

MTL concatène un identifiant de tâche à l'entrée — notre choix initial. Le routage MoE [9] dispatche vers l'un des N sous-réseaux. Les hypernetworks [10] génèrent les poids du prédicteur à partir de l'entrée de conditioning. La concaténation one-hot est la plus simple — et, comme nous le montrons, la plus exposée à l'interférence par centrage.

---

## 3. Méthodologie

### 3.1 Architecture

Le prédicteur est un MLP à 2 couches avec une connexion résiduelle : `h_{t+1} = skip·h_t + W_2 · ReLU(W_1 · [h_t ; α·one_hot(s)])`. Dimensions 384 → 256 → 384, ~100K paramètres entraînables (numpy float32). Loss cosinus (MSE collapserait vers la moyenne). Le vecteur one-hot de stack a dimension `n_stacks = 16`, concaténé à *l'entrée* du MLP. Source : `src/memory/aeon_predictor.py` (~280 lignes).

### 3.2 Centrage

Après chaque forward pass, nous mettons à jour `μ ← 0.9·μ + 0.1·mean(ĥ)` et soustrayons `μ` de `ĥ` avant la loss cosinus — à la DinoV3 mais stateless (pas de teacher, pas de sharpening, pas de stop-gradient). Moyenne mobile figée à l'inférence. Un tripwire std-ratio restaure le dernier checkpoint si `std(ĥ)/std(h) < 0.1`.

### 3.3 Stack-conditioning

L'identifiant de stack de l'échantillon `s ∈ {0,…,15}` est encodé one-hot et concaténé à `h_t`. Puisque `dim(h_t)=384` domine `n_stacks=16`, nous appliquons un facteur de scaling `α = √(dim/n_stacks) = √24 ≈ 4.9` pour amener la norme du one-hot dans le même ordre que la norme de l'embedding. Sans ce fix (ajouté en PoC B v2 après que le v1 a montré le one-hot brut se faire écraser), `win_stack` reste à un chiffre même sur la condition A. Le patch de scaling est ce qui produit le chiffre de 23 % ci-dessous.

### 3.4 Setup expérimental

Cinq conditions sur flux synthétiques de 1000 tours, 100 requêtes held-out :

- **A (vanilla)** : flux random-walk (stack non corrélé à la dynamique), pas de centrage, 50 epochs, lr=1e-3.
- **B (+centrage)** : même flux, centrage activé.
- **C (stack-stream)** : flux structuré stack (drift per-stack dans l'espace latent), pas de centrage.
- **D (stack+centrage)** : stack-structured + centrage.
- **E (D long)** : D avec 300 epochs, lr=5e-3 — une convergence plus longue récupère-t-elle le signal de stack ?

Métriques : `recall@5` et `MRR` pour trois variantes — baseline (retrieval seul), predictor-aware (top-5 reranké par similarité avec `ĥ_{t+1}` prédit), null-stack (prédicteur avec one-hot à zéro). Dérivées : `win_pred` (% requêtes où le prédicteur bat la baseline sur `recall@5`), `win_stack` (% requêtes où le stack-aware bat le null-stack) — cette dernière isole la contribution du stack-conditioning.

Artefacts : `/Users/electron/Documents/Projets/micro-kiki-poc-aeon/results/2026-04-17-aeon-poc-{A,B,C,D,E}-*.json`, narratifs `…/2026-04-17-aeon-predictor-poc-{alpha,beta,gamma}.md`.

---

## 4. Résultats

### 4.1 Tableau des résultats

| Condition | baseline_r@5 | predict_r@5 | null_r@5 | baseline_mrr | predict_mrr | null_mrr | win_pred | win_stack | final_loss |
|-----------|--------------|-------------|----------|--------------|-------------|----------|----------|-----------|------------|
| A (vanilla) | 0.66 | 0.62 | 0.62 | 0.263 | 0.264 | 0.252 | 20 % | **23 %** | 0.835 |
| B (+ centrage) | 0.66 | 0.53 | 0.56 | 0.263 | 0.228 | 0.232 | 17 % | 18 % | 0.835 |
| C (stack-stream) | 1.00 | 1.00 | 1.00 | 0.413 | 0.415 | 0.412 | 5 % | 5 % | 0.567 |
| D (stack+centrage) | 1.00 | 1.00 | 1.00 | 0.413 | 0.498 | 0.498 | 51 % | **0 %** | 0.567 |
| E (D long) | 1.00 | 1.00 | 1.00 | 0.413 | 0.500 | 0.498 | 52 % | 1 % | 0.520 |

### 4.2 Résultat clé : le centrage détruit le stack-conditioning

L'observation principale est `predict_mrr ≈ null_mrr` sur D et E. Sur D ils sont bit-for-bit identiques (0.498). Sur E, avec 6× d'entraînement et un learning rate plus élevé, l'écart est d'un point MRR (0.500 vs 0.498) et `win_stack = 1 %`. Le signal one-hot n'a pas d'effet mesurable sous centrage — pas du bruit noyant le signal, mais un signal qui est *soustrait*.

### 4.3 Le centrage délivre par lui-même

Mettons de côté le stack-conditioning et D est une story positive : baseline MRR 0.413 → prédicteur MRR 0.498, +22 % relatif, `win_pred = 51 %`. E pousse le MRR à 0.500 à 52 % win. Le centrage est une contribution non triviale *indépendante* du stack-conditioning — d'où le recadrage du Paper A. Sur random-walk (A–B) le centrage nuit légèrement au MRR (0.263 → 0.228) : son bénéfice se concentre dans le régime « rerank within saturated recall ».

### 4.4 Le stack-conditioning fonctionne en isolation

La condition A montre qu'un one-hot avec scaling dimension-matched n'est pas vacuous : `win_stack = 23 %`, `predict_mrr` (0.264) > `null_mrr` (0.252). Le mécanisme fonctionne quand on le laisse faire. Ce qui le tue en D est la couche de centrage, pas le mécanisme lui-même ni un flux trop facile.

### 4.5 Plafond de saturation

Sur C, D, E le `recall@5` baseline = 1.0 pour chaque requête : le retrieval seul résout la tâche. Le MRR, pas le recall, est le seul axe que le prédicteur peut améliorer. Le plafond est une caractéristique du générateur synthétique et borne le headroom observable. Un benchmark plus difficile (distance plus bruitée, plus grande galerie, plus de distracteurs) pousserait le `recall@5` baseline en dessous de 1.0 et exposerait une fenêtre MRR plus large. C'est la principale limitation expérimentale.

### 4.6 Ablation étendue — candidats de mitigation (conditions F, L)

Pour tester deux des quatre résolutions proposées (Sections 6.1, 6.5), nous avons ajouté les conditions F (centrage per-stack) et L (LayerNorm du delta) :

| Condition | win_stack | predictive_mrr | null_mrr | baseline_mrr | Verdict |
|-----------|-----------|----------------|----------|--------------|---------|
| F1 : centrage per-stack (50 ep) | 0 % | 0.413 | 0.495 | 0.413 | FAIL |
| F2 : centrage per-stack (300 ep) | 0 % | 0.433 | 0.498 | 0.413 | FAIL |
| L1 : LayerNorm(delta) (50 ep) | 3 % | 0.012 | 0.015 | 0.413 | sous-entraîné |
| **L2 : LayerNorm(delta) (300 ep, lr=5e-3)** | **59 %** | **0.447** | **0.090** | 0.413 | **SUCCESS** |
| L3 : LayerNorm(delta) + centrage | 1 % | 0.005 | 0.012 | 0.413 | catastrophique |

**Résultats clés :**
- Le centrage per-stack (F1, F2) ne récupère pas le signal de stack. Maintenir 32 moyennes mobiles séparées par `stack_id` et soustraire `μ_s` de la prédiction de chaque échantillon produit toujours 0 % de `win_stack` à 50 comme à 300 epochs. Cela révèle que le diagnostic est plus profond qu'une interférence de moyenne partagée : même la normalisation intra-stack supprime le signal.
- LayerNorm(delta) (L2) réussit : à 300 epochs avec lr=5e-3, `win_stack = 59 %` et le prédicteur stack-aware (0.447 MRR) bat de manière décisive le null-stack (0.090 MRR) — un écart relatif de +397 %. L'entraînement précoce (L1, 50 epochs) montre 3 % de `win_stack`, indiquant que la convergence est lente mais finit par converger.
- Combiner LayerNorm(delta) + centrage (L3) est catastrophique : `win_stack = 1 %`, `predictive_mrr = 0.005`, pire que L1. Les deux mécanismes doivent être utilisés de manière exclusive.

Artefacts : `/Users/electron/Documents/Projets/micro-kiki-poc-aeon/results/2026-04-17-aeon-poc-{F,L}-*.json`, narratifs `…/2026-04-17-aeon-predictor-poc-{delta,layernorm}.md`.

---

## 5. Diagnostic : pourquoi centrage et stack-conditioning sont mutuellement exclusifs

### 5.1 Vue mathématique

La pré-activation de la première couche pour un échantillon avec stack `s` est `z = W_1^{(h)}·h_t + α·W_1^{(s)}[:, s] + b`, où `W_1^{(s)}[:, s]` est la colonne de la première matrice de poids qui absorbe le one-hot. Le terme `α·W_1^{(s)}[:, s]` est un offset additif spécifique au stack. Après ReLU et `W_2`, il se propage en sortie comme une moyenne per-stack `μ_s`.

Le centrage par moyenne mobile maintient `μ ≈ E_s[μ_s] ≈ (1/N) Σ_s μ_s` sous fréquence de stack à peu près uniforme. Soustraire `μ` laisse nominalement intacte la variance inter-stacks, mais la loss tire ensuite la prédiction centrée vers le vrai `h_{t+1}` (indépendamment de l'offset injecté). En pratique, le prédicteur annule sa dépendance à `s` : puisque l'offset de stack ne peut plus aider après centrage, le gradient à travers `W_1^{(s)}` rétrécit et la voie one-hot devient poids mort.

Ce n'est pas un bug dans la formulation du centrage — c'est précisément ce *à quoi* sert le centrage DINOv3, introduit justement pour prévenir les solutions triviales per-class [6, 7]. Dans notre setting, cette « solution triviale per-class » est ce que nous voulions.

### 5.2 Pourquoi cela n'affecte pas A (random-walk)

Sur A, la dynamique du flux est indépendante de `s`. Le prédicteur n'a aucune raison d'exploiter le one-hot et le centrage n'a rien de spécifique au stack à retirer — le one-hot agit comme un conditioning bruité faible que le facteur de scaling rend juste détectable (d'où 23 %). B dégrade légèrement cela en retirant quel que soit le faible signal per-stack existant.

### 5.3 Pourquoi cela détruit D (stack-structured)

Sur D, la dynamique *est* spécifique au stack. Sans centrage, le prédicteur apprendrait à ajouter `μ_s` à ses prédictions et à ranger plus haut le bon next-state par stack. Avec le centrage, `μ_s` est précisément ce qui est retiré. Le 0 % de `win_stack` est le résultat attendu.

### 5.4 Diagnostic affiné après expériences

L'échec de F (centrage per-stack) a révélé que le diagnostic est plus profond qu'une interférence de moyenne partagée. Même quand nous maintenons 32 moyennes mobiles séparées — une par `stack_id` — et soustrayons `μ_s` de la sortie de chaque échantillon, le signal de stack s'effondre quand même (0 % de `win_stack` sur F1, F2). Le problème n'est pas que les moyennes soient poolées, mais que la normalisation (qu'elle soit globale ou per-stack) opère sur l'axe du batch.

Le succès de L2 (LayerNorm(delta)) identifie le bon axe : **per-sample, pas per-batch**. La LayerNorm normalise le delta résiduel de chaque échantillon en isolation. L'offset spécifique au stack injecté via concaténation one-hot devient une constante per-sample ; la LayerNorm préserve cette constante (gamma, beta appris l'amplifient si besoin). La normalisation par batch — globale ou per-stack — moyenne à travers des échantillons de stacks différents et efface l'offset au sein des batches de chaque stack.

Cela explique aussi la catastrophe de L3 : LayerNorm(delta) + centrage par moyenne mobile compose la pression de régularisation sur la sortie. Le premier mécanisme normalise la variance des features par échantillon ; le second retire la dérive de la moyenne par batch. Ensemble ils ne laissent aucun signal cohérent que le prédicteur puisse apprendre, pire que l'un ou l'autre seul.

---

## 6. Résolutions proposées

**6.1 Centrage per-stack.** Maintenir `N` moyennes mobiles `μ_1,…,μ_N`, soustraire `μ_s` en utilisant l'id de stack de l'échantillon. Préserve les offsets inter-stacks tout en normalisant la distribution intra-stack. Coût mémoire `O(N·dim)` — ~24 KB pour `N=16, dim=384`, négligeable. Un lookup de hash par forward. **STATUT EMPIRIQUE : TESTÉ — ÉCHEC** (conditions F1, F2 ; 0 % de `win_stack` à 50 comme à 300 epochs). L'échec indique que le centrage par batch (même per-stack) est fondamentalement incompatible avec le conditioning per-sample.

**6.2 Dense conditioning.** Remplacer le one-hot par un vecteur dense — p.ex. la sortie soft-max d'un router upstream (le VQC de micro-kiki). Chaque dimension porte de l'information, donc le centrage ne peut effacer que la *moyenne* de cette distribution, pas sa structure per-dimension. Des observations préliminaires sur une branche non reliée montrent moins d'interférence ; la caractérisation complète est un travail futur. **STATUT EMPIRIQUE : NON TESTÉ** (en attente du design doc D, ETA prochain sprint).

**6.3 Centrage retardé.** Appliquer le centrage seulement après une phase de warm-up (p.ex. les premiers 20 % d'epochs). Le prédicteur établit les offsets per-stack d'abord ; le centrage façonne ensuite le résidu. Commun dans les schedules JEPA, peu coûteux à implémenter, non testé ici — l'expérience la moins chère à lancer ensuite. **STATUT EMPIRIQUE : NON TESTÉ**.

**6.4 Architectures empilées (hypernetworks, MoE).** Utiliser `s` pour *générer les poids* plutôt que d'ajouter un offset en entrée. Un hypernetwork [10] produit `W_1^{(s)}` à partir de `s` ; un MoE-predictor [9] sélectionne l'un des `N` sous-prédicteurs complets. Le centrage en sortie ne peut pas effacer une différence au niveau des poids comme il efface un offset additif. Coût : `N×` paramètres pour un MoE complet (1.6M pour `N=16` à 100K params) — déployable mais une hausse substantielle de budget. **STATUT EMPIRIQUE : NON TESTÉ**.

**6.5 LayerNorm per-sample du delta résiduel.** Au lieu de soustraire les statistiques de batch (centrage), normaliser le résidu `delta = mlp(x) - x` per-sample en utilisant la LayerNorm standard. Implémentation : calculer la moyenne et la variance à travers la dimension des features pour CHAQUE échantillon indépendamment ; normaliser ; appliquer gamma/beta appris. **STATUT EMPIRIQUE : TESTÉ — SUCCÈS**.

Pourquoi cela fonctionne : l'offset `o_s` spécifique au stack injecté via concaténation one-hot produit un delta où l'offset est une constante per-sample. La normalisation per-sample préserve cette constante (gamma, beta peuvent apprendre à la réintroduire ou l'amplifier). La normalisation par batch moyenne à travers des échantillons de stacks différents et l'efface dans les batches de chaque stack. C'est la distinction clé entre régularisation per-sample et per-batch — la première est compatible avec le conditioning per-sample.

Résultats empiriques : la condition L2 (300 epochs, lr=5e-3) atteint `win_stack = 59 %`, `predictive_mrr = 0.447`, `null_mrr = 0.090` — un win décisif. La convergence est lente : à 50 epochs (L1) le signal est à 3 %, mais étendre l'entraînement navigue avec succès le paysage d'optimisation. Combiner LayerNorm(delta) ET centrage par moyenne mobile (condition L3) est catastrophique — 1 % de `win_stack`, 0.005 predictive_mrr — pire que L1 seul. Les deux mécanismes doivent être utilisés exclusivement.

Code : `src/memory/aeon_predictor.py::LatentMLP`, flag de config `use_layernorm_delta`. Commité au SHA `3c7eded` (code) et `804bf02` (benchmarks) sur la branche `feat/layernorm-delta`, mergé dans main à `d30ffb8`.

Référence : LayerNorm (Ba, Kiros, Hinton 2016) [11].

---

## 7. Discussion

**Adaptations de la famille JEPA.** Quiconque porte des régulariseurs à la JEPA (centrage DINOv3, LeJEPA/SIGReg, teachers EMA) vers un prédicteur latent *conditionnel* fait face à une version de cette tension. Le régulariseur impose une structure distributionnelle sur les sorties ; le conditionneur injecte une structure per-condition dans les sorties ; les deux se battent. Les mitigations les plus propres sont structurelles — statistiques per-condition (§6.1) ou conditioning au niveau des poids (§6.4) — fixant le problème par construction plutôt que par hack de schedule.

**Routage MoE dans les systèmes mémoire.** Les LLMs à mémoire augmentée qui routent vers des experts de domaine devraient s'attendre à cette interaction dès qu'ils ajoutent une régularisation anti-collapse en aval d'un simple conditioning par ID d'expert. Le défaut MTL emprunté de concaténation one-hot peut ne pas survivre à une régularisation agressive. Planifier le dense conditioning dès le départ, ou budgéter pour des statistiques per-expert.

**Ce que nous n'avons pas testé.** Hypernetworks, MoE-predictors, centrage retardé ou centrage per-stack — tous restent ouverts. Nous nous sommes restreints à des flux synthétiques de 1000 tours avec un signal de structure stack propre ; les embeddings conversationnels réels introduisent du bruit, de la non-stationnarité et du chevauchement qui peuvent changer le tableau. Le facteur de scaling `√(dim/n_stacks)` est une heuristique ; un paramètre appris de force de conditioning serait une généralisation peu coûteuse.

---

## 8. Leçons apprises

1. **Quand deux mécanismes ciblent chacun un « comportement au niveau de la moyenne », s'attendre à de l'interférence.** Le centrage réduit la dérive de moyenne par construction ; le conditioning one-hot ajoute des offsets de moyenne par construction. Leur composition est, par linéarité, le retrait des offsets mêmes que le conditionneur ajoute. Nous aurions pu le prédire sur papier ; nous l'avons trouvé via l'ablation à 5 conditions. Les ablations peu coûteuses attrapent ce que les recherches bibliographiques manquent.

2. **Le conditioning one-hot sparse est faible en haute dimension ; le scaling est un fix partiel au mieux.** Même avec un scaling `√(dim/n_stacks)`, le one-hot vit dans un sous-espace minuscule. Le prédicteur alloue la majorité de sa capacité à l'embedding, et toute régularisation downstream affecte disproportionnellement le petit signal. Le dense conditioning scale mieux.

3. **Régularisation per-sample vs per-batch est la distinction critique.** La LayerNorm opère per-sample ; le centrage par moyenne mobile opère per-batch. Quand le signal d'intérêt est per-sample (comme l'offset de stack-conditioning), seule la régularisation per-sample est compatible. Cela généralise au-delà de notre architecture spécifique : tout conditioning discret ou continu introduisant une structure per-sample sera en conflit avec le retrait de statistiques par batch.

4. **Tester deux résolutions proposées clarifie l'espace des résolutions.** F (centrage per-stack) échoue à court comme à long horizon, éliminant l'hypothèse « peut-être qu'il faut juste plus d'entraînement ». L (LayerNorm delta) réussit à l'échelle, validant le principe de normalisation per-sample. Ensemble, ils resserrent le diagnostic et réduisent l'incertitude sur les deux candidats restants (dense conditioning, centrage retardé).

---

## 9. Related work

Le centrage dérive de DINO [6] et DINOv3 [7] ; la philosophie EMA-free est plus proche de LeJEPA [8]. Le framing JEPA plus large (I-JEPA [1], V-JEPA 2 [2]) motive la prédiction dans l'espace latent. Les world models latents génératifs (DreamerV3 [3], TD-MPC2 [4]) offrent une alternative que nous n'avons pas poursuivie. Les travaux LM à mémoire augmentée (MemGPT [5]) forment notre substrat de déploiement. Les mécanismes de conditioning discutés incluent la concaténation MTL (notre baseline échouée), le routage MoE [9] et les hypernetworks [10]. La LayerNorm (Ba, Kiros, Hinton 2016) [11] fournit le fondement théorique pour la normalisation per-sample ; son utilisation dans les architectures Transformer et comme régulariseur anti-collapse général est bien établie, bien que son application pour préserver le signal de conditioning dans les prédicteurs latents apparaisse nouvelle dans ce contexte.

## 10. Conclusion

Ce case study documente que le centrage par moyenne mobile à la DinoV3/JEPA et le stack-conditioning discret se composent de manière catastrophique. Nous avons testé empiriquement deux des quatre résolutions proposées : centrage per-stack (condition F, ÉCHEC — 0 % de `win_stack` à 50 et 300 epochs) et LayerNorm per-sample du delta résiduel (condition L, SUCCÈS — 59 % de `win_stack` à 300 epochs, lr=5e-3).

L'insight clé est que **les mécanismes de régularisation per-sample et per-batch sont fondamentalement différents** : la LayerNorm préserve la structure per-sample ; le centrage par moyenne mobile l'efface. Quand le signal de conditioning est per-sample (comme un offset one-hot), seule la régularisation per-sample est compatible. Ce principe généralise au-delà de notre architecture spécifique.

L'échec du centrage per-stack — même avec des moyennes mobiles séparées par stack — clarifie que l'incompatibilité n'est pas à propos de statistiques poolées, mais à propos de l'axe de normalisation. Le succès de LayerNorm(delta) valide la solution per-sample et déplace la narration du papier de « 4 fixes proposés, 0 testés » vers « 4 fixes proposés, 2 testés, 2 ouverts (dense conditioning, centrage retardé, hypernetworks/MoE) ».

Le Paper A a adopté LayerNorm(delta) comme mécanisme anti-collapse préservant le stack pour le cas conditionnel, tandis que le centrage reste le mécanisme anti-collapse pour le cas non conditionnel. Ce rapport documente le mode de défaillance, son diagnostic et la mitigation validée empiriquement, avec des implications pour quiconque adapte des régulariseurs de la famille JEPA à des prédicteurs latents conditionnels.

## Références

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

**Métadonnées du document**
Auteur : équipe recherche micro-kiki
Licence : CC BY 4.0 (texte), Apache 2.0 (code companion)
Version : v1.0 (2026-04-19)
Papier companion : Paper A (en préparation)
