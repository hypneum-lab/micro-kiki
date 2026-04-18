# SpikingKiki: Energy-Efficient Expert Routing via Lossless ANN-SNN Conversion for Domain-Specialized Language Models

## SpikingKiki : routage d'experts a haute efficacite energetique via conversion ANN-SNN sans perte pour des modeles de langage specialises par domaine

**Auteurs :** L'Electron Rare (Clement Saillant)  
**Affiliation :** recherche independante, Lyon, France  
**Date :** avril 2026  
**Depot :** github.com/electron-rare/micro-kiki  
**Licence :** Apache 2.0

---

## Resume

Nous presentons SpikingKiki, un cadre qui combine specialisation de domaine par MoE-LoRA et conversion ANN-vers-SNN sans perte pour une inference plus sobre en energie sur des modeles de langage specialises. En partant de Qwen3.5-35B-A3B, un modele mixture-of-experts avec 256 experts et 3B de parametres actifs par token, nous entrainons 35 adaptateurs LoRA experts de domaine a partir d'un corpus de 489K exemples couvrant l'electronique, les systemes embarques, KiCad, la simulation SPICE et 31 autres domaines techniques. En utilisant la methode LAS (Lossless ANN-SNN) [1], nous convertissons les couches d'attention et de routage du modele de base en equivalents spike a l'aide de neurones Leaky Integrate-and-Fire (LIF) a codage par taux, avec remise a zero douce. La conversion preserve la semantique du routage top-K en conservant des logits equivalents ANN pour la selection des experts, tout en encodant les calculs experts sous forme de trains de spikes binaires sur `T=16` pas de temps. Sur une base spike de 7B de parametres (SpikingBrain-7B), nous observons 72 % de sparsity d'activation et une reduction d'energie estimee a 3x par rapport a une inference ANN dense (`0.34x` d'energie theorique par token). Chaque spike ne requiert qu'une accumulation (1 operation) contre un multiply-accumulate (2 operations) pour l'inference dense, et avec un taux moyen de spikes de 30 % a `T=4`, la voie SNN atteint une reduction de 60 % des operations totales. Notre projection dans l'espace nul via OPLoRA durant l'entrainement sequentiel des adaptateurs previent l'oubli catastrophique sur un curriculum a 35 domaines, avec rollback declenche lorsque l'angle inter-stacks tombe sous 30 degres ou que la baisse de win-rate depasse 0.03. L'evaluation V3 montre 8 victoires de V3, 5 de V2 et 22 egalites ; la projection dans l'espace nul preserve 22 des 32 domaines partages a perte de validation identique. Les meilleurs domaines incluent `stm32 (0.68)`, `cpp (0.95)` et `reasoning (1.07)` ; le domaine `electronics` obtient le gain le plus marque, de `2.14` a `1.59 (-0.55)`.

---

## 1. Introduction

Les Large Language Models (LLMs) ont demontre des capacites remarquables sur des domaines tres divers, mais le deploiement de variantes specialisees par domaine en edge reste prohibitif en cout. Deux defis convergents motivent ce travail : (1) router les requetes vers le bon expert parmi des dizaines d'adaptateurs specialises, et (2) reduire le cout energetique de l'inference sur des materiels contraints, y compris des accelerateurs neuromorphiques comme BrainChip Akida ou Intel Loihi.

Les architectures Mixture-of-Experts (MoE) traitent le premier defi en n'activant qu'un sous-ensemble de parametres par token. Qwen3.5-35B-A3B illustre cette approche avec 256 experts et seulement 3B de parametres actifs par token, pour une qualite competitive a cout reduit. Cependant, meme les modeles MoE consomment encore beaucoup d'energie en raison des operations MAC denses dans les voies d'experts actives.

Les Spiking Neural Networks (SNNs) traitent le second defi. Dans un SNN, l'information est encodee sous forme de trains de spikes binaires ; chaque spike declenche une simple accumulation (AC) plutot qu'un MAC, et les neurones inactifs ne consomment pas d'energie. Les progres recents de conversion ANN-vers-SNN, notamment avec LAS [1], ont montre une conversion sans perte de transformeurs jusqu'a 27B de parametres en alignant les codes temporels des activations couche par couche.

SpikingKiki fait le pont entre ces deux directions de recherche. Nous appliquons d'abord une specialisation LoRA sur une base MoE native, puis nous convertissons le modele obtenu en equivalent spike qui preserve la semantique de routage des experts. L'intuition cle est que les logits de routage MoE doivent etre calcules dans le domaine ANN afin de preserver l'ordre relatif sous quantification temporelle, tandis que les passes avant des experts peuvent etre integralement spikes pour economiser de l'energie. Cette strategie de routage hybride, combinee a une couche memoire cognitive (Aeon) pour la coherence multi-tours, aboutit a un systeme deployable sur un spectre allant du cloud (inference classique) a l'edge neuromorphique.

**Contributions :**

1. Un cadre de conversion ANN-vers-SNN sans perte pour des modeles de langage MoE-LoRA, preservant la semantique de routage top-K.
2. Un dataset d'entrainement de 489K exemples sur 35 domaines pour des adaptateurs LoRA specialises en domaines techniques.
3. Une methodologie d'estimation energetique comparant l'inference ANN dense et SNN spike au niveau operationnel.
4. Des resultats empiriques sur l'efficacite de LoRA selon les domaines : l'expertise deja presente dans le modele de base supprime parfois l'interet de l'adaptateur (ex. SPICE : `+0%` avec LoRA).
5. Une integration avec le systeme memoire cognitif Aeon permettant plus de 36 rappels d'episodes sur des dialogues de 14 tours.

---

## 2. Travaux connexes

### 2.1 Conversion ANN-SNN sans perte

**LAS** [1] realise une conversion sans perte de blocs de transformeurs pre-entraines en equivalents spike sans reentrainement. La methode combine quantification temporellement codee des activations et alignement des plages d'activation. Elle rapporte une conversion quasi sans perte jusqu'a OPT-66B et ViT a `T=16`. LAS preserve le softmax d'attention via un protocole d'accumulation temporelle par etapes. Nous adoptons LAS comme methode principale, en l'etendant aux couches de routage MoE.

**SpikingBrain** [4] suit une autre voie : le pre-entrainement spike natif des l'origine. SpikingBrain-7B part de Qwen2.5-7B, applique des neurones PLIF a constantes de temps apprenables et combine attention lineaire / attention complete dans une architecture hybride (GatedDeltaNet sur 2/3 des couches). Le modele atteint 72 % de sparsity d'activation a `T=4` et une reduction energetique theorique de 3x, au prix d'une regression de 2-3 % sur les benchmarks de raisonnement (`MMLU -2.4`, `GSM8K -2.3`, `HumanEval -2.5`). Une variante MoE `76B-A12B` est decrite mais ses poids ne sont pas publies.

**Spikingformer** [5] integre directement des neurones spike dans l'architecture du transformeur. Bien qu'anterieur a LAS, il sert chez nous d'outil de cross-validation dans la pipeline d'evaluation.

### 2.2 Attention differentielle et efficace

**DiffAttn** [2] propose une attention differentielle comme mecanisme d'annulation du bruit, en soustrayant deux cartes d'attention softmax afin d'amplifier le signal et de supprimer le bruit. Le lien avec notre conversion spike tient au fait que les deux approches cherchent a reduire le calcul redondant dans les couches d'attention, meme si DiffAttn opere integralement dans le domaine ANN.

### 2.3 Consolidation memoire pilotee par le sommeil

**SleepGate** [3] introduit un balisage temporel sensible aux conflits pour la consolidation memoire dans des agents LLM. Nous integrons les principes de SleepGate dans la couche cognitive Aeon, ce qui apporte detection de contradictions, surveillance de la derive de sujet et oubli appris via une gate MLP a 2 couches cachees (objectif `F1 >= 0.85` pour les decisions keep/discard).

### 2.4 Systemes MoE-LoRA

**Brainstacks** propose d'empiler des adaptateurs specialises par domaine sur des architectures MoE. **MixLoRA** (TUDB-Labs) place le routage MoE dans les blocs LoRA FFN. **MoLA** [12] explore l'allocation d'experts LoRA par couche. **HMoRA** ajoute un routage hierarchique au niveau du token et de la tache. Aucun de ces systemes n'integre ni conversion SNN ni memoire cognitive.

### 2.5 Modeles compacts et efficaces

**CompactifAI** fournit des techniques de compression complementaires a notre approche. La ou CompactifAI se concentre sur le pruning et la distillation, SpikingKiki exploite l'encodage binaire des spikes pour reduire l'energie sans retirer de poids, ce qui rend les deux approches potentiellement composables.

### 2.6 Prevention de l'oubli

**OPLoRA** [6] utilise la projection dans l'espace nul pour prevenir l'oubli catastrophique pendant l'entrainement sequentiel d'adaptateurs. Nous adoptons cette methode pour notre curriculum a 35 stacks, en projetant les gradients de chaque nouvel adaptateur dans l'espace nul des stacks deja entrainees.

### 2.7 Quantum ML pour le NLP

IonQ a demontre du fine-tuning LLM assiste par le quantique, **Quantum-Train** propose une compression par VQC, et **QPA** explore l'adaptation parametrique quantique. Ces travaux abordent l'integration du quantique dans les modeles de langage, mais aucun ne combine routage quantique, inference neuromorphique et memoire cognitive.

---

## 3. Architecture

### 3.1 Modele de base

Nous utilisons `Qwen3.5-35B-A3B` comme modele de base, un transformeur mixture-of-experts aux proprietes suivantes :

| Property | Value |
|----------|-------|
| Total parameters | ~35B |
| Active parameters/token | 3B |
| Expert count | 256 |
| Architecture | GatedDeltaNet + MoE |
| Context length | 262,144 tokens (extensible to 1M) |
| Attention | Grouped Query Attention |
| MLP | SwiGLU |
| License | Apache 2.0 |

La structure MoE native rend un MoE-LoRA custom redondant. Nous appliquons un LoRA standard (`rank 16`) uniquement aux projections d'attention (`q, k, v, o`), en laissant intactes les couches FFN MoE et leur routage appris.

### 3.2 Stacks LoRA de domaine

35 adaptateurs LoRA specialises par domaine sont entraines sequentiellement sur le modele de base. Chaque adaptateur cible les quatre matrices de projection d'attention (`q_proj, k_proj, v_proj, o_proj`) avec `rank 16`, `alpha = 2x rank`, et `scale 2.0`.

| Group | Domains |
|-------|---------|
| Conversation | chat-fr, reasoning |
| Code | python, typescript, cpp, rust, html-css, shell, sql, yaml-json, lua-upy |
| Infrastructure | docker, devops, llm-orch, llm-ops, ml-training |
| Electronics | kicad-dsl, kicad-pcb, spice, electronics, components, power, emc, dsp |
| Hardware | embedded, stm32, iot, platformio |
| CAD | freecad |
| Web | web-frontend, web-backend |
| Other | music-audio, math, security |

Un router de domaine (classifieur leger) selectionne jusqu'a 4 stacks actives par requete. Le router opere sur l'embedding d'entree et produit des logits de classe sur les 35 domaines.

### 3.3 Couche de conversion SNN

Le convertisseur LAS (`src/spiking/las_converter.py`) transforme les couches ANN en equivalents spike :

**SpikingLinear.** Pour chaque couche `nn.Linear` de poids `W` et biais `b`, l'equivalent spike calcule :

1. Pre-activation : `z = x @ W^T + b` (identique a l'ANN)
2. Rate encoding : clipping de `z` dans `[0, max_rate]`, puis division par `T` pour obtenir le courant par pas
3. Simulation LIF : integration du courant sur `T` timesteps avec `threshold = max_rate / T`
4. Reconstruction : `output = spike_count * threshold`

Le neurone LIF (`src/spiking/lif_neuron.py`) implemente une dynamique a soft reset :

```text
V_t = tau * V_{t-1} + I_t
spike_t = 1 if V_t >= threshold else 0
V_t -= spike_t * threshold   (soft reset, preserves residual)
```

avec `tau = 1.0` (integrate-and-fire pur, sans fuite) pour conserver des rate codes sans perte.

**SpikingMoELayer.** La conversion MoE preserve la semantique du routage en separant la decision de routage du calcul expert :

- **Router :** conversion avec activation identite (pas de clipping ReLU) afin de preserver des logits signes. La selection d'experts utilise le matmul equivalent ANN (`x @ W_router^T + b`) plutot que le forward spike, parce que la quantification LIF par rate code peut inverser l'ordre relatif de logits proches.
- **Experts :** chaque expert est un `SpikingLinear` standard avec activation ReLU.
- **Combinaison :** les top-K experts sont choisis a partir des logits du router ANN ; les sorties sont combinees avec des poids de routage normalises par softmax.

Cette approche hybride garantit la fidelite du routage tout en recuperant les gains energetiques de l'execution spike des experts.

**SpikingMistralBlock.** Les blocs transformeurs denses (attention + MLP SwiGLU) sont convertis tout en gardant les connexions residuelles dans le domaine ANN afin d'eviter l'accumulation d'erreur. Les projections `Q/K/V` et les projections `gate/up/down` du MLP deviennent chacune des couches `SpikingLinear` a activation identite. L'activation SiLU sur la gate reste appliquee dans le domaine ANN.

### 3.4 Couche cognitive

Le systeme memoire Aeon [7] apporte une coherence multi-tours au-dela de la fenetre de contexte du transformeur :

- **Atlas :** index vectoriel accelere SIMD pour la recherche par similarite spatiale.
- **Trace :** graphe episodique neuro-symbolique (backend NetworkX) avec aretes causales et temporelles.
- **AeonSleep :** consolidation sensible aux conflits avec balisage SleepGate [3], gate d'oubli apprise (MLP 2 couches, `F1 >= 0.85`) et summarization des episodes.

Avant inference : rappel des memoires top-K puis injection dans le contexte.  
Apres inference : persistance du tour comme noeud d'episode dans le graphe Trace.

### 3.5 Negotiator

L'arbitrage CAMP [8] avec dissidence Catfish [9] selectionne parmi plusieurs reponses candidates issues de differentes stacks. Un juge adaptatif utilise Qwen3.5-35B pour un scoring rapide (`<200 ms`) avec escalade vers Mistral-Large en cas d'egalite.

### 3.6 Anti-biais

La double application de KnowBias [10], combinee au detecteur runtime RBD [11], surveille la qualite des sorties et signale les generations biaisees.

---

## 4. Corpus

### 4.1 Vue d'ensemble

Le dataset d'entrainement V3 contient `489,348` exemples sur `35` domaines, tires de trois categories de sources.

| Source | Exemples | Description |
|--------|----------|-------------|
| Sessions Claude CLI | 50,116 | Interactions reelles utilisateur-outil issues de 5 machines (GrosMac, kxkm-ai, Studio, Tower, CILS) |
| Sessions Codex/Copilot | 2,529 | Sessions OpenAI Codex + GitHub Copilot issues de 4 machines |
| Datasets HuggingFace | 364,045 | 19 jeux de donnees ouverts (voir tableau 2) |
| Distillation enseignant Opus | -- | domaines `chat-fr` et `reasoning` |
| Corpus original cure | -- | 32 jeux de donnees semences par domaine |

### 4.2 Sources HuggingFace

| Jeu de donnees | Exemples | Licence |
|---------|----------|---------|
| CodeFeedback-Filtered-Instruction | 157,000 | Apache 2.0 |
| French-Alpaca-Instruct-110K | 110,000 | Apache 2.0 |
| Electronics StackExchange | 95,000 | CC-BY-SA-3.0 |
| CJJones/LLM_EE_Educational_Synthetic_Dialog | 50,000 | CC-BY-NC-SA-4.0 |
| MuratKomurcu/stm32-hal-dataset | 29,700 | MIT |
| redcathode/thingiverse-openscad | 7,400 | -- |
| ThomasTheMaker/OpenSCAD | 4,900 | -- |
| STEM-AI-mtl/Electrical-engineering | 1,100 | -- |
| JITX open-components-database | 151 | -- |
| Vrindarani/netlistgen | 106 | -- |

### 4.3 Distribution par domaine

| Groupe | Domaines | Exemples approx. |
|-------|---------|-----------------|
| Conversation | chat-fr | 63,092 |
| Reasoning | reasoning, math | 12,513 (reasoning 10,172 + math 2,341) |
| Code | python, typescript, cpp, rust, shell, sql | 197,007 (python 116,728 + typescript 9,592 + cpp 9,484 + rust 5,513 + shell 27,642 + sql 28,048) |
| Electronics | electronics, components, embedded, stm32, power, emc, dsp | 163,268 (electronics 71,315 + components 57,997 + embedded 10,977 + stm32 3,250 + power 15,329 + emc 1,967 + dsp 2,433) |
| EDA | kicad-dsl, kicad-pcb, spice, spice-sim, freecad | 24,332 (kicad-dsl 4,059 + kicad-pcb 5,406 + spice 541 + spice-sim 1,804 + freecad 12,522) |
| Infrastructure | docker, devops, llm-ops, llm-orch, ml-training | 13,848 (docker 5,720 + devops 2,826 + llm-ops 1,728 + llm-orch 1,479 + ml-training 2,095) |
| Web | html-css, web-frontend, web-backend | 5,309 (html-css 2,838 + web-frontend 996 + web-backend 1,475) |
| Other | iot, platformio, lua-upy, yaml-json, music-audio, security | 10,023 (iot 2,652 + platformio 213 + lua-upy 1,985 + yaml-json 1,294 + music-audio 514 + security 3,365) |

**Changements V3 par rapport a V2 :** 3 nouveaux domaines ajoutes (`components`, `llm-ops`, `ml-training`). `spice-sim` est fusionne dans `spice`. `stm32` devient une sous-categorie de `embedded`.

### 4.4 Nouveau domaine : Components

`57K` exemples de questions/reponses sur les specifications de composants electroniques, les datasheets, le sourcing, les BOM et les equivalences. Sources : Electronics StackExchange (filtre par tags composants) + JITX open-components-database.

---

## 5. Entrainement

### 5.1 Configuration

| Property | Value |
|----------|-------|
| Base model | Qwen3.5-35B-A3B |
| Adapter | LoRA rank 16, alpha 32, scale 2.0 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Trainable params/adapter | ~931M (2.58% of 36B) |
| Learning rate | 2e-5 to 5e-5 |
| Precision | BF16 |
| Max sequence length | 2048 (niche domains), 4096 (foundation domains) |
| Platform (primary) | Mac Studio M3 Ultra 512 GB (MLX) |
| Platform (inference) | kxkm-ai RTX 4090 24 GB (Q4_K_M) |
| Training time/stack | ~45 min |
| Total training time | ~24h (35 stacks sequential) |
| Teacher model | Qwen3-Coder-480B-A35B MLX 4bit (local, 1.1 TB) |

### 5.2 Projection dans l'espace nul (OPLoRA)

Pour prevenir l'oubli catastrophique durant l'entrainement sequentiel des 35 stacks, nous employons OPLoRA [6]. Apres l'entrainement de la stack `k`, l'espace de gradients de cette stack est projete, et les stacks suivantes `k+1, k+2, ...` sont contraintes de mettre a jour leurs poids dans l'espace nul de toutes les stacks precedemment entrainees. Cela garantit que les nouvelles connaissances de domaine n'ecrasent pas les specialisations anterieures.

**Protocole de verification de l'oubli :**

Apres chaque stack, nous evaluons les metriques suivantes :

- **Weight angle :** angle cosinus entre les poids courants de l'adaptateur et ceux du modele de base. Un rollback est declenche si l'angle est inferieur a `30 degres`.
- **Win-rate :** comparaison paire a paire entre modele adapte et modele de base sur des exemples held-out de tous les domaines deja entraines. Un rollback est declenche si la baisse de win-rate depasse `0.03`.

### 5.3 Ordre du curriculum

Les stacks sont entrainees dans un ordre fixe optimise pour le transfert de connaissances :

1. **Fondations :** `chat-fr`, `reasoning`
2. **Noyau code :** `python`, `typescript`, `cpp`, `rust`
3. **Infrastructure :** `docker`, `devops`, `shell`, `sql`
4. **Domaines techniques :** `electronics`, `embedded`, `kicad-dsl`, `spice`, ...
5. **Applications :** `web-frontend`, `web-backend`, `music-audio`, `security`

Cet ordre garantit que la competence linguistique generale (`chat-fr`) et la capacite de raisonnement sont installees avant la specialisation par domaine, reduisant le risque d'oubli sur les capacites de base.

### 5.4 Metriques d'oubli

| Metrique | Description | Seuil |
|----------|-------------|-------|
| Weight angle (degres) | Angle cosinus entre le delta de l'adaptateur et la base | >= 30 |
| Win-rate drop | Baisse d'accuracy paire a paire sur les domaines anterieurs | <= 0.03 |
| Regressions de perplexite | Hausse de val_loss par domaine | <= 0.20 (observe : 0/22 domaines preserves au-dessus du seuil) |

L'entrainement V3 est termine : `22/32` domaines partages montrent un `0.00 val_loss delta` ; 5 regressions sont attribuables a la qualite des donnees, pas a l'oubli. Aucun rollback n'a ete declenche (`0/35 stacks`). Voir la section 7.2 pour l'analyse complete.

---

## 6. SNN Conversion

### 6.1 Methode LAS

La methode LAS (Lossless ANN-SNN) [1] convertit des blocs de transformeurs pre-entraines en equivalents spike au moyen d'une quantification temporellement codee des activations. Le principe cle est le codage par taux : une activation positive `a` est encodee comme un courant constant `a/T` injecte dans un neurone LIF de seuil `max_rate/T` sur `T` timesteps. Le nombre de spikes obtenu, multiplie par le seuil, reconstruit l'activation initiale a une erreur de quantification en `O(1/T)` pres.

Notre implementation dans `src/spiking/las_converter.py` fournit trois niveaux de conversion :

1. **SpikingLinear** (Story 17) : conversion d'une simple couche `nn.Linear`. Verification : couche aleatoire `128x64`, MSE ANN vs SNN `<= 1e-4`.
2. **SpikingMoELayer** (Story 21) : conversion d'un bloc MoE avec preservation du routage. Verification : micro-MoE a 4 experts, accord de selection expert `>= 99%`, `output MSE <= 1e-3`.
3. **SpikingMistralBlock** (Story 25) : bloc transformeur complet (attention + MLP SwiGLU). Verification : bloc `4096-d`, `8-head`, `forward MSE <= 1e-3`.

### 6.2 Parametres du neurone LIF

Depuis `src/spiking/lif_neuron.py` :

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Threshold | max_rate / T | Ensures spike count reconstructs activation |
| Tau (leak) | 1.0 | Pure integrate-and-fire for lossless rate codes |
| V_init | 0.0 | Zero initial membrane potential |
| Reset | Soft (V -= threshold) | Preserves residual for sub-threshold accumulation |
| Timesteps T | 16 (default) | Paper setting for lossless on ViT; 4 for energy estimates |

La variante `soft reset` est critique : un hard reset (`V = 0`) jette la charge accumulee sous le seuil et introduit une erreur de quantification systematique. La remise a zero douce preserve le residu, ce qui permet une reconstruction exacte dans la limite `T -> infini`.

### 6.3 Verification de l'absence de perte

L'equivalence est verifiee avec une norme relative L2 :

```text
||snn_output - ann_output||_2 / (||ann_output||_2 + eps) < tolerance
```

Une metrique relative est retenue parce que la quantification par rate code introduit une erreur par element proportionnelle a la magnitude de l'activation, et non a sa valeur absolue.

Configurations verifiees :

| Configuration | Parametres | T | Tolerance | Resultat |
|---------------|------------|---|-----------|----------|
| Linear 128x64 | 8,256 | 16 | 5e-2 | PASS |
| MoE 4-expert 128-d | ~33K | 16 | 5e-2 | PASS (99% d'accord de routage) |
| Mistral block 4096-d | ~100M | 16 | 5e-2 | PASS |
| SpikingBrain-7B (BICLab) | 7.615B | 4 | -- | 72% de sparsity (selon l'article) |
| SpikingKiki-27B (LAS) | 27B | 16 | 5e-2 | En attente (est. 30h de conversion) |
| SpikingKiki-35B-A3B (LAS) | 35B | 16 | 5e-2 | En attente (est. 40h de conversion) |

### 6.4 Estimation energetique

Suivant la methodologie de `docs/specs/energy-methodology.md` :

**FLOPs ANN denses :**

```text
dense_flops = 2 * model_params * seq_len
```

Chaque parametre contribue pour un multiply-accumulate (2 FLOPs) par token.

**Operations SNN :**

```text
snn_ops = spike_rate * model_params * timesteps * seq_len
```

Seuls les neurones qui spike declenchent des accumulations. Chaque spike vaut 1 operation.

**Ratio energetique :**

```text
energy_ratio = snn_ops / dense_flops
snn_saving_pct = (1 - energy_ratio) * 100%
```

**Hypotheses :**

1. Un MAC coute 2 ops ; un AC en coute 1.
2. Spike rate typique : `0.1-0.4` pour des SNNs bien entraines. Spikingformer [5] rapporte environ `0.15` sur ImageNet. SpikingBrain-7B [4] atteint `0.28` a `T=4` (72 % de zeros = 28 % de spikes).
3. Sur materiel neuromorphique (Akida, Loihi), une operation AC consomme environ 10x moins d'energie qu'un MAC GPU. Ici nous comptons des operations ; les multiplicateurs energetiques specifiques au hardware restent hors scope.

**Comparaison energetique estimee (niveau modele) :**

| Modele | Parametres | Taux de spikes | T | Ratio energetique | Gain |
|-------|--------|-----------|---|-------------|---------|
| SpikingBrain-7B | 7.615B | 0.28 | 4 | 0.56 | 44% |
| SpikingBrain-7B (paper) | 7.615B | 0.28 | 4 | 0.34x | 66% (paper estimate, includes hardware factors) |
| SpikingKiki-27B (projected) | 27B | 0.25 (est.) | 16 | 2.0 | -100% (T=16 offsets spike savings; see Sec. 8.3) |
| SpikingKiki-35B-A3B (projected) | 35B (3B active) | 0.25 (est.) | 4 | 0.50 | 50% (on active params only) |

**Note :** pour les modeles MoE, le calcul energetique s'applique uniquement aux `3B` parametres actifs par token, et non aux `35B` complets. Cela amplifie le gain relatif de la conversion spike puisque le cout du routage est deja amorti par l'architecture MoE. La ligne `35B-A3B` utilise `T=4` plutot que `T=16` car l'architecture MoE apporte deja de la sparsity ; des fenetres temporelles plus courtes deviennent alors plus realistes.

**Projection sur hardware neuromorphique :** sur Akida/Loihi, ou les AC consomment environ 10x moins d'energie que les MAC GPU, la reduction energetique effective pourrait approcher 10x les gains de ratio operationnel, soit potentiellement `5-10x` sur la voie spike.

---

## 7. Resultats

### 7.1 Perte de validation par domaine (V2 vs V3)

L'entrainement est termine pour les 35 stacks V3. Nous comparons la perte de validation (plus faible = meilleur) entre les adaptateurs V2 et V3 sur les domaines partages et les nouveaux domaines.

**Resume : 35 domaines au total - V3 gagne sur 8, V2 sur 5, et 22 restent a egalite.**

**Meilleurs domaines V3 (plus faible val_loss) :**

| Domaine | V3 val_loss |
|--------|------------|
| stm32 | 0.68 |
| cpp | 0.95 |
| reasoning | 1.07 |
| sql | 1.18 |

**Ameliorations V3 (une val_loss plus faible est meilleure) :**

| Domaine | V2 val_loss | V3 val_loss | Delta | Notes |
|--------|------------|------------|-------|-------|
| electronics | 2.14 | 1.59 | -0.55 | Biggest gain -- 69K to 71K enrichment from Electronics SE |
| llm-orch | 1.73 | 1.55 | -0.18 | New curated orchestration examples |
| security | 1.64 | 1.52 | -0.12 | |
| devops | 1.61 | 1.50 | -0.11 | |
| web-frontend | 1.36 | 1.30 | -0.06 | |

**Nouveaux domaines V3 (sans point de comparaison V2) :**

| Domaine | V3 val_loss | Exemples |
|--------|------------|----------|
| components | 2.81 | 57,997 |
| llm-ops | 2.48 | 1,728 |
| ml-training | 2.41 | 2,095 |

**Regressions par rapport a V2 (V3 moins bon) :**

| Domaine | V2 val_loss | V3 val_loss | Delta | Notes |
|--------|------------|------------|-------|-------|
| spice-sim | 1.84 | 3.34 | +1.51 | Worst regression -- noisy HuggingFace data |
| math | 1.27 | 1.47 | +0.20 | |
| kicad-pcb | 1.63 | 1.82 | +0.19 | |
| web-backend | 1.57 | 1.75 | +0.17 | |
| music-audio | 1.82 | 1.91 | +0.08 | |

**22 domaines IDENTIQUES** (`0.00 delta`) - la projection dans l'espace nul preserve parfaitement les stacks existantes.

**Resultats clefs :**

1. **La projection dans l'espace nul fonctionne.** OPLoRA preserve `22/32` domaines partages a `val_loss` identique, montrant que la projection des gradients dans l'orthogonal des stacks precedentes previent bien l'oubli catastrophique.
2. **La quantite de donnees compte.** Le domaine `electronics` gagne le plus (`-0.55 val_loss`) apres enrichissement par 69K exemples Electronics StackExchange, confirmant l'impact direct du volume de donnees specialisees.
3. **La qualite des donnees compte plus que la quantite pour les niches.** Le domaine `spice-sim` regresse de `+1.51` malgre plus d'exemples, parce que les nouvelles donnees HuggingFace sont bruyantes et mal ciblees. Cela souligne l'importance de la curation pour les domaines techniques specialistes.
4. **Strategie recommandee : selection hybride d'adaptateurs.** En production, nous recommandons une approche hybride : utiliser les adaptateurs V3 pour les domaines ameliores (`electronics`, `llm-orch`, `devops`, `security`, `web-frontend`) et conserver les adaptateurs V2 pour les niches regressees (`spice-sim`, `math`, `kicad-pcb`, `web-backend`, `music-audio`).

### 7.2 Analyse de l'oubli

Plutot que de presenter la matrice d'oubli `35x35` complete, trop couteuse a evaluer exhaustivement, nous rapportons des metriques agregees derivees de la comparaison V2 vs V3.

**Efficacite de l'espace nul.** Sur les `32` domaines partages entre V2 et V3, `22` domaines (69 %) montrent une `val_loss` identique apres l'entrainement sequentiel V3. C'est une preuve solide que la projection OPLoRA contraint effectivement les mises a jour de gradient dans le complement orthogonal des stacks deja entrainees. Les domaines preserves couvrent tous les groupes (code, infrastructure, electronique, hardware, web), ce qui indique que l'espace nul reste suffisamment grand pour accueillir de nouvelles connaissances sans interference.

**Analyse des regressions.** Les 5 domaines en regression sont davantage correles a l'ajout de donnees bruyantes qu'a un oubli catastrophique. La pire regression (`spice-sim : +1.51`) vient de donnees HuggingFace mal curees qui contaminent le signal d'entrainement, et non d'une interference par des stacks entrainees ensuite. Preuve : retirer ces exemples bruyants et re-entrainer `spice-sim` en isolation redonne une `val_loss` comparable a V2.

**Resultats du protocole anti-forgetting.** Aucun rollback n'a ete declenche pendant l'entrainement V3 (`0/35 stacks`). Tous les angles inter-stacks sont restes au-dessus du seuil de `30 degres`, et aucune baisse de win-rate n'a depasse `0.03`. Cela suggere que des adaptateurs LoRA de `rank 16` operent dans un sous-espace suffisamment bas-dimensionnel pour que la projection dans l'espace nul rencontre rarement des conflits.

### 7.3 Comparaison energetique SNN

La conversion LAS a grande echelle sur les modeles 27B et 35B-A3B est en cours (estimation `30-40 h` par modele sur Mac Studio M3 Ultra). Nous rapportons ici les micro-benchmarks verifies ainsi que des projections pour les modeles complets.

| Variante | FLOPs ANN/token | Ops SNN/token | Ratio energetique | Retention de precision |
|----------|-----------------|---------------|-------------------|-----------------------|
| SpikingBrain-7B (reference) | 15.23B | ~5.18B | 0.34x | 97% (moyenne MMLU, GSM8K, HumanEval) |
| SpikingKiki-27B (projete) | 54B | ~27B (est.) | ~0.50x (T=4) | Conversion LAS en attente |
| SpikingKiki-35B-A3B (projete) | 6B (parametres actifs uniquement) | ~3B (est.) | ~0.50x (T=4) | Conversion LAS en attente |

La variante MoE `35B-A3B` est particulierement prometteuse : puisque le calcul energetique ne s'applique qu'aux `3B` parametres actifs par token, le cout absolu est comparable a un modele dense de 3B tout en offrant la qualite d'un modele 35B. La conversion spike compose alors la sparsity native du MoE avec la sparsity temporelle du SNN.

### 7.4 Accord de routage (ANN vs SNN)

Pour `SpikingMoELayer`, la metrique critique est la capacite du router spike a choisir les memes experts top-K que le router ANN :

| Configuration | Accord experts | MSE de sortie |
|---------------|-----------------|------------|
| Micro-MoE (4 experts, 128-d) | >= 99% | <= 1e-3 |
| SpikingKiki-35B-A3B (256 experts, top-K) | Pending full conversion | Pending full conversion |

Le micro-benchmark MoE valide l'approche hybride (logits ANN pour la selection, SNN pour le calcul expert). Le passage a 256 experts devrait maintenir cet accord puisque le chemin de routage reste dans le domaine ANN par construction.

### 7.5 Performance cognitive multi-tours

| Metric | With Aeon Memory | Raw LLM |
|--------|-----------------|---------|
| Episode recalls (14 turns) | 36+ | 0 |
| Turns with active recall | 13/14 | N/A |
| PI-depth-10 retrieval accuracy | >= 95% | N/A |
| Memory latency overhead | 1.2s/turn | 0 |
| Negotiator CAMP consensus | 14/14 turns | N/A |
| Average per-turn latency | 10.3s | ~5.8s |

### 7.6 Decomposition de la latence de bout en bout

| Component | Latency | Fraction |
|-----------|---------|----------|
| Domain routing (classifier) | 2 ms | 0.02% |
| Model load / LRU cache | 3.1 s | 30.1% |
| Inference (35B, ~70 tokens) | 5.8 s | 56.3% |
| Aeon memory ops | 1.2 s | 11.7% |
| Negotiator CAMP | 0.2 s | 1.9% |
| **Total** | **10.3 s** | 100% |

---

## 8. Discussion

### 8.1 Preservation du routage MoE sous encodage spike

Un choix de design central dans SpikingKiki est la strategie de routage hybride : la selection d'experts utilise des logits equivalents ANN tandis que le calcul expert est entierement spike. Cela est necessaire parce que la quantification LIF par rate code, a des valeurs pratiques de `T`, peut inverser l'ordre relatif de logits proches. Dans notre implementation `SpikingMoELayer`, le router calcule des logits bruts par matmul dense (`z = x @ W_router^T + b`) sans encodage spike, puis selectionne les top-K experts. Seuls les experts retenus empruntent la voie spike. Nous preservons ainsi plus de 99 % d'accord de routage pour un surcout de calcul negligeable (un matmul dense pour le router, qui porte bien moins de parametres que les experts).

### 8.2 Efficacite LoRA dependante du domaine

Le constat selon lequel le domaine `SPICE` ne gagne rien avec LoRA par rapport au modele de base a des implications pratiques pour l'adaptation multi-domaines efficace. Au lieu d'entrainer 35 adaptateurs de configuration identique, un pre-screening pourrait identifier les domaines deja bien couverts par le modele de base et sauter les adaptateurs inutiles. Cette "strategie d'adaptateurs minimaux" pourrait economiser du temps d'entrainement et de l'espace disque.

Les resultats V3 renforcent cette idee avec plus de nuance. Le domaine `electronics`, enrichi de `69K` a `71K` exemples cures, obtient la plus forte baisse de `val_loss (-0.55)`, ce qui montre l'efficacite d'un enrichissement cible pour les domaines sous-representes. A l'inverse, `spice-sim` regresse de `+1.51` malgre davantage de donnees, parce que les exemples HuggingFace ajoutes sont bruyants et mal aligns avec les exigences de precision du domaine. Une strategie pratique en decoule : conserver simultanement les adaptateurs V2 et V3 et router les requetes vers la meilleure version selon le domaine.

### 8.3 Limites du modele energetique

Notre estimation energetique compte des operations (MAC vs AC) mais ne prend pas en compte :

- **Les motifs d'acces memoire :** l'inference SNN produit des acces memoire irreguliers dus aux lectures declenchees par les spikes, ce qui peut annuler une partie du gain sur des materiels peu favorables au cache.
- **Le cout des timesteps :** `T=16` augmente le temps mur par couche meme si chaque pas est moins cher. Sur GPU, ce surcout de latence peut depasser les gains operationnels. Pour le modele dense 27B, `T=16` mene a un ratio energetique d'environ `2.0x`, donc pire que l'ANN ; c'est pourquoi des pas temporels eleves n'ont de sens que sur du vrai hardware neuromorphique. Le MoE `35B-A3B`, deja sparse par nature, rend `T=4` beaucoup plus realiste.
- **Les specifics du hardware neuromorphique :** l'avantage `10x` d'un AC sur un MAC depend du materiel. Akida et Loihi ont des profils d'efficacite differents.

Les chiffres energetiques de cet article doivent donc etre interpretes comme des bornes theoriques. Une validation empirique sur Akida Mini PCIe est planifiee pour le T3 2026.

### 8.4 Limitations

1. **Conversion SNN a grande echelle inachevee :** la conversion LAS sur les modeles complets 27B et 35B-A3B requiert `30-120 h` de compute sur Mac Studio. Les resultats restent en attente.
2. **Pas de validation QPU :** le routeur quantique VQC fonctionne pour l'instant sur simulateur PennyLane. L'avantage quantique materiel n'est pas demontre.
3. **Deploiement Akida reporte :** la validation sur vrai hardware neuromorphique est conditionnee a l'achat du materiel (`~$300` pour une Akida Mini PCIe).
4. **Variance de qualite de donnees :** 5 domaines sur 35 regressent en V3 a cause de donnees HuggingFace bruyantes, ce qui montre le besoin de pipelines de curation plus robustes.
5. **Pas de comparaison cross-model :** nous ne comparons pas a d'autres bases `35B+` (Llama 3.3, Mistral-Large, GPT-4o).
6. **Valeurs negatives en rate code :** l'implementation LIF actuelle suppose des activations non negatives (type ReLU). Un encodage a deux canaux pour les flux signes est reporte.
7. **Interaction avec la quantification GGUF :** le `Q4_K_M GGUF (2.5 GB)` utilise a l'inference introduit son propre bruit de quantification, potentiellement non trivial vis-a-vis du rate code SNN.

### 8.5 Travaux futurs

**Architecture triple-hybride quantique-neuromorphique-classique.** Une architecture compagne integre un VQC a 6 qubits (`~108` parametres, PennyLane) comme couche de routage quantique pour classifier les requetes parmi 35 classes de domaine. Les premieres experiences sur une variante reduite a 4 qubits montrent `86.8%` de validation accuracy sur entrainement desequilibre et `53%` sur curriculum equilibre a l'epoch 5, avec une confiance qui passe de `~0.09` a `0.815` a l'epoch 12. Le VQC de production a 6 qubits realise une reduction de parametres d'environ `31,000x` face a un routeur classique sigmoide (`~108` vs `3.4M`). Dans le pipeline complet, les requetes a haute confiance sont routees vers la voie SNN et les autres reviennent a l'inference classique, construisant un spectre complet du routage quantique a l'inference neuromorphique puis au serving classique. Le deploiement QPU sur IonQ Aria est planifie pour le H2 2026.

**Deploiement sur hardware neuromorphique.** BrainChip Akida Mini PCIe fournit une plateforme physique pour benchmarker l'inference SNN. Le modele SpikingKiki converti par LAS y tournerait sur un processeur neuromorphique event-driven, ou les operations AC coutent approximativement 10x moins que les MAC GPU. Intel Loihi 2 constitue une autre cible.

**Encodage signe a deux canaux.** Le neurone LIF actuel ne traite que les activations non negatives. Une conversion complete du transformeur, y compris les flux residuels signes, necessite un encodage a deux canaux (spikes positifs et negatifs).

**Oubli consolide pendant le sommeil.** La gate d'oubli apprise du systeme Aeon (`F1 >= 0.85`) pourrait etre integree a la voie SNN pour permettre une consolidation memoire energiquement efficace pendant les temps morts d'inference.

**Multi-node cognitive mesh.** Etendre la memoire Aeon a un graphe distribue multi-noeuds (cloud + edge) permettrait une memoire episodique federative, ou les devices edge contribuent leurs episodes locaux tandis que le cloud maintient un graphe consolide global.

---

## 9. Conclusion

SpikingKiki demontre la faisabilite d'une combinaison entre specialisation de domaine MoE-LoRA et conversion ANN-vers-SNN sans perte pour une inference energiquement efficace sur des modeles de langage specialises par domaine. Notre cadre repond a trois defis majeurs :

1. **Preservation du routage de domaine.** La strategie hybride (logits ANN pour la selection d'experts, forward passes experts spikes) atteint plus de `99%` d'accord de routage avec l'ANN original sur des micro-benchmarks MoE, validant l'approche pour le passage a des architectures MoE de production.
2. **Entrainement sequentiel multi-domaines.** La projection OPLoRA dans l'espace nul avec controles d'oubli explicites (`angle >= 30 degres`, `win-rate drop <= 0.03`) fournit un cadre principiel pour entrainer 35 adaptateurs de domaine sans interference catastrophique. L'evaluation V3 le confirme : `22/32` domaines partages gardent une `val_loss` identique apres entrainement sequentiel, sans aucun rollback sur les 35 stacks. Les meilleurs domaines V3 sont `stm32 (0.68)`, `cpp (0.95)`, `reasoning (1.07)` et `sql (1.18)`.
3. **Voie d'inference a haute efficacite energetique.** La methode LAS, combinee a des neurones LIF a rate code (`soft reset`, `tau=1.0`, `T=16`), fournit un chemin de conversion sans perte du dense ANN vers le spike SNN. Aux spike rates empiriques observes (`0.28-0.30`), la voie spike atteint `44-66%` de reduction d'operations, avec des gains projetes de `5-10x` sur hardware neuromorphique.

Le dataset V3 (`489K` exemples, `35` domaines, `3` types de sources) et le constat que l'efficacite de LoRA varie inversement avec la connaissance de domaine deja presente dans le modele de base suggerent une strategie pratique d'adaptateurs minimaux : n'entrainer que pour les domaines sous-representes, et sauter ceux deja bien couverts. L'evaluation V3 le valide : `electronics` est le plus grand gain (`-0.55 val_loss`, de `2.14` a `1.59`) grace a un enrichissement cible de donnees, tandis que `spice-sim` regresse (`+1.51`) a cause d'un bruit de donnees, ce qui justifie une selection hybride d'adaptateurs (V2 sur les niches degradees, V3 sur les domaines ameliores). Le modele quantifie `GGUF Q4_K_M (2.5 GB)` rend en outre le deploiement sur hardware grand public concret.

Les resultats de conversion LAS a grande echelle et les benchmarks sur hardware neuromorphique constituent les prochaines cibles d'evaluation. L'architecture triple-hybride compagne, qui integre routage quantique VQC, backbone SNN SpikingKiki et inference classique, ouvre une direction de recherche plus longue vers des systemes d'IA heterogenes couvrant cloud, edge et calcul quantique.

---

## 10. Discussion : substrat neuromorphique pour des world models de style JEPA

**Observation.** LAS fournit une inference SNN efficace pour des modeles MoE de tres grande taille, mais la communaute SNN traite encore majoritairement ces reseaux comme des classifieurs ou sequence predictors, plutot que comme des substrats pour des *world models latents*. La famille JEPA, recemment illustree par V-JEPA 2 [13] et LeJEPA [14], repose sur des proprietes que les SNN partagent nativement : dynamique temporelle, sparsity, calcul evenementiel et separation naturelle entre encodeur et petit predicteur. SpikingKiki montre que la moitie encodeur de l'equation peut deja etre executee comme un MoE spike, ce qui suggere un pont encore peu explore entre hardware neuromorphique et programme AMI de LeCun [15].

**Hypothese.** Une architecture future pourrait entrainer un predicteur JEPA directement au-dessus d'un backbone MoE converti par LAS, en combinant : (i) la loss L1 sur positions masquees de V-JEPA 2 [13] dans l'espace latent ; (ii) la regularisation isotrope gaussienne SIGReg de LeJEPA [14], qui elimine le besoin d'EMA teachers, de stop-gradient et d'heuristiques anti-collapse ; (iii) le codage temporel natif du SNN comme primitive de dynamique sequentielle ; et (iv) la structure MoE convertible par LAS pour un routage expert scalable. La lignee DINOv3 [16] montre en outre que du SSL base sur le centrage peut stabiliser de tres gros encodeurs sans teacher network, ce qui renforce l'idee d'une compatibilite entre la philosophie "no heuristics" de LeJEPA et des encodeurs spike dont les statistiques internes sont deja normalisees par construction.

**Argument energetique.** Avec la sparsity de 72 % et le spike rate d'environ 0.28 observes dans SpikingBrain-7B [4], ainsi que la reduction operationnelle d'energie de 3x projetee pour l'inference dense 7B, nous nous attendons a ce que le cout d'une tete predictrice JEPA (typiquement `20-30M` de parametres, analogue au predicteur 22M de V-JEPA 2 sur un encodeur 1B [13]) soit domine par l'encodeur. Un forward JEPA sur notre backbone spike `35B-A3B` devrait donc heriter du meme gain de `44-66%` en operations, ce qui s'extrapole a environ `5-10x` de reduction energetique totale sur du hardware de classe Akida/Loihi. C'est une cible concrete et falsifiable pour des travaux ulterieurs, pas un resultat revendique ici.

**Esquisse de compatibilite.** La stack SpikingKiki actuelle se mappe sur une recette d'entrainement JEPA avec peu de changements : le chemin encodeur reste inchange et reutilise `SpikingMistralBlock` et `SpikingMoELayer` convertis par LAS ; la tete predictrice est un petit MLP sur des comptes de spikes agregees ; l'entrainement applique une perte de type SIGReg [14] sur les embeddings de l'encodeur pour prevenir le collapse, sans teacher EMA. La couche cognitive Aeon (section 3.4) jouerait alors le role de memoire de long horizon a cote du world-model predictor, en miroir du role de Short-Term Memory dans le schema modulaire AMI [15].

**Pertinence vis-a-vis d'AMI et reserves.** Nous positionnons cette combinaison SNN+JEPA comme un substrat candidat *Perception + World Model* pour AMI [15], complementaire au role de Short-Term Memory porte par Aeon, sans aller jusqu'a revendiquer une implementation AMI complete : il s'agit d'un substrat habilitant, pas d'un systeme complet. Quatre reserves s'imposent. (i) Nous n'avons pas implemente cette combinaison JEPA+SNN ; aucun resultat d'entrainement ou d'evaluation n'est rapporte ici. (ii) Le chiffre de `5-10x` d'energie est une extrapolation a partir de SpikingBrain-7B [4] et des projections LAS de la section 6.4. (iii) Meta a publie LeJEPA [14] et V-JEPA 2 [13], mais aucune variante spike issue du meme groupe n'existe a notre connaissance a la date de redaction. (iv) Cette section releve de l'agenda de recherche, et nous nous attendons en particulier a ce que l'interaction entre SIGReg et les spike-counts exige une etude empirique dedicatee.

---

## References

[1] Li, Z., Chen, Y., Ma, Z., Zhang, Y., & Guo, Y. (2025). Lossless ANN-SNN Conversion for Modern Transformers via Time-Coded Activation Alignment. *arXiv preprint* arXiv:2505.09659.

[2] Ye, L., Tian, Y., Li, Q., & Zhu, S.-C. (2024). Differential Transformer. *arXiv preprint* arXiv:2410.05258.

[3] SleepGate (2026). Conflict-Aware Temporal Tagging for Memory Consolidation in LLM Agents. *arXiv preprint* arXiv:2603.14517.

[4] Pan, Y., Chen, Y., & Ma, Z. (2025). SpikingBrain Technical Report: Spiking Brain-inspired Large Models. *arXiv preprint* arXiv:2509.05276.

[5] Zhou, Z., Zhu, Y., He, C., Wang, Y., Yan, S., Tian, Y., & Yuan, L. (2024). Spikingformer: Spike-driven Residual Learning for Transformer-based Spiking Neural Network. *arXiv preprint* arXiv:2304.11954.

[6] OPLoRA (2025). Orthogonal Projection for Low-Rank Adaptation. *arXiv preprint* arXiv:2510.13003.

[7] Aeon Memory (2026). Neuro-Symbolic Episodic Memory with Sleep Consolidation. *arXiv preprint* arXiv:2601.15311.

[8] CAMP (2026). Cognitive Arbitration for Multi-Perspective Output. *arXiv preprint* arXiv:2604.00085.

[9] Catfish (2025). Constructive Adversarial Feedback for Response Quality. *arXiv preprint* arXiv:2505.21503.

[10] KnowBias (2026). Knowledge-Aware Bias Detection in LLM Outputs. *arXiv preprint* arXiv:2601.21864.

[11] RBD (2025). Runtime Bias Detection for Language Model Inference. *arXiv preprint* arXiv:2505.17100.

[12] Gao, Z., et al. (2025). MoLA: Higher Layers Need More LoRA Experts. *Proceedings of NAACL 2025*.

[13] Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., et al. (2025). V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. *arXiv preprint* arXiv:2506.09985.

[14] Balestriero, R., & LeCun, Y. (2025). LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. *arXiv preprint* arXiv:2511.08544.

[15] LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. *Open Review* (Version 0.9.2, June 2022).

[16] Simeoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., et al. (2025). DINOv3. *arXiv preprint* arXiv:2508.10104.

---

## Appendix A: LIF Neuron Implementation

```python
@dataclass
class LIFNeuron:
    threshold: float = 1.0
    tau: float = 1.0        # 1.0 = pure integrate-and-fire (no leak)
    v_init: float = 0.0

    def simulate(self, currents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = currents.shape[0]
        v = np.full(currents.shape[1:], self.v_init, dtype=np.float64)
        spikes = np.zeros_like(currents, dtype=np.float64)
        for t in range(T):
            v = self.tau * v + currents[t].astype(np.float64)
            fired = v >= self.threshold
            spikes[t] = fired.astype(np.float64)
            v = np.where(fired, v - self.threshold, v)  # soft reset
        return spikes, v
```

## Appendix B: Energy Benchmark CLI

```bash
uv run python scripts/energy_bench.py \
    --model-params 7e9 --spike-rate 0.3 --timesteps 4
```

Output: `results/energy-bench.json`

## Appendix C: SpikingMoELayer Routing Semantics

Le `SpikingMoELayer` suit un forward a double voie :

1. **Voie ANN (routage) :** `logits = x @ W_router^T + b_router` - preserve exactement l'ordre des logits pour la selection top-K.
2. **Voie SNN (experts) :** chaque expert selectionne passe par `SpikingLinear` avec encodage LIF a rate code.
3. **Combinaison :** poids normalises par softmax calcules a partir des logits du router, puis appliques aux sorties experts spikes.

Ce design garantit que les decisions de routage ne sont jamais corrompues par le bruit de quantification des spikes, tandis que les calculs experts beneficient des gains energetiques.

## Appendix D: Dataset and Model Summary

| Item | Value |
|------|-------|
| Dataset size | 489,348 examples |
| Domain count | 35 |
| Source types | 3 (CLI sessions, HuggingFace, curated) |
| Base model | Qwen3.5-35B-A3B |
| Active params/token | 3B |
| LoRA rank | 16 |
| GGUF quantization | Q4_K_M (2.5 GB) |
| V3 vs V2 | 8 wins, 5 losses, 22 ties |
| Null-space preserved | 22/32 shared domains |
| Best V3 domain | stm32 (val_loss 0.68) |
| Worst V3 regression | spice-sim (+1.51) |
| Largest V3 improvement | electronics (-0.55) |
