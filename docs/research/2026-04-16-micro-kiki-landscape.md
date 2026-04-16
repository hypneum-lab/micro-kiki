# Rapport de recherche : paysage technologique de micro-kiki

> Date : 2026-04-16
> Projet : micro-kiki (electron-rare/micro-kiki)
> Contexte : systeme multi-experts LoRA sur Qwen3.5-35B-A3B, entrainement MLX sur Mac Studio M3 Ultra 512 Go, pipeline de conversion SNN

---

## Table des matieres

1. [Etat de l'art MoE-LoRA](#1-etat-de-lart-moe-lora)
2. [Fine-tuning MLX sur Apple Silicon pour modeles 35B+](#2-fine-tuning-mlx-sur-apple-silicon-pour-modeles-35b)
3. [Routage domain-specific dans les systemes multi-adaptateurs](#3-routage-domain-specific-dans-les-systemes-multi-adaptateurs)
4. [Reseaux de neurones a impulsions (SNN) appliques aux LLM](#4-reseaux-de-neurones-a-impulsions-snn-appliques-aux-llm)
5. [Projets comparables : fine-tuning Qwen avec LoRA pour des domaines specifiques](#5-projets-comparables-fine-tuning-qwen-avec-lora-pour-des-domaines-specifiques)
6. [Implications pour micro-kiki](#6-implications-pour-micro-kiki)

---

## 1. Etat de l'art MoE-LoRA

### 1.1 Definition et motivation

Le MoE-LoRA (Mixture of Experts with LoRA) combine l'architecture Mixture of Experts avec les adaptateurs Low-Rank Adaptation. Au lieu de fine-tuner un seul adaptateur LoRA monolithique, on insere plusieurs modules LoRA comme "experts" dans les couches du transformeur, avec un routeur qui selectionne dynamiquement les experts pertinents par token ou par sequence.

**Motivation principale** : le fine-tuning LoRA standard atteint un plafond de performance en multi-tache. Les experts MoE-LoRA permettent une specialisation par domaine tout en partageant la base gelee, avec un cout parametrique marginal.

### 1.2 Papers fondateurs et variantes recentes

| Paper | Date | Contribution cle | Lien |
|-------|------|-------------------|------|
| **MixLoRA** (Li et al.) | Avr 2024 | Insere des experts LoRA dans le FFN + routeur top-k. +9% de precision vs PEFT SOTA en multi-tache. Framework haute performance reduisant la memoire GPU de 40% | [arXiv:2404.15159](https://arxiv.org/abs/2404.15159) |
| **MoLoRA** (Shah & Wagle, Microsoft) | Mars 2026 | Routage **par token** (pas par sequence). Qwen3-1.7B + MoLoRA depasse Qwen3-8B sur 4 benchmarks de raisonnement. Specialisation composable a l'inference | [arXiv:2603.15965](https://arxiv.org/abs/2603.15965) |
| **ReMix** (Qiu et al., Meta/UIUC) | Mars 2026 | Routage par **renforcement** (RLOO). Poids de routage non-apprenables pour eviter le "routing weight collapse". +3.19 pts vs SOTA | [arXiv:2603.10160](https://arxiv.org/abs/2603.10160) |
| **Brainstacks** (Abu Ayyash) | Avr 2026 | Stacks MoE-LoRA **geles** pour apprentissage continu. Routeur meta-sigmoid outcome-based. Null-space projection via SVD randomise = zero oubli. Decouverte cle : les stacks encodent des primitives cognitives transferables, pas du savoir domaine-specifique | [arXiv:2604.01152](https://arxiv.org/abs/2604.01152) |
| **HotMoE** (Huang et al.) | 2025 (AAAI) | Routage **hybride** hierarchique : attention expert-expert en couches basses, attention token-expert en couches hautes | [GitHub:Starlight039/HotMoE](https://github.com/Starlight039/HotMoE) |
| **LD-MoLE** (ICLR 2026) | 2026 | Routage **dynamique appris** avec Sparsegen, allocation adaptive par couche et par token | [arXiv:2509.25684](https://arxiv.org/abs/2509.25684) |
| **LoRA-Switch** (Kong et al.) | Mai 2024 | Co-design systeme/algorithme pour routage dynamique token-wise. Kernel CUDA fusionne pour reduire la latence de decodage de 2.4x | [arXiv:2405.17741](https://arxiv.org/abs/2405.17741) |
| **TT-LoRA MoE** (Kunwar et al.) | Avr 2025 | Combine Tensor-Train LoRA + MoE sparse. 2% des parametres LoRA, 0.03% des parametres de fusion. Surpasse AdapterFusion de 4 pts | [arXiv:2504.21190](https://arxiv.org/abs/2504.21190) |

### 1.3 Frameworks et implementations

| Framework | Description | Lien |
|-----------|-------------|------|
| **MoE-PEFT / MixLoRA** | Framework de reference pour MixLoRA, support top-k/top-p/switch routing | [GitHub:TUDB-Labs/MixLoRA](https://github.com/TUDB-Labs/MixLoRA) |
| **mlx-tune** | Fine-tuning MoE natif sur Apple Silicon. Detecte les couches MoE et applique LoRA per-expert via `LoRASwitchLinear`. 39+ architectures dont Qwen3.5-35B-A3B | [GitHub:ARahim3/mlx-tune](https://github.com/ARahim3/mlx-tune) |
| **Unsloth** | Support Qwen3 MoE fine-tuning. Qwen3-30B-A3B fonctionne avec 17.5 Go VRAM. Mise a jour 2026 "Faster MOE" | [unsloth.ai](https://unsloth.ai/docs/fr/modeles/tutorials/qwen3-how-to-run-and-fine-tune) |

### 1.4 Taxonomie des strategies de routage

```
Routage MoE-LoRA
 |
 +-- Par sequence (classique)
 |     +-- LoRAHub: fusion gradient-free a l'inference
 |     +-- LORAUTER: routage via task embeddings
 |
 +-- Par token
 |     +-- MoLoRA: gating appris par token
 |     +-- LoRA-Switch: token-wise merge dans backbone
 |     +-- ReMix: routage par renforcement (RLOO)
 |     +-- LD-MoLE: Sparsegen dynamique
 |
 +-- Hybride
 |     +-- HotMoE: expert-expert (couches basses) + token-expert (couches hautes)
 |     +-- HiLoRA: hierarchique, training-free
 |
 +-- Par stack/domaine
       +-- Brainstacks: meta-routeur sigmoid, primitives cognitives
```

### 1.5 Pertinence pour micro-kiki

Le pivot v0.2 de micro-kiki (Qwen3.5-35B-A3B natif MoE + LoRA standard) est **valide** par l'etat de l'art :

- Le modele de base est **deja** MoE (256 experts, 3B actifs/token). Ajouter un MoE-LoRA par-dessus serait redondant -- c'est exactement ce que les auteurs de MoLoRA et Brainstacks recommandent d'eviter sur des bases MoE.
- Le routeur 11-output de micro-kiki (10 domaines niche + 1 passthrough) opere au **niveau sequence**, ce qui est adapte a un systeme ou chaque requete releve typiquement d'un seul domaine hardware/EDA.
- La decouverte de Brainstacks (les stacks encodent des primitives cognitives, pas du savoir domaine) valide l'approche de micro-kiki de laisser les 22 domaines generaux en passthrough : la base Qwen3.5-35B-A3B maitrise deja ces primitives.

---

## 2. Fine-tuning MLX sur Apple Silicon pour modeles 35B+

### 2.1 Etat des lieux MLX (avril 2026)

MLX est le framework ML d'Apple optimise pour l'architecture memoire unifiee des puces Apple Silicon. Le fine-tuning LoRA via `mlx-lm` et `mlx-tune` est la methode de reference pour l'entrainement local.

**Version courante** : MLX 0.30+, mlx-lm (ml-explore), mlx-tune v0.4.22 (ARahim3).

### 2.2 Memoire et limites pratiques

| Configuration | Memoire typique | Notes |
|--------------|-----------------|-------|
| Qwen3.5-35B-A3B BF16 LoRA (micro-kiki) | ~195 Go peak | Confirme par le README micro-kiki. Metal GPU a 100% |
| 20B BF16 LoRA (M3 Ultra 512 Go) | ~487 Go peak | Rapport Petie Clark, 14h pour 3 epochs |
| 8B BF16 LoRA sans grad checkpoint | ~134 Go (seq 8192) | Mesure par angeloskath (Apple) |
| 8B BF16 LoRA avec grad checkpoint | ~53 Go (seq 8192, batch 1) | Meme source. Decouple la memoire du nombre de couches LoRA |
| 30B MoE 8-bit inference (MLX server) | ~83 Go sur 96 Go | Crash kernel documente (issue #883) |

**Regle empirique** : garder les poids du modele sous 60% de la RAM unifiee totale, pour laisser de la marge au KV-cache, a l'OS et au runtime d'inference.

### 2.3 Problemes connus et mitigations

1. **Fuite memoire avec gradient checkpointing** : un bug confirme dans MLX 0.19.x causait une croissance continue de la memoire. Corrige dans le PR #1548 (awni, oct 2024). Verifier qu'on utilise MLX >= 0.20.

2. **Kernel panics sur modeles MoE 30B+** : le serveur `mlx_lm.server` sans `--max-kv-size` provoque des kernel panics quand le KV-cache grandit sans borne. Documente dans l'issue #883 (BrunoCerberus, fev 2026). 9 panics en 6 jours sur Mac Studio M3 Ultra 96 Go avec un modele 122B MoE. **Mitigation** : toujours fixer `--max-kv-size` et `mx.set_wired_limit()`.

3. **Memoire croissante pendant le training** : issue #828. Un hack recommande :
   ```python
   wired_limit = int(mx.metal.device_info()["max_recommended_working_set_size"] * 0.9)
   mx.set_wired_limit(wired_limit)
   mx.set_memory_limit(wired_limit)
   ```

4. **GGUF export depuis modeles quantises** : l'export GGUF ne fonctionne pas directement depuis les modeles 4-bit. Il faut d'abord fusionner les adaptateurs puis re-quantifier. Limitation connue de mlx-tune.

### 2.4 Bonnes pratiques pour 35B MoE

- **Gradient checkpointing obligatoire** : `--grad-checkpoint`. Augmente le temps de ~30% mais decouple la memoire du nombre de couches LoRA.
- **Batch size 1-2 maximum** pour les modeles 35B en BF16.
- **Sequence length progressive** : le curriculum 512 -> 1280 -> 4096 de micro-kiki est la bonne approche. Les longues sequences multiplient la memoire.
- **Ne pas LoRA-tuner les couches FFN MoE** : uniquement les projections d'attention (q/k/v/o). Le routage MoE natif est deja appris.
- **`UNSLOTH_COMPILE_DISABLE=1`** : necessaire pour les modeles MoE en precision mixte.
- **Monitorer la pression memoire** : utiliser `asitop` et surveiller que la pression reste verte. Jaune/rouge = swap actif = performance degradee.

### 2.5 Outils

| Outil | Role | Lien |
|-------|------|------|
| **mlx-lm** | Fine-tuning LoRA officiel Apple | [GitHub:ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) |
| **mlx-tune** | Fine-tuning multi-modalite, MoE natif, API Unsloth-compatible | [GitHub:ARahim3/mlx-tune](https://github.com/ARahim3/mlx-tune) |
| **MLX Training Studio** | GUI macOS pour configurer et gerer les jobs LoRA | [GitHub:stevenatkin/mlx-lm-gui](https://github.com/stevenatkin/mlx-lm-gui) |

### 2.6 Pertinence pour micro-kiki

La configuration de micro-kiki (Qwen3.5-35B-A3B, BF16 LoRA, rank 16, ~195 Go peak sur M3 Ultra 512 Go) est validee par les retours communautaires. Les 512 Go de RAM unifiee donnent ~317 Go de marge apres le pic, ce qui est confortable. Le risque principal est le kernel panic pendant l'inference longue du teacher 480B -- fixer `--max-kv-size` est critique.

---

## 3. Routage domain-specific dans les systemes multi-adaptateurs

### 3.1 Approches par classification

La methode la plus directe : un classifieur entraine pour determiner le domaine de la requete, qui selectionne ensuite le(s) adaptateur(s) LoRA correspondant(s).

| Approche | Description | Reference |
|----------|-------------|-----------|
| **Classifieur a N sorties** | MLP ou transformer leger entraine sur des exemples etiquetes par domaine. C'est l'approche de micro-kiki (11 sorties, seuil 0.65) | Architecture micro-kiki |
| **LORAUTER** | Routage via **task representations** derivees de petits jeux de validation. Scale avec le nombre de taches, pas le nombre d'adaptateurs. +5.2 pts sur taches inconnues | [arXiv:2601.21795](https://arxiv.org/abs/2601.21795) |
| **LoRAHub** | Composition gradient-free a l'inference. Poids de fusion optimises par CMA-ES sur ~5 exemples. Seulement 5 Go de memoire (inference-only) | [arXiv:2307.13269](https://arxiv.org/abs/2307.13269) |

### 3.2 Approches par routage appris (gating)

| Approche | Granularite | Description |
|----------|------------|-------------|
| **MoLoRA** | Token | Gating appris, specialisation composable. Routeur MLP 2 couches |
| **LD-MoLE** | Token + couche | Sparsegen differentiable, allocation adaptative |
| **ReMix** | Token | Poids non-apprenables + RLOO. Evite le routing weight collapse |
| **HiLoRA** | Hierarchique | Training-free, selection par vraisemblance gaussienne |

### 3.3 Approches par composition/fusion

| Approche | Description |
|----------|-------------|
| **Brainstacks meta-router** | Routeur sigmoid outcome-based. Decouvre empiriquement quelles combinaisons de stacks sont optimales. Routage a N domaines, pas 1-of-N |
| **CLoRA** | Composition contrastive de LoRA multiples |
| **LoRA-Mixer** | Routage par attention serielle entre experts |

### 3.4 Routage multi-tier (pertinent pour micro-kiki)

L'architecture de micro-kiki utilise un routage a **plusieurs niveaux** :
1. **Dispatcher** (7 meta-intents, training-free, zero latence)
2. **Routeur domaine** (classifieur 11 sorties)
3. **Routage de confiance** : >= 0.65 -> base + stack(s) niche ; < 0.65 -> passthrough base seule
4. **Routage multi-modele** : 35B+LoRA / 35B passthrough / 480B teacher / devstral-v3

C'est une approche **sequence-level** qui est pertinente pour des domaines bien delimites (KiCad, SPICE, EMC...). Le routage token-level de MoLoRA serait pertinent si les requetes melangeaient regulierement plusieurs domaines dans un seul prompt (ex: "ecris du code SPICE pour ce schema KiCad"), ce qui arrive dans ~10-15% des cas hardware.

### 3.5 Recommandation

Pour les prochaines iterations, considerer un **routage hybride** :
- Routage sequence-level pour la selection du domaine principal (rapide, 0 latence)
- Routage token-level optionnel pour les requetes cross-domain (activer 2-3 stacks simultanement via un gating leger)
- Cela s'aligne avec la contrainte micro-kiki de max 4 stacks actifs

---

## 4. Reseaux de neurones a impulsions (SNN) appliques aux LLM

### 4.1 Contexte

Les SNN (Spiking Neural Networks) sont des reseaux bio-inspires ou l'information est encodee sous forme d'impulsions binaires (spikes) au lieu de valeurs continues. Leur interet principal pour les LLM est la **reduction drastique de la consommation energetique** (jusqu'a 85% selon les benchmarks), grace au calcul evenementiel et creux.

### 4.2 Papers cles sur SNN + LLM

| Paper | Date | Contribution | Echelle | Reduction energie |
|-------|------|-------------|---------|-------------------|
| **SpikeGPT** (Zhu et al.) | 2023 (TMLR 2024) | Premier LLM generatif SNN. Architecture RWKV modifiee avec activations binaires | 216M params | 32.2x theorique sur hardware neuromorphe |
| **NeuTransformer** (Balaji et al.) | Sep 2025 | Conversion ANN-vers-SNN par fine-tuning supervise. SSA (Spike-based Self-Attention) remplace le self-attention classique | GPT-2 Large (774M) | 64-85% reduction estimee du bloc attention |
| **LAS** (Chen et al., AAAI 2026) | Mai 2025 | Conversion **loss-less** ANN-vers-SNN. Neurone OAT (Outlier-Aware Threshold) + neurone HG hierarchique. Premier a convertir OPT-66B sans perte | OPT-66B + vision-language | Quasi zero-loss sur 6 LLM et 2 VLM. +2% sur WSC pour OPT-66B |
| **Dual ANN-to-SNN** (anonyme, ICLR 2026 review) | 2025 | Framework dual de conversion, calibration couche par couche. Performance comparable a la quantification SOTA sur LLaMA | LLaMA (7B-13B) | Comparable a la quantification INT4 |
| **WD-SpikingFormer** (anonyme, ICLR 2026 review) | 2025 | Spiking Transformer decoder-only avec mecanisme Winner-Take-All. Softmax-free, entierement spike-driven | 16 datasets NLU | Premier spiking transformer decoder-only pleinement spike-driven |
| **SFormer** (anonyme, ICLR 2026 review) | 2025 | Scale-and-Fire Neuron pour conversion single-timestep. 88.8% ImageNet-1K a T=1 | ViT-Base/Large | 5% reduction vs SOTA a T=2 |
| **Spikingformer** (Zhou et al., AAAI 2026) | Nov 2025 | Modele fondateur pour SNN. Spike-driven residual learning. 75.85% ImageNet-1K | 66M params | 57% reduction vs Spikformer |
| **Xpikeformer** (Song et al.) | Avr 2025 | Premier accelerateur hardware hybride analogique-digital pour spiking transformers. 13x reduction energetique vs SOTA digital ANN | Vision tasks | 13x vs digital ANN, 1.9x vs ASIC SNN optimal |

### 4.3 Approches de conversion ANN-vers-SNN

Trois paradigmes coexistent :

1. **Entrainement direct (DT)** : SpikeGPT, SpikeBERT. Entrainement from scratch avec gradients surrogate. Limite aux petits modeles (<250M).

2. **Conversion ANN-vers-SNN** : NeuTransformer, LAS, Dual conversion. Convertit un modele pre-entraine en SNN. Plus scalable mais necessite T timesteps (latence accrue). LAS a reduit cela a 16 timesteps pour OPT-66B.

3. **Hybride** : Fine-tuning post-conversion (NeuTransformer). Convertit le FFN, fine-tune le SSA. Meilleur compromis precision/scalabilite.

### 4.4 SNN pour le routage (approche micro-kiki)

L'approche SNN de micro-kiki (branche `neuroscience`, v0.3) est **avant-gardiste** et ne correspond a aucun systeme existant dans la litterature. Les SNN sont utilises dans la litterature pour :

- **L'inference complete** du LLM (SpikeGPT, LAS) -- remplacement total du transformer
- **L'acceleration hardware** (Xpikeformer, Loihi) -- deploiement neuromorphe

Mais **aucune publication** n'utilise les SNN specifiquement pour le **routage** de requetes entre adaptateurs LoRA. L'idee de micro-kiki d'un routeur bio-inspire (spiking) pour la selection d'experts est un angle **original et inexplore**.

Le pipeline LAS (Loss-less ANN-to-SNN) cite dans le README micro-kiki est le SOTA actuel pour la conversion. Les variantes SpikingKiki-27B/35B/122B visees par micro-kiki seraient les premiers modeles MoE convertis en SNN -- un territoire completement vierge.

### 4.5 Hardware neuromorphe pertinent

| Hardware | Type | Disponibilite | Notes |
|----------|------|---------------|-------|
| **BrainChip Akida** | PCIe Mini (~300$) | Commercial | Cible dans BRANCH-neuroscience |
| **Intel Loihi 2** | Recherche | Simulateur (KAPOHO) | Supporte par le Intel Neuromorphic Research Community |
| **IBM NorthPole** | Recherche | Non public | 256 coeurs, 22nm, optimise pour inference |

### 4.6 Pertinence pour micro-kiki

- **Pipeline LAS** : la reference pour la conversion. Code disponible sur [GitHub:lc783/LAS](https://github.com/lc783/LAS). Applicable directement a Qwen3.5-35B-A3B.
- **Routeur SNN** : angle original. Aucune competition directe. Risque eleve mais potentiel de publication.
- **Hardware** : Akida Mini PCIe est le chemin le plus court vers un deploiement reel. Le simulateur suffit pour la validation.

---

## 5. Projets comparables : fine-tuning Qwen avec LoRA pour des domaines specifiques

### 5.1 Fine-tuning Qwen pour le francais

| Projet | Base | Technique | Domaine | Lien |
|--------|------|-----------|---------|------|
| **Qwen2.5-0.5B-DPO-French-Orca** (BounharAbdelaziz) | Qwen2.5-0.5B-Instruct | DPO sur french_orca_dpo_pairs | Assistant conversationnel FR | [HuggingFace](https://huggingface.co/BounharAbdelaziz/Qwen2.5-0.5B-DPO-French-Orca) |
| **Qwen2.5-0.5B-Instruct-French-LoRa** (Volko76) | Qwen2.5-0.5B-Instruct | LoRA via Unsloth | General FR | [HuggingFace](https://huggingface.co/Volko76/Qwen2.5-0.5B-Instruct-French-LoRa) |
| **French summarization pipeline** (Mercier & Melliti) | Qwen2.5-32B-Instruct (gen) + Qwen-0.5B (tune) | LoRA QLoRA r=8-128 | Resumation FR | [GitHub](https://github.com/gabriel-mercier/Small-Langage-Model-Finetuning-for-Summarization) |

### 5.2 Fine-tuning Qwen pour des domaines techniques

| Projet | Base | Technique | Domaine | Lien |
|--------|------|-----------|---------|------|
| **CyberSec-Assistant-3B** (AYI-NEDJIMI) | Qwen2.5-3B | QLoRA r=64 alpha=128 | Cybersecurite (MITRE, ISO 27001, NIST) | [HuggingFace](https://huggingface.co/datasets/AYI-NEDJIMI/article-fine-tuning-llm-cybersecurite-qlora) |
| **Qwen3-finetune benchmark** (ceresnam) | Qwen3 4B/8B/14B | QLoRA | Medical reasoning (medical-o1) | [GitHub](https://github.com/ceresnam/qwen3-finetune) |
| **Qwen3.5 domain fine-tuning** (AI News Grid) | Qwen3.5 | LoRA r=16-32 | Guide general | [ainewsgrid.com](https://ainewsgrid.com/blog/how-to-fine-tune-qwen-35-on-your-own-data) |

### 5.3 Guides et references officielles

| Ressource | Description | Lien |
|-----------|-------------|------|
| **Qwen official LoRA docs** | Config LoRA, cibles q/k/v/o + w1/w2, memoire par taille | [Qwen docs](https://www.mintlify.com/QwenLM/Qwen/finetuning/lora) |
| **Unsloth Qwen3 MoE** | Support natif Qwen3-30B-A3B et 235B-A22B. 17.5 Go VRAM pour 30B-A3B | [unsloth.ai](https://unsloth.ai/docs/fr/modeles/tutorials/qwen3-how-to-run-and-fine-tune) |
| **Guide FR complet LoRA/QLoRA** (AYI-NEDJIMI) | Guide francophone detaille, benchmarks, comparatifs | [ayinedjimi-consultants.fr](https://ayinedjimi-consultants.fr/ia-fine-tuning-llm-lora-qlora.html) |
| **SOTAAZ Qwen3.5 LoRA guide** | Data prep -> training -> eval -> deploy | [sotaaz.com](https://sotaaz.com/post/qwen35-finetuning-en) |

### 5.4 Analyse comparative avec micro-kiki

**Aucun projet existant** ne combine les elements suivants simultanement :
- Modele MoE natif (Qwen3.5-35B-A3B, 256 experts)
- Multi-adaptateurs LoRA domaine-specifiques (10 stacks hardware/EDA)
- Routeur multi-tier avec seuil de confiance
- Couche cognitive (memoire, negotiation, anti-biais)
- Pipeline de conversion SNN
- Entrainement full BF16 sur Apple Silicon 512 Go

Les projets comparables restent sur des echelles beaucoup plus modestes (0.5B-3B) et un seul domaine a la fois. Le projet **Brainstacks** est le plus proche architecturalement (stacks MoE-LoRA geles, routeur outcome-based, apprentissage continu) mais vise des domaines generaux (chat, code, math, medical) sur des modeles plus petits (TinyLlama 1.1B, Gemma 12B).

---

## 6. Implications pour micro-kiki

### 6.1 Validations de l'architecture actuelle

L'etat de l'art confirme les choix de conception de micro-kiki v0.2 :

1. **Base MoE native + LoRA standard** (pas MoE-LoRA custom) : correct. MoLoRA et Brainstacks montrent que l'ajout d'un MoE-LoRA sur une base deja MoE est redondant.

2. **LoRA cible sur q/k/v/o uniquement** (pas les couches FFN/MoE) : valide par les recommandations Qwen et l'experience communautaire.

3. **Routeur sequence-level a seuil de confiance** : adapte aux domaines bien delimites de micro-kiki. Le routage token-level n'est necessaire que pour les requetes cross-domain.

4. **Entrainement sequentiel avec forgetting check** : valide par Brainstacks (null-space projection) et le consensus general. Micro-kiki pourrait beneficier de l'OPLoRA deja adopte (init projection pour prevention de l'oubli).

### 6.2 Opportunites d'amelioration

1. **Routage hybride sequence/token** : pour les 10-15% de requetes cross-domain, un gating leger au niveau token permettrait d'activer 2-3 stacks simultanement (dans la limite des 4 max).

2. **Meta-routeur outcome-based** (a la Brainstacks) : au lieu d'un classifieur entraine sur des labels domaine, entrainer le routeur sur les **resultats empiriques** (quelle combinaison de stacks produit la meilleure reponse). Decouvrirait potentiellement des synergies inattendues entre stacks.

3. **LORAUTER pour les stacks futurs** : utiliser des task representations au lieu de features de la requete pour le routage. Scale mieux avec le nombre de stacks.

4. **Conversion SNN (LAS)** : le pipeline LAS est mature pour les modeles denses jusqu'a OPT-66B. Son application a un modele MoE natif comme Qwen3.5-35B-A3B est **inexploree** et represente une contribution potentielle originale. Risque : les 256 experts MoE ajoutent une complexite significative a la conversion des activations.

5. **mlx-tune pour MoE** : considerer mlx-tune (ARahim3) comme alternative a la pipeline KIKI-Mac_tunner actuelle. Support natif MoE avec `LoRASwitchLinear`, API Unsloth-compatible.

### 6.3 Risques identifies

| Risque | Impact | Mitigation |
|--------|--------|------------|
| Kernel panic MLX avec modele 35B MoE + KV-cache long | Perte de session de training | Fixer `--max-kv-size`, limiter la wired memory a 90% |
| Fuite memoire avec grad checkpoint (MLX < 0.20) | Crash apres quelques epochs | Verifier MLX >= 0.20, monitorer avec asitop |
| Conversion SNN sur MoE natif non documentee | Echec de la branche neuroscience | Valider d'abord sur un modele dense (Qwen3.5-7B) avant de tenter le 35B MoE |
| Interference entre stacks entrainees sequentiellement | Degradation des performances globales | OPLoRA init + forgetting check apres chaque stack (deja en place) |

---

## Sources

### MoE-LoRA
- [MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA-based Mixture of Experts](https://arxiv.org/abs/2404.15159) -- Li et al., 2024
- [MoLoRA: Composable Specialization via Per-Token Adapter Routing](https://arxiv.org/abs/2603.15965) -- Shah & Wagle, Microsoft, Mars 2026
- [ReMix: Reinforcement Routing for Mixtures of LoRAs in LLM Finetuning](https://arxiv.org/abs/2603.10160) -- Qiu et al., Meta/UIUC, Mars 2026
- [Brainstacks: Cross-Domain Cognitive Capabilities via Frozen MoE-LoRA Stacks](https://arxiv.org/abs/2604.01152) -- Abu Ayyash, Avr 2026
- [HotMoE: Hybrid Routing for a Mixture of LoRA Experts](https://ojs.aaai.org/index.php/AAAI/article/download/40383/44344) -- Huang et al., AAAI
- [LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](https://arxiv.org/abs/2509.25684) -- ICLR 2026
- [LoRA-Switch: Boosting the Efficiency of Dynamic LLM Adapters](https://arxiv.org/abs/2405.17741) -- Kong et al., 2024
- [TT-LoRA MoE](https://arxiv.org/abs/2504.21190) -- Kunwar et al., 2025
- [MoE-PEFT / MixLoRA framework](https://github.com/TUDB-Labs/MixLoRA)

### MLX Fine-tuning
- [mlx-tune: Fine-tune LLMs on Apple Silicon](https://github.com/ARahim3/mlx-tune) -- ARahim3, v0.4.22
- [Fine-Tuning a 20B Model on 512GB Apple Silicon](https://blog.petieclark.com/fine-tuning-a-20b-model-on-512gb-apple-silicon-what-i-learned/) -- Petie Clark, Jan 2026
- [Fine-Tuning on Mac: LoRA & QLoRA with MLX](https://insiderllm.com/guides/fine-tuning-mac-lora-mlx/) -- InsiderLLM, Fev 2026
- [MLX-LM memory issue #828](https://github.com/ml-explore/mlx-lm/issues/828) -- simap, Jan 2026
- [MLX-LM kernel panic #883](https://github.com/ml-explore/mlx-lm/issues/883) -- BrunoCerberus, Fev 2026
- [MLX-examples gradient checkpointing leak #1076](https://github.com/ml-explore/mlx-examples/issues/1076) -- hschaeufler, Oct 2024
- [Run and Fine-Tune LLMs on Mac with MLX-LM](https://markaicode.com/run-fine-tune-llms-mac-mlx-lm/) -- Markaicode, Mars 2026

### Routage multi-adaptateurs
- [LORAUTER: Effective LoRA Adapter Routing using Task Representations](https://arxiv.org/abs/2601.21795) -- Randl et al., Jan 2026
- [LoRAHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition](https://arxiv.org/abs/2307.13269) -- Huang et al., 2023
- [HiLoRA: Adaptive Hierarchical LoRA Routing](https://openreview.net/pdf?id=NPoMZuiHnM) -- ICLR 2026 submission
- [Building a Production-Ready Adaptive Multi-Adapter LLM System](https://isuruig.medium.com/building-a-production-ready-adaptive-multi-adapter-llm-system) -- Chathuranga, Medium

### SNN + LLM
- [SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks](https://arxiv.org/abs/2302.13939) -- Zhu et al., TMLR 2024
- [LAS: Loss-less ANN-SNN Conversion for Fully Spike-Driven LLMs](https://ojs.aaai.org/index.php/AAAI/article/view/37151) -- Chen et al., AAAI 2026
- [NeuTransformer: LLM Inference Engines based on SNN](https://arxiv.org/abs/2510.00133) -- Balaji et al., Sep 2025
- [How to Get Spiking LLMs? Dual ANN-to-SNN Conversion](https://openreview.net/pdf?id=nzkObUsajY) -- ICLR 2026 submission
- [Winner-Take-All Spiking Transformer for Language Modeling](https://openreview.net/pdf?id=7PKGMNcM0w) -- ICLR 2026 submission
- [SFormer: One-Timestep ANN-to-SNN via Scale-and-Fire Neurons](https://openreview.net/pdf?id=ShOT80BjUZ) -- ICLR 2026 submission
- [Spikingformer: A Key Foundation Model for SNN](https://github.com/TheBrainLab/Spikingformer) -- AAAI 2026
- [Xpikeformer: Hybrid Analog-Digital Hardware for Spiking Transformers](https://arxiv.org/abs/2408.08794) -- Song et al., 2025
- [LAS code repository](https://github.com/lc783/LAS)

### Qwen Fine-tuning
- [Qwen LoRA Fine-tuning official docs](https://www.mintlify.com/QwenLM/Qwen/finetuning/lora)
- [How to Fine-Tune Qwen 3.5 on Your Own Data](https://ainewsgrid.com/blog/how-to-fine-tune-qwen-35-on-your-own-data) -- AI News Grid, Mars 2026
- [Unsloth Qwen3 fine-tuning (FR)](https://unsloth.ai/docs/fr/modeles/tutorials/qwen3-how-to-run-and-fine-tune)
- [Guide complet LoRA/QLoRA (FR)](https://ayinedjimi-consultants.fr/ia-fine-tuning-llm-lora-qlora.html) -- AYI-NEDJIMI, Fev 2026
- [Qwen3.5 Fine-Tuning with LoRA guide](https://sotaaz.com/post/qwen35-finetuning-en) -- SOTAAZ, Mars 2026
