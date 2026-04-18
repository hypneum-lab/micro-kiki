# Paper A Reframe Plan — Aeon-as-AMI-Memory

## Plan de recadrage — Aeon comme mémoire AMI

**Date** : 2026-04-19
**Statut** : BROUILLON — résultats PoC A (Text-JEPA) intégrés ; rétroportages du case study LayerNorm(delta) confirmés
**Remplace** : `paper-outline-triple-hybrid.md` comme direction principale v0.3 du papier de recherche

---

## 1. Motivation du recadrage

Le plan initial (`paper-outline-triple-hybrid.md`) positionnait micro-kiki comme « First hybrid quantum-neuromorphic-classical routing for domain-expert LLM inference ». Trois raisons de pivoter :

1. **Cadre théorique plus solide**. LeCun's « A Path Towards Autonomous Machine Intelligence » (arXiv:2206.15331, 2022) fournit un cadre modulaire (7 modules) où nos composants se rangent naturellement. La communauté AMI/JEPA (I-JEPA, V-JEPA 2, LeJEPA/SIGReg arXiv:2511.08544) cherche précisément des implémentations concrètes du Module 7.

2. **Les résultats empiriques orientent le framing**. Le PoC B v2 (`2026-04-17-aeon-predictor-poc-alpha.md`) a prouvé que le centrage DinoV3 + rollback produit +22 % MRR sur flux structurés (condition D), mais que le stack-conditioning est fragile sous centrage (0 % win_stack D, 1 % E). La contribution principale n'est donc pas « quantum router » mais **working memory alignée JEPA avec garde-fous runtime**.

3. **Quantique et neuromorphique pas mûrs pour un papier unique**. Le VQC reste simulateur, la SNN LAS en cours, Akida non livré. Mieux vaut isoler la contribution défendable (Aeon-Module-7) et différer VQC/SNN à un Paper B.

## 2. Nouvelle thèse

**Aeon est une implémentation candidate du module Short-Term Memory (Module 7) de l'architecture Autonomous Machine Intelligence de LeCun, construite sur des principes alignés JEPA : elle prédit les états latents successeurs via un MLP numpy, applique des mécanismes anti-collapse runtime (centrage à la DinoV3 par défaut ; LayerNorm(delta) comme alternative préservant le stack), et détecte l'effondrement représentationnel via un tripwire déterministe std-ratio qui déclenche un rollback des poids.** Nous démontrons trois résultats alignés : (1) le centrage apporte +22 % de MRR sur flux structurés sans coût runtime supplémentaire (< 1 MB de poids, < 2 s d'entraînement pour 1000 tours sur M5) ; (2) l'anti-collapse par LayerNorm(delta) préserve le signal de stack-conditioning (59 % win_stack à 300 epochs, 0.447 predictive_mrr vs 0.090 null_mrr, condition L2) ; (3) la compression Text-JEPA du Configurator embedding atteint une compression 3× (384→128 dims) tout en conservant 97 % de la précision de routage VQC (0.925 → 0.900). Ensemble, ces mécanismes forment le substrat AMI Module 7 : working memory avec robustesse anti-collapse, compatibilité stack-conditioning, et routage à signal compressible.

**Ce que nous NE revendiquons PAS** : (a) implémentation AMI complète — Module 7 seul, pas de Perception / World Model / Cost / Critic / Actor bouclés en feedback ; (b) world model génératif — transition latente, pas observation ni token ; (c) planification multi-étapes — horizon = 1 ; (d) validation complète quantique / neuromorphique — VQC + Text-JEPA Configurator mentionnés comme composants intégrés hors scope ici, spin-off au Paper B ; (e) comparaison en production contre des compétiteurs — toutes les expériences sur données synthétiques et conversationnelles, divulgué.

## 3. Mapping des modules AMI

| Module AMI | Composant micro-kiki | Force de la claim | Notes |
|------------|----------------------|-------------------|-------|
| **1. Configurator** | VQC router (6 qubits, 35 classes, ~180 params) + compression Text-JEPA | **Strong** | VQC : 86,8 % val_acc non équilibré, 53 % équilibré. Text-JEPA : compression 3× (384→128 dims) conserve 97 % de la précision de routage (0.925 → 0.900 sur classification 10 domaines). C'est la voie Configurator du Module 1 AMI dans le pipeline complet. |
| **2. Perception** | n/a | **Aucune** | Pas d'environnement externe ; les entrées sont directement des embeddings texte (MiniLM-L6) |
| **3. World Model** | Aeon LatentMLP (h_t → h_{t+1}) | **Partielle** | Prédit des transitions latentes à 1 pas, pas la dynamique complète du monde |
| **4. Cost** | juge CAMP (arXiv:2604.00085) | **Partielle** | Évaluation post-hoc ; pas de feedback d'apprentissage bouclé |
| **5. Critic** | n/a | **Aucune** | Pas de fonction de valeur |
| **6. Actor** | stack LLM (Qwen3.5-35B-A3B + LoRAs) | **Déléguée** | Exécution déléguée à la stack LLM ; hors scope du Paper A |
| **7. Short-Term Memory** | **Aeon (Atlas + Trace + LatentMLP + anti-collapse + rollback)** | **STRONG** | Claim principale du papier ; backing empirique PoC B v2 (centrage + LayerNorm(delta)) + PoC A Text-JEPA (compression Configurator) |

Le papier se concentre sur les lignes 1 (compressed) et 7 (working memory). Les lignes 3, 4, 6 sont mentionnées dans la discussion comme points d'ancrage pour des papiers suivants.

## 4. Plan section par section

### §1 Introduction
Le cadre AMI de LeCun et la question ouverte des implémentations concrètes du Module 7 pour texte/dialogue. Succès de JEPA en vision vs. le gap pour l'état symbolique. Énoncé de la thèse (§2). *Source* : PoC B α §1–§2, arXiv:2206.15331, arXiv:2511.08544.

### §2 Related Work
Famille JEPA (I-JEPA, V-JEPA 2, LeJEPA/SIGReg) ; contraste avec les world models génératifs (DreamerV3 arXiv:2301.04104, TD-MPC2 arXiv:2410.16662) ; DINO self-distillation (DINO/v2/v3, arXiv:2104.14294 / 2304.07193 / 2508.10104) ; LMs avec mémoire augmentée (MemGPT arXiv:2310.08560, Larimar arXiv:2403.11901, RETRO arXiv:2112.04426). *Source* : `related-work-aeon-predictor.md` (104 lignes, directement réutilisables).

### §3 Architecture Aeon
Substrat (Atlas SIMD + Trace NetworkX) ; LatentMLP 384→256→384 numpy loss cosinus (< 1 MB) ; centrage à la DinoV3 (stateless, pas d'EMA) ; détecteur de collapse + rollback des poids (sécurité runtime) ; fallback cold-start (identité en dessous de 500 paires). *Source* : `src/memory/aeon_predictor.py`, PoC B α §2.

### §4 Protocole expérimental
Flux synthétiques (random-walk + stack-structured), 1000 tours, 100 requêtes held-out, 5 ablations A–E. Métriques : Recall@5, MRR, win_pred %, win_stack %, final_loss. Tableau complet issu du PoC B α §3.

### §5 Résultats
- **5.1 Le centrage fournit l'anti-collapse** — condition D MRR 0.413 → 0.498 (+22 %), E se stabilise à 0.500. Testé sur flux structurés (régime de saturation).
- **5.2 LayerNorm(delta) restaure le signal de stack** — condition L2 à 300 epochs : win_stack = 59 %, predictive_mrr 0.447 vs null_mrr 0.090. La normalisation per-sample du résidu préserve les offsets spécifiques au stack que le centrage par moyenne mobile détruit.
- **5.3 Text-JEPA valide la compression Configurator** — compression 3× (384→128) conserve 97 % de la précision de routage VQC (0.925 → 0.900 sur classification 10 domaines). Embeddings conversationnels réels.
- **5.4 Le centrage nuit sur random-walk** — A–B MRR 0.263 → 0.228 ; borne la claim au régime de saturation. Divulgué.
- **5.5 Activation du rollback** — test unitaire + détection déterministe de collapse ; télémétrie issue du long-run PoC B v2.
- **5.6 Persistance cross-session** — AeonSleep : 36 recalls / 14 tours vs 0 pour le LLM brut.

### §6 Discussion
Centrage+rollback comme primitive Module 7 ; hypothèse d'interférence centrage↔stack (µ/σ par stack ou adaptateur de stack appris) ; plafond de saturation ; limitations (synthétique, horizon=1, pas de boucle fermée) ; feuille de route vers AMI complet via VQC (Configurator) + stack LLM (Actor).

### §7 Conclusion
Résumé de la claim forte ; disclaimer partial-AMI ; release du code + weights (Apache 2.0) ; travaux futurs (conversations réelles via PoC A Text-JEPA, centrage per-stack, horizon multi-étapes).

### Appendices
A. Couverture de tests (33 tests) ; B. Budget compute (M5, ~2 s / 1000 tours, pas de GPU) ; C. Hyperparamètres + seeds.

## 4. Scorecard empirique

**Ce que le PoC B v2 et le PoC A Text-JEPA ont effectivement prouvé (strong claims, backing empirique)** :

| Résultat | Preuve | Force |
|----------|--------|-------|
| Le centrage apporte +22 % MRR sur flux structurés | Condition D vs baseline, MRR 0.413 → 0.498 | **Strong** |
| Préservation du stack par LayerNorm(delta) | Condition L2 à 300 epochs : win_stack = 59 %, predictive_mrr 0.447 vs null_mrr 0.090. La normalisation per-sample du delta résiduel préserve les offsets spécifiques au stack. | **Strong** |
| La compression Text-JEPA valide la voie Configurator | Compression 3× de l'embedding (384→128 dims) conserve 97 % de la précision de routage VQC (baseline 0.925, Text-JEPA 0.900 sur classification 10 domaines). | **Strong** |
| Le rollback sur collapse std fonctionne de manière déterministe | Test unitaire `test_collapse_detector_triggers` | **Strong** |
| Déploiement numpy 100K paramètres faisable | Taille du code + < 1 MB de poids, runtime mesuré sur M5 | **Strong** |
| Mémoire cross-session via AeonSleep | Design AeonSleep existant, 36 recalls / 14 tours | **Strong** |
| Le fallback cold-start est gracieux | `predict_next()` retourne h_t quand non prêt | **Strong** |

**Résultats faibles / divulgués** :

| Résultat | Preuve | Traitement |
|----------|--------|------------|
| Le centrage détruit le signal de stack, LayerNorm(delta) le restaure | A 23 % → D 0 % sous centrage ; L2 59 % sous LayerNorm(delta) | **Divulgué**, présenté comme deux stratégies anti-collapse compatibles avec des propriétés de stack différentes |
| Le centrage nuit sur random-walk (retrieval non saturé) | A–B MRR 0.263 → 0.228 | **Divulgué**, borne la claim au régime de saturation |
| Flux synthétiques uniquement | Toutes les expériences sur random-walk + stack-structured | **Divulgué**, engagement à un follow-up sur données réelles |

**Ce qui reste nécessaire avant soumission** :

- Baseline LeJEPA si le code sort. <TBD — statut du code arXiv:2511.08544>
- Latence sous charge serving (> 100 requêtes concurrentes). <TBD>
- Ablation centrage on/off au moment du serving. <TBD — script d'eval nécessaire>
- Benchmark contre la baseline LeJEPA (si publiée). <TBD — eval comparative nécessaire>

## 5. Mise à jour du statut stack-conditioning

**Stack-conditioning : validé via LayerNorm(delta).** L'anti-collapse par centrage détruit le signal de stack (0–1 % win_stack sur les conditions D, E, F du PoC B v2). Remplacer le centrage par une LayerNorm per-sample du delta résiduel restaure le signal de stack à 59 % win_stack à 300 epochs (condition L2, PoC B case study). **Ce n'est pas un papier à mécanisme unique mais une ÉTUDE DE COMPATIBILITÉ des choix anti-collapse.** Nous présentons les deux : le centrage comme défaut plus simple et robuste en production ; LayerNorm(delta) comme alternative préservant le stack. Le trade-off en déploiement réel dépend de la tâche downstream.

## 6. Anticipation des reviewers

1. **« Ce n'est pas une vraie implémentation AMI — LeCun a 7 modules, vous en touchez un. »** Nous ne revendiquons jamais un AMI complet. Titre + abstract scopent à « candidate Module 7 implementation ». §6.5 énumère les pièces manquantes. Le framing est « building block », pas « system ».

2. **« Ce n'est pas un vrai prédicteur JEPA (pas de masking, pas de teacher network). »** Nous revendiquons une convergence méthodologique sur trois principes : pas d'EMA, pas de hacks stop-gradient, prédiction dans l'espace latent. Nous ne reproduisons pas I-JEPA / V-JEPA 2 sur le plan architectural. Le centrage est notre mécanisme spécifique, philosophiquement cousin de SIGReg (arXiv:2511.08544).

3. **« Le stack-conditioning était votre nouveauté PoC et il a échoué. »** Le stack-conditioning fonctionne sous anti-collapse LayerNorm(delta) mais échoue sous centrage à la DinoV3. Le papier présente LES DEUX résultats : le centrage comme défaut plus simple et robuste ; LayerNorm(delta) comme alternative préservant le stack. Ce n'est pas un papier à mécanisme unique mais une **ÉTUDE DE COMPATIBILITÉ des choix anti-collapse** — voir §5 pour tous les détails.

4. **« Toutes les expériences synthétiques. »** Divulgué en §6.4. Engagement au follow-up Paper A' sur données réelles PoC A Text-JEPA (compression Text-JEPA validée sur tours conversationnels réels ; le centrage n'a pas d'hypothèse spécifique au synthétique). Le rollback est agnostique aux données (mécanisme de sécurité).

5. **« Pourquoi AMI-class sans world model ni boucle d'actor ? »** Le Module 7 est la contribution working-memory. LeCun 2022 §3.6 décrit le Module 7 comme standalone-describable. Nous scopons à « Module 7 substrate », pas « AMI system ».

## 7. Ciblage de venue

**Primaire** : NeurIPS 2026 Workshop on **World Models & Cognitive Architectures** (historique pour I-JEPA, V-JEPA, DreamerV3). Call attendu mai-juin 2026, deadline typiquement juillet-septembre.

**Secondaire (désormais plausible)** : ICLR 2027 workshop track on **Cognitive Architectures** ou **Memory in LLMs** (avec les wins LayerNorm(delta) + Text-JEPA, le portfolio empirique se renforce pour ICLR). Call attendu septembre 2026, deadline typiquement novembre 2026.

**Tertiaire** : ICML 2026 Workshop on **Cognitive Architectures for Language Agents** (call attendu février-mars, déjà passé ; soumission tardive possible ou pivot post-acceptation iterate).

**Stratégie** : soumettre d'abord NeurIPS 2026 workshop pour peer review + feedback (turnaround plus rapide), préparer simultanément la soumission ICLR 2027 (plus de pages, plus d'expériences). Main-track ICLR possible si nous sécurisons une story de déploiement réel ou une comparaison baseline supplémentaire d'ici octobre 2026.

## 8. Ce qu'il faut couper du papier original

Référence : `paper-outline-triple-hybrid.md`, 359 lignes.

**Couper ou réduire drastiquement** (déplacer vers Paper B ou le papier SpikingKiki) :
- Deep-dive mécaniciste du VQC quantique (§3.2 ancien, §5.1 ancien) — GARDER une story brève d'intégration Text-JEPA en §3 (1 paragraphe sur le VQC router + voie de compression). Déplacer SQA training / gate-optimization vers Paper B.
- Détails de conversion SNN LAS (§3.3, §5.3, §7.2) — déplacer entièrement vers le papier `spikingkiki-v3-final.md`.
- Curriculum d'entraînement LoRA 32 domaines (§3.4, §5.2, §7.3) — garder uniquement la mention de l'identité Qwen base ; discussion complète → papier systems micro-kiki.
- Breakdown de latence end-to-end du pipeline cognitif multi-tour (§5.4, §6.4) — trimmer à une demi-page focalisée sur la latence spécifique à la mémoire.
- Arbitrage CAMP Negotiator (§5.4, §6.5) — garder 1 paragraphe ; ce n'est pas le focus.

**Étendre substantiellement** :
- Architecture Aeon (§3.5 ancien → §3 entier nouveau, 4-5 pages).
- Philosophie du centrage DinoV3-style (nouvelle sous-section).
- Positionnement AMI Module 7 (nouveau §1 + §6.1).
- Divulgation complète de l'ablation stack-conditioning (§5.4 nouveau).
- Sécurité runtime via rollback (nouvelle sous-section §3.4).

Effet net : le Paper A devient ~12–14 pages focalisées sur Aeon comme substrat Module 7. Le Paper B (travail systems VQC + SNN) est spin-off séparément.

## 9. Calendrier d'écriture

En supposant que les résultats PoC A Text-JEPA arrivent cette semaine (Task 14 selon la mémoire projet) :

- **Semaine 1 (2026-04-20 → 04-26)** : finaliser ce recadrage, sécuriser les chiffres PoC A, rédiger §1 + §2 + §3.
- **Semaine 2 (04-27 → 05-03)** : rédiger §4 + §5 + §6 + §7. Importer les figures depuis le script d'eval PoC B α.
- **Semaine 3 (05-04 → 05-10)** : revue interne (coauteur ou auto-revue attentive), traiter préemptivement les objections reviewer-anticipation, resserrer la prose.
- **Semaine 4 (05-11 → 05-17)** : polish final, appendices, checklist de reproductibilité, soumission au workshop choisi ou upload arXiv.

**Horizon réaliste** : 3–4 semaines pour une soumission workshop. Un main-track ICLR demanderait +6–8 semaines d'expériences supplémentaires.

## 10. Décisions ouvertes pour l'auteur

Cinq décisions nécessaires avant de démarrer réellement la rédaction :

1. **Garder ou abandonner complètement le framing quantique dans le Paper A ?** CLOS : **GARDER le quantique comme module Configurator.** Text-JEPA valide le combo VQC+Text-JEPA comme Configurator AMI (§3). Story brève d'intégration (1 para en §3), détails SQA complets du VQC → Paper B.

2. **Papier unique (Aeon-as-Module-7) ou split (A1 Aeon + A2 VQC Configurator) ?** Plan actuel : papier unique, VQC+Text-JEPA intégrés en §3 comme claim Configurator forte. Alternative : écrire A1 (Aeon+Centrage+LayerNorm(delta)) maintenant, différer A2 pour les systems VQC. **Recommandation : papier unique A, §3 couvre Configurator+compression, Paper B gère les détails quantiques**.

3. **Citer les résultats PoC A (Text-JEPA) de manière prominente ?** Plan actuel : OUI, citer en §3 (Configurator) et §4 (scorecard). Text-JEPA est désormais une validation co-revendiquée, pas optionnelle. **Recommandation : citer dans la section §3.1 Configurator et dans le tableau en §4**.

4. **Inclure une section théorique sur « la loss JEPA comme régulariseur de working-memory » ?** Plan actuel : non — garder le papier empirique. Alternative : une section théorique d'1 page positionnant le centrage comme opérateur de projection analogue aux projections Cramér-Wold de SIGReg. **Recommandation : garder la théorie pour un short paper companion ou un tech report**.

5. **Workshop track ou main track ?** Plan actuel : workshop d'abord (NeurIPS 2026 World Models, mai-sept). Alternative : sauter le workshop, aller main ICLR 2027 (septembre-novembre, avec expériences supplémentaires). **Recommandation : workshop d'abord — turnaround plus rapide, feedback reviewer précieux, pipeline vers ICLR main-track avec les lifts LayerNorm(delta)+Text-JEPA**.

---

**Métadonnées du document**
Auteur : équipe recherche micro-kiki
Statut de revue : en attente du sign-off auteur sur les décisions ouvertes §10
Étape suivante : une fois §10 résolu, démarrer la rédaction de §1 du papier proprement dit
