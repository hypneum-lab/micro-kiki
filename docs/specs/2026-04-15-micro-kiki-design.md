# Micro_KIKI — 32 Expert Brainstacks MoE

## Overview

Fleet de 32 domaines spécialisés sur base Qwen3.5-4B, assemblés via Brainstacks (MoE-LoRA + null-space projection + meta-routeur sigmoid). Déployable sur RTX 4090 24 Go.

## Architecture

```
┌──────────────────────────────────────────────────┐
│              META-ROUTEUR SIGMOID                 │
│     32 sorties indépendantes par prompt           │
│     Input: mid-layer + last-layer hidden states   │
│     ~2M params, 5ms overhead                      │
└──────────┬───────────────────────────────────────┘
           │ active les stacks pertinents (seuil 0.12)
           ▼
┌──────────────────────────────────────────────────┐
│         BASE MODEL: Qwen3.5-4B (gelée, Q4)       │
│         2.5 Go VRAM permanent                     │
│         GatedDeltaNet, 262K ctx, thinking natif   │
├──────────────────────────────────────────────────┤
│  Stack 1: chat-fr       │  Stack 17: embedded     │
│  Stack 2: python        │  Stack 18: stm32        │
│  Stack 3: typescript    │  Stack 19: iot           │
│  Stack 4: cpp           │  Stack 20: freecad       │
│  Stack 5: rust          │  Stack 21: platformio    │
│  Stack 6: html-css      │  Stack 22: power         │
│  Stack 7: shell         │  Stack 23: emc           │
│  Stack 8: sql           │  Stack 24: dsp           │
│  Stack 9: yaml-json     │  Stack 25: spice-sim     │
│  Stack 10: kicad-dsl    │  Stack 26: electronics   │
│  Stack 11: spice        │  Stack 27: web-frontend  │
│  Stack 12: docker       │  Stack 28: web-backend   │
│  Stack 13: lua-upy      │  Stack 29: music-audio   │
│  Stack 14: math         │  Stack 30: devops        │
│  Stack 15: security     │  Stack 31: llm-orch      │
│  Stack 16: reasoning    │  Stack 32: kicad-pcb     │
└──────────────────────────────────────────────────┘
  Chaque stack : MoE-LoRA (4 experts, rank 16, top-2)
  ~150 Mo/stack × 32 = ~4.8 Go disque
  2-4 stacks actifs simultanément en VRAM (~1-2 Go)
```

## Contraintes hardware

| Machine | VRAM | Rôle |
|---------|------|------|
| Mac M3 Ultra 512 Go | Illimité | Training des stacks, distillation teacher |
| RTX 4090 24 Go (kxkm-ai) | 24 Go | Inférence, training Unsloth |

### Budget VRAM RTX 4090 (inférence)

| Composant | VRAM |
|-----------|------|
| Base Qwen3.5-4B Q4 | 2.5 Go |
| Meta-routeur | 0.01 Go |
| 2-4 stacks actifs | 0.6-1.2 Go |
| KV cache (4K ctx) | ~0.5 Go |
| **Total** | **~4-5 Go** |
| **Marge** | **19 Go** |

## Base model : Qwen3.5-4B

### Pourquoi

| Critère | Qwen3.5-4B | Gemma 4 E4B | Nemotron Nano 4B |
|---------|-----------|------------|-----------------|
| MMLU-Pro | **79.1** | 69.4 | ~65 |
| GPQA-D | **76.2** | 58.6 | — |
| Thinking natif | **Oui** | Non | Non |
| 262K context | **Oui** | Non | Non |
| Même famille que 122B | **Oui** | Non | Non |
| French (201 langues) | **Oui** | Limité | Non |
| Q4 VRAM | 2.5 Go | 3 Go | 2.5 Go |
| License | Apache 2.0 | Apache 2.0 | NVIDIA Open |

Qwen3.5-4B gagne sur tous les critères sauf HumanEval brut (Gemma 4 E4B a 85%+). Mais avec le thinking mode et la même architecture DeltaNet que nos teachers 122B/35B, la distillation sera optimale.

## Brainstacks — adaptations pour 32 domaines

### Paramètres

| Param | Papier (5 dom.) | Micro_KIKI (32 dom.) |
|-------|----------------|---------------------|
| Base model | Gemma 3 12B | Qwen3.5-4B |
| `h_dim` | 3840 | 3072 (Qwen3.5-4B) |
| `ns_top_k_dirs` | 64 | **32** |
| Espace null utilisé | 8.3% | **33%** (32×32/3072) |
| MoE experts/stack | 4 | 4 |
| LoRA rank | 16 | 16 |
| Residual boost rounds | 2-3 | 1-2 |
| Stack size | 567 Mo (12B) | **~150 Mo** (4B) |
| Total disque | 5.67 Go | **~4.8 Go** |
| Meta-routeur sorties | 5 | **32** |

### Ordre curriculum (séquentiel, chaque domaine ne dégrade pas les précédents)

```
Phase 1 — Fondations (scaffolding)
  1. chat-fr        : instruction-following + français
  2. reasoning      : meta-raisonnement, thinking chains

Phase 2 — Coding core (logique procédurale)
  3. python         : coding principal
  4. typescript     : web + types
  5. cpp            : systèmes + embedded
  6. rust           : safety + concurrence

Phase 3 — Coding secondaire
  7. html-css       : frontend markup
  8. shell          : scripts, DevOps
  9. sql            : requêtes, schémas
  10. yaml-json     : configs, schemas
  11. docker        : containers
  12. kicad-dsl     : netlists, footprints
  13. spice         : simulations
  14. lua-upy       : scripting embarqué

Phase 4 — Domaines techniques (upgrade kiki-*)
  15. embedded      : ESP-IDF, firmware général
  16. stm32         : STM32 HAL, CubeMX
  17. iot           : protocoles, MQTT, BLE
  18. freecad       : CAO mécanique
  19. platformio    : build system
  20. power         : alimentation, régulateurs
  21. emc           : CEM, filtrage
  22. dsp           : traitement signal
  23. spice-sim     : simulation circuits
  24. electronics   : analogique, RF, composants
  25. kicad-pcb     : routage PCB, DRC

Phase 5 — Applications
  26. web-frontend  : React, Vite, patterns
  27. web-backend   : FastAPI, Hono, Express
  28. music-audio   : audio DSP, TTS, instruments
  29. devops        : Docker, Tailscale, CI/CD
  30. llm-orch      : RAG, agents, routing LLM

Phase 6 — Compléments
  31. math          : raisonnement math/physique
  32. security      : crypto, auth, OWASP
```

## Distillation — chaîne progressive multi-teacher

```
Teachers (sur Mac 512 Go) :
  ├── Qwen3.5-122B-A10B Opus-v3 (en training, val 0.497)
  ├── Gemma 4 31B (18 Go bf16, rapide)
  └── Devstral 2 123B (pour le coding)

Chaîne :
  122B → 35B → 4B (progressive, 80-88% qualité retenue)

Par domaine :
  1. Générer ~2K exemples spécialisés avec le teacher approprié
  2. Dédupliquer cross-domaine
  3. SFT via Brainstacks (inner loop + null-space)
```

### Teachers par domaine

| Domaines | Teacher principal | Teacher secondaire |
|----------|------------------|-------------------|
| Coders (3-14) | Devstral 2 123B | Gemma 4 31B |
| Embedded (15-25) | 122B Opus-v3 | Données kiki-* existantes |
| Reasoning/Math (1-2, 31) | 122B Opus-v3 | Opus API |
| Web/DevOps (26-30) | Gemma 4 31B | 122B Opus-v3 |
| Security (32) | 122B Opus-v3 | — |

## Données — sources et déduplication

### Sources existantes

| Source | Exemples | Domaines |
|--------|----------|----------|
| final-opus-v3-1 | 11 880 | Reasoning, général |
| 10 LoRA kiki-* datasets | ~5 000 estimé | Embedded, hardware |
| CodeFeedback | 156K | Coding |
| OpenCodeReasoning | 735K | Python coding |
| Magicoder-OSS | 75K | Multi-langue code |

### Budget par domaine

| Type | Exemples/domaine | Total 32 domaines |
|------|-----------------|-------------------|
| Minimum viable | 500 | 16K |
| Recommandé | 2 000 | 64K |
| Optimal | 5 000 | 160K |

### Déduplication cross-domaine

Chaque exemple va dans **1 seul domaine** (celui avec le score de pertinence le plus élevé). Pas de doublons entre stacks — le null-space projection gère le transfert cross-domaine.

## Pipeline d'entraînement

### Par stack (~30 min sur Mac, ~20 min sur RTX)

```
1. Charger base Qwen3.5-4B gelée
2. Calculer projecteur null-space des stacks frozen
3. Ajouter stack MoE-LoRA (4 experts, rank 16, 7 projections)
4. SFT sur ~2K exemples du domaine (~500 steps)
5. Residual boost : round 2 si amélioration > 0.002
6. Freeze → offload CPU/disque
7. Évaluer tous les domaines précédents (forgetting check)
```

### Temps total

| Phase | Mac seul | Mac + RTX parallèle |
|-------|----------|---------------------|
| Distillation données (32 × 2K) | ~48h | ~48h (Mac teacher) |
| Training 32 stacks | ~16h | **~8h** |
| Residual boost | ~8h | ~4h |
| Meta-routeur | ~2h | ~1h |
| Éval + itérer | ~4h | ~4h |
| **Total** | **~78h** | **~65h** |

## Meta-routeur — 32 sigmoid

### Architecture

```
Input: 0.45 × mid_hidden + 0.55 × last_hidden (Qwen3.5-4B h_dim=3072)
→ Linear(3072, 512)
→ Global attention (learned query)
→ 32 × cross-attention (domain query vectors)
→ MLP fusion (GELU, dropout 0.1)
→ 32 sigmoid outputs avec temperature scaling
```

### Entraînement (outcome discovery)

Pour chaque prompt du dataset mixte :
1. Loss base-only
2. Loss avec chaque stack individuel (32 forwards)
3. Greedy search : ajouter les stacks qui réduisent la perte > 0.01
4. Target : 80% découvert + 20% prior label
5. BCE loss, 8 epochs, cosine LR

### Règles d'inférence

- Chat floor : 0.20 (toujours actif minimum)
- Gate threshold : 0.12 (en dessous, stack pas chargé)
- Max stacks simultanés : 4 (contrainte VRAM)

## Export et déploiement

### RTX 4090 (kxkm-ai)

```
Base Q4 : models/Qwen3.5-4B-Q4.gguf (2.5 Go)
Stacks : output/micro-kiki/stacks/ (32 × 150 Mo)
Routeur : output/micro-kiki/router.safetensors (8 Mo)
Inference : vLLM avec LoRA switching OU script custom
```

### Mac Studio (local)

```
Base bf16 : models/Qwen3.5-4B (8 Go)
Stacks MLX : même format
Inference : mlx-lm avec adapter switching
```

## Critères de succès

| Métrique | Cible |
|----------|-------|
| Coding (HumanEval) | > 70% (base Qwen3.5-4B ~55%) |
| Reasoning (GPQA) | > 80% (base 76.2%) |
| Embedded (custom eval) | Réponses correctes ESP-IDF, KiCad |
| French (custom eval) | Fluent, pas de code-switching |
| Zero forgetting | Delta < 0.03 sur tous les domaines précédents |
| VRAM RTX 4090 | < 8 Go en inférence |
| Latence routeur | < 10 ms |
| Swap stack | < 2s |

## Apple Silicon — Triple pipeline ANE+GPU+CPU (Mac uniquement)

Sur le Mac M3 Ultra, l'ANE (Neural Engine) est libre quand le GPU fait l'inférence MoE. Trois intégrations exploitent cette ressource inutilisée.

### Architecture triple pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    MAC M3 ULTRA 512 Go                        │
│                                                               │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │   GPU METAL      │  │   ANE (32 cores)  │  │    CPU      │ │
│  │   76 cores       │  │   ~2W, 14 tok/s   │  │  24 cores   │ │
│  │                  │  │                    │  │             │ │
│  │  Base Qwen3.5-4B │  │  A. Scorer GRPO   │  │  Routeur    │ │
│  │  + 2-4 stacks    │  │  B. Draft 0.8B    │  │  sigmoid    │ │
│  │  actifs          │  │     (speculative)  │  │  (5ms)      │ │
│  │                  │  │  C. Meta-routeur   │  │             │ │
│  │  Génération      │  │     + embedding    │  │  Offload    │ │
│  │  principale      │  │                    │  │  stacks     │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
│         ↕ mémoire unifiée (zero-copy)  ↕                      │
└─────────────────────────────────────────────────────────────┘
```

### A. ANE comme scorer/filtre qualité

Pendant le training GRPO (Phase 3), le GPU génère K=4 réponses par prompt.
L'ANE score chaque réponse en parallèle via un reward model léger.

```
GPU : génère réponse[i+1]  ──────────────────────→
ANE : score réponse[i]     ──→ reward = 0.85 ──→
                               (14 tok/s)
```

| Composant | Unité | Modèle |
|-----------|-------|--------|
| Générateur | GPU Metal | Qwen3.5-4B + stacks MoE |
| Scorer | ANE CoreML | Qwen3.5-0.8B converti CoreML |
| Reward head | ANE | Linear(h_dim, 1) sur le scorer |

Gain : **scoring gratuit** (0 impact sur la vitesse de génération GPU).

### B. Speculative decoding via ANE

Un draft model Qwen3.5-0.8B (0.5 Go) tourne sur ANE. Il propose N tokens,
le GPU (4B + stacks) vérifie en un seul forward pass.

```
ANE (draft 0.8B) : propose tokens [t1, t2, t3, t4, t5]  → 200+ tok/s
GPU (4B + stacks) : vérifie [t1✓, t2✓, t3✓, t4✗]       → 1 forward
                    accepte 3 tokens au lieu de 1
```

| Métrique | Sans speculative | Avec speculative ANE |
|----------|-----------------|---------------------|
| tok/s GPU | ~30-50 | ~30-50 |
| tok/s effectifs | ~30-50 | **~60-100** (2-3x) |
| VRAM supplémentaire | 0 | 0 (ANE séparé) |

Le draft 0.8B partage le même tokenizer que le 4B (même famille Qwen3.5).
La conversion CoreML est prouvée (on a déjà converti le 9B DeltaNet).

### C. ANE pour meta-routeur + embedding

Le meta-routeur (2M params) et l'embedding layer sont des opérations légères
qui peuvent tourner entièrement sur ANE, libérant le GPU pour les stacks MoE.

```
Prompt arrive
  │
  ▼
ANE : embedding(tokens) → hidden states          (~0.5 ms)
ANE : meta_routeur(hidden) → 32 sigmoid scores   (~2 ms)
CPU : sélectionne top-4 stacks, charge du SSD     (~50 ms)
  │
  ▼
GPU : forward(hidden, stacks actifs) → tokens     (bulk du compute)
```

Le forward GPU ne fait QUE le compute MoE lourd. Embedding + routing = gratuit sur ANE.

### Modèles CoreML nécessaires

| Modèle | Usage | Taille CoreML | Conversion |
|--------|-------|--------------|------------|
| Qwen3.5-0.8B | Draft speculative | ~1 Go | À faire (ANEMLL ou custom) |
| Meta-routeur | Routing 32 stacks | ~8 Mo | Trivial (petit MLP) |
| Embedding layer | Token → hidden | ~50 Mo | Trivial |
| Reward scorer | GRPO scoring | ~1 Go | Clone du draft + reward head |

**Note** : Le 0.8B Qwen3.5 utilise GatedDeltaNet comme le 4B/9B.
Notre conversion DeltaNet → CoreML (Phase 1 ANE research) s'applique directement.

### Quand utiliser le triple pipeline

| Scénario | GPU | ANE | CPU | Gain |
|----------|-----|-----|-----|------|
| Inférence standard | 4B + stacks | Draft 0.8B (spec) | Routeur | **2-3x tok/s** |
| Training GRPO | Génère K=4 | Score réponses | Routeur | **Scoring gratuit** |
| Training SFT | Training LoRA | Idle | — | Pas de gain |
| Batch scoring | Idle | Score dataset | — | **14 tok/s continu** |

### Impact sur les critères de succès

| Métrique | Sans ANE | Avec ANE |
|----------|----------|----------|
| Inférence tok/s | 30-50 | **60-100** (speculative) |
| GRPO scoring overhead | +50% temps | **~0%** (parallèle) |
| Latence routeur | ~5 ms CPU | **~2 ms ANE** |
| Consommation | ~20W GPU seul | ~22W (GPU+ANE) |

## Risques

| Risque | Mitigation |
|--------|-----------|
| 32 domaines saturent le null-space | Réduire ns_top_k_dirs à 32 (33% espace) |
| Base 4B trop petite pour 32 spécialisations | Upgrade vers Qwen3.5-9B (5.5 Go Q4, tient RTX) |
| Brainstacks pas testé avec Qwen3.5 | Port du code Gemma → Qwen (même API transformers) |
| kxkm-ai inaccessible (Tailscale) | Training 100% sur Mac, deploy GGUF via NFS |
