# Micro_KIKI: Custom MoE from 32 LoRA Experts on RTX 4090

> Deep research report — 2026-04-15
> Target: RTX 4090 24GB (kxkm-ai), Q4 quantization, 32 specialized experts

---

## 1. Base Model Selection for Experts

### Ranking (best to worst for this use case)

| Rank | Model | Params | Q4 Size | HumanEval | MMLU-Pro | Reasoning | French | Architecture | License |
|------|-------|--------|---------|-----------|----------|-----------|--------|-------------|---------|
| **1** | **Qwen3.5-4B** | 4B | ~2.5 GB | Best-in-class 4B | Best-in-class 4B | Strong (thinking mode) | 201 languages | GatedDeltaNet + GatedAttn hybrid | Apache 2.0 |
| 2 | Qwen3.5-2B | 2B | ~1.3 GB | Good | Good | Good | 201 languages | Same hybrid arch | Apache 2.0 |
| 3 | SmolLM3-3B | 3B | ~1.8 GB | Strong | Competitive | Dual-mode reasoning | 6 languages | GQA + NoRope | Apache 2.0 |
| 4 | Phi-4-mini | 3.8B | ~2.2 GB | 83.7% ARC-C | Good | Strong math | Limited | Dense transformer | MIT |
| 5 | Gemma-3-4B | 4B | ~2.5 GB | 71.3% | Good | 89.2% GSM8K | 140+ languages | Dense transformer | Gemma license |
| 6 | Qwen3.5-0.8B | 0.8B | ~0.5 GB | Limited | Limited | Basic | 201 languages | Same hybrid arch | Apache 2.0 |

### Verdict: **Qwen3.5-4B**

Raisons:
- **Meilleur rapport performance/taille** dans la classe 4B, gagne sur quasiment tous les benchmarks
- **Architecture hybride GatedDeltaNet** = KV-cache constant, 262K contexte natif meme en petit
- **201 langues** dont le francais, critique pour nos use cases
- **Meme famille que nos modeles de production** (Qwen3.5-122B, 35B) = transfert de connaissance optimal lors de la distillation
- **Apache 2.0** = aucune restriction commerciale
- **MLX et Unsloth support natif** pour le training
- **Thinking mode natif** via `<think>` tags = compatible avec notre pipeline de distillation Opus existante

### Alternative low-VRAM: Qwen3.5-2B

Si 32 x 4B ne rentre pas (cf. section VRAM), la fallback est Qwen3.5-2B. Meme architecture, meme tokenizer, meme pipeline.

---

## 2. MoE Upcycling: 3 Approches Viables

### Approche A: MoLoRA — Mixture of LoRA (RECOMMANDE)

**Paper**: [MoLoRA: Composable Specialization via Per-Token Adapter Routing](https://arxiv.org/abs/2603.15965) (Mars 2026, Microsoft Research)

**Principe**: Base model gelé + N adaptateurs LoRA spécialisés + routeur léger (MLP 2 couches) qui sélectionne l'expert par token.

**Résultat clé**: MoLoRA permet à Qwen3-1.7B de dépasser Qwen3-8B sur 4 benchmarks de raisonnement tout en étant 4.7x plus petit.

**Pourquoi c'est optimal pour Micro_KIKI**:
- **Base gelée**: 1 seule copie de Qwen3.5-4B en mémoire (~2.5 GB en Q4)
- **32 adaptateurs LoRA rank-32**: chacun ~50-100 MB en Q4 = 1.6-3.2 GB total
- **Routeur**: quelques MB, négligeable
- **Total VRAM estimé**: **~6-8 GB** — largement dans le budget 24 GB
- **Training**: chaque expert s'entraîne indépendamment avec GRPO/SFT standard
- **Inférence**: top-2 routing = 2 LoRA actifs par token, très rapide

**Implémentation**:
- [aicrumb/MoLora](https://github.com/aicrumb/MoLora) — implémentation de référence
- [Applied-Machine-Learning-Lab/MOELoRA-peft](https://github.com/Applied-Machine-Learning-Lab/MOELoRA-peft) — SIGIR'24
- PEFT issue HF [#1156](https://github.com/huggingface/peft/issues/1156) — intégration en cours

### Approche B: Brainstacks — Frozen MoE-LoRA Stacks

**Paper**: [Brainstacks](https://arxiv.org/abs/2604.01152) (Avril 2026)

**Principe**: Stacks de MoE-LoRA gelés, empilés séquentiellement, avec un méta-routeur sigmoïde par outcome. Compatible QLoRA 4-bit + rsLoRA.

**Résultat clé**: Les stacks encodent des "primitives cognitives" transférables cross-domaine. Le routeur découvre que les prompts médicaux routent vers chat+math dans 97% des cas.

**Avantage**: Continual learning — on peut ajouter de nouveaux domaines sans retrainer les anciens. Null-space projection = zéro forgetting.

**Inconvénient**: Plus complexe à implémenter, papier très récent, pas encore de repo public mature.

### Approche C: mergekit-moe — Full Model MoE Upcycling

**Repo**: [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit) + [mergekit-moe-qwen3](https://github.com/llmcompe2025-team-semishigure/mergekit-moe-qwen3)

**Principe**: Prend N copies complètes d'un modèle dense fine-tuné sur N domaines, combine les MLP en experts MoE avec attention/layernorm partagés.

**Config type**:
```yaml
base_model: Qwen/Qwen3.5-4B
gate_mode: hidden  # ou cheap_embed pour init, random pour sparse upcycling
dtype: float16
experts_per_token: 2
experts:
  - source_model: ./expert-python/
    positive_prompts: ["python", "def ", "class "]
  - source_model: ./expert-rust/
    positive_prompts: ["rust", "fn ", "impl "]
  # ... 30 autres experts
```

**Problème critique pour 32 experts**: Chaque expert stocke les poids MLP complets. 32 x Qwen3.5-4B MLP = **tous les poids dupliqués 32 fois**. En Q4, le modèle résultant ferait ~32 x 2.5 GB = **~80 GB** — ne rentre PAS sur RTX 4090.

**Verdict**: mergekit-moe est adapté pour 4-8 experts. Pour 32, c'est MoLoRA ou rien.

### Tableau comparatif

| Critère | MoLoRA | Brainstacks | mergekit-moe |
|---------|--------|-------------|--------------|
| VRAM 32 experts Q4 | **~6-8 GB** | **~8-12 GB** | **~80 GB** (KO) |
| Training indépendant | Oui | Oui (séquentiel) | Oui puis merge |
| Routeur | Learned MLP | Outcome sigmoïde | Hidden/embed init |
| Continual learning | Manuel | **Natif** | Non |
| Maturité | Papier + implémentations | Papier récent | **Outil mature** |
| Support Qwen3.5 | Via PEFT/custom | Via QLoRA | Fork Qwen3 dispo |

---

## 3. Distillation: De 122B vers 4B

### Pipeline Recommandée (Progressive Chain)

```
Qwen3.5-122B-A10B (Opus-distilled, notre meilleur modèle)
    │
    ▼  Distillation KL + SFT
Qwen3.5-35B-A3B (intermédiaire, réduit le gap de capacité)
    │
    ▼  Distillation KL + SFT
Qwen3.5-9B (step intermédiaire optionnel)
    │
    ▼  Distillation KL + SFT
Qwen3.5-4B × 32 (experts spécialisés)
```

### Pourquoi une chaîne progressive

La recherche 2025-2026 confirme: quand le gap de capacité entre teacher et student est trop grand, la distillation directe est sous-optimale. Un "middle teacher" intermédiaire bridge le gap. C'est exactement ce que tu as déjà avec tes modèles 122B → 35B.

### Stratégie par expert

Pour chaque expert spécialisé (ex: `expert-python`, `expert-embedded-c`):

1. **Dataset spécialisé**: Filtrer notre dataset Opus (11880 train) par domaine + générer des données spécifiques avec le teacher 122B
2. **SFT spécialisé**: Fine-tune Qwen3.5-4B sur le dataset filtré avec LoRA rank-32
3. **GRPO spécialisé**: RL avec récompenses vérifiables (tests code, validation math, etc.)
4. **SimPO**: Alignement préférence sur le domaine

### Données existantes exploitables

D'après le projet KIKI actuel:
- **final-opus-v3-1**: 11880 train + 626 valid (multi-domaine)
- **Distilled 35B**: ~2000 exemples
- **Sonnet-coding pipeline**: 18K exemples coding (OpenCodeReasoning, Magicoder, etc.)

### Tools

- **Sur Mac Studio M3 Ultra**: SFT + LoRA via `mlx-lm` / `mlx-tune` (notre pipeline existante)
- **Sur KXKM-AI RTX 4090**: GRPO/SimPO via Unsloth (plus rapide pour les boucles RL)
- **Distillation teacher**: Qwen3.5-122B-A10B sur Mac Studio en bf16

---

## 4. VRAM Budget Table (RTX 4090 24GB)

### Approche MoLoRA (RECOMMANDEE)

| Composant | Params | Q4 Size | In VRAM |
|-----------|--------|---------|---------|
| Qwen3.5-4B base (gelé) | 4B | 2.5 GB | 2.5 GB |
| 32 LoRA rank-32 (all loaded) | 32 x ~6.5M = 208M | ~0.1 GB | 0.1 GB |
| 32 LoRA rank-64 (all loaded) | 32 x ~13M = 416M | ~0.2 GB | 0.2 GB |
| 32 LoRA rank-128 (all loaded) | 32 x ~26M = 832M | ~0.5 GB | 0.5 GB |
| Routeur MLP (2 couches) | ~0.5M | negligible | ~0 |
| KV-cache (8K ctx, Q4) | — | ~1-2 GB | 1-2 GB |
| Framework overhead | — | ~1.5 GB | 1.5 GB |
| **Total rank-32** | **~4.2B** | — | **~5.1 GB** |
| **Total rank-64** | **~4.4B** | — | **~5.3 GB** |
| **Total rank-128** | **~4.8B** | — | **~5.7 GB** |

**Marge restante**: 18-19 GB libres sur 24 GB. On pourrait aller jusqu'à rank-256 ou 64 experts sans problème.

### Approche mergekit-moe (REFERENCE / KO pour 32)

| Config | Total Params | Q4 Size | Fits 24GB? |
|--------|-------------|---------|------------|
| 8 x Qwen3.5-4B, top-2 | ~32B total | ~18 GB | Juste |
| 16 x Qwen3.5-4B, top-2 | ~64B total | ~36 GB | Non |
| 32 x Qwen3.5-4B, top-2 | ~128B total | ~72 GB | Non |
| 32 x Qwen3.5-2B, top-2 | ~64B total | ~36 GB | Non |
| 32 x Qwen3.5-0.8B, top-2 | ~25.6B total | ~14 GB | Oui mais qualité faible |
| 8 x Qwen3.5-4B, top-2, offload | ~32B total | ~6 GB GPU + RAM | Oui avec latence |

### Approche Brainstacks-style

| Composant | Size | Notes |
|-----------|------|-------|
| Base model Qwen3.5-4B (QLoRA 4-bit) | 2.5 GB | Frozen |
| 10 MoE-LoRA stacks (4 LoRA par stack) | ~1-2 GB | rsLoRA, rank-32 |
| Meta-router | negligible | Sigmoid outcome |
| KV-cache + overhead | 2-3 GB | |
| **Total** | **~6-8 GB** | Large margin |

### Training VRAM (single expert, sur RTX 4090)

| Step | VRAM Required | Fits 24GB? |
|------|---------------|------------|
| SFT LoRA rank-32 sur Qwen3.5-4B (QLoRA) | ~8 GB | Oui |
| SFT LoRA rank-64 sur Qwen3.5-4B (QLoRA) | ~10 GB | Oui |
| GRPO (K=4 rollouts) sur Qwen3.5-4B | ~14 GB | Oui |
| SimPO sur Qwen3.5-4B | ~12 GB | Oui |
| SFT LoRA sur Qwen3.5-9B (QLoRA) | ~12 GB | Oui |
| GRPO sur Qwen3.5-9B | ~18 GB | Oui (serré) |

---

## 5. Plan d'Experts Proposé (32)

### Coding (12 experts)

| # | Expert | Dataset Source |
|---|--------|---------------|
| 1 | Python general | OpenCodeReasoning + Magicoder |
| 2 | Python ML/data science | Filtered from above |
| 3 | TypeScript/JavaScript | OpenCodeInstruct |
| 4 | Rust | OpenCodeInstruct + filtered |
| 5 | C/C++ embedded | Custom from 122B teacher |
| 6 | Go | OpenCodeInstruct |
| 7 | System programming | Custom (kernel, drivers, bare metal) |
| 8 | SQL/databases | Custom |
| 9 | Shell/DevOps | Custom |
| 10 | Web frontend (React, CSS) | Custom |
| 11 | Code review / debugging | OpenHands trajectoires |
| 12 | Architecture / design patterns | Custom from 122B |

### Reasoning (6 experts)

| # | Expert | Dataset Source |
|---|--------|---------------|
| 13 | Math / formal reasoning | DeepSeek-Math, GSM8K |
| 14 | Logic / puzzles | GRPO verifiable |
| 15 | Scientific reasoning | Custom |
| 16 | Chain-of-thought general | Existing Opus data |
| 17 | Planning / decomposition | Custom from 122B |
| 18 | Multi-step problem solving | Codeforces-CoTs |

### Embedded / Hardware (5 experts)

| # | Expert | Dataset Source |
|---|--------|---------------|
| 19 | ESP32/STM32 firmware | Custom from KIKI-Mac_tunner data |
| 20 | KiCad / PCB design | Custom from mascarade dataset |
| 21 | Electronics theory | Custom |
| 22 | RTOS / bare metal | Custom |
| 23 | Hardware debugging | Custom |

### Knowledge (5 experts)

| # | Expert | Dataset Source |
|---|--------|---------------|
| 24 | French language | Filtered multilingual |
| 25 | Technical writing | Custom |
| 26 | General knowledge | MMLU-style |
| 27 | Tools / API usage | Agent trajectories |
| 28 | System administration | Custom |

### Creative / Other (4 experts)

| # | Expert | Dataset Source |
|---|--------|---------------|
| 29 | Creative writing FR | Custom |
| 30 | Instruction following | Alpaca-style + Opus |
| 31 | Safety / alignment | Custom preference data |
| 32 | Meta-routing / fallback | Mixed data |

---

## 6. Tooling Concret

### Stack recommandé

| Outil | Usage | Repo |
|-------|-------|------|
| **Unsloth** | Training LoRA/QLoRA sur RTX 4090 | [unslothai/unsloth](https://github.com/unslothai/unsloth) |
| **mlx-lm / mlx-tune** | Training LoRA sur Mac Studio | [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) |
| **PEFT** | LoRA management, merge | [huggingface/peft](https://github.com/huggingface/peft) |
| **MoLoRA (custom)** | Router training + inference MoE | [aicrumb/MoLora](https://github.com/aicrumb/MoLora) |
| **MOELoRA-peft** | MoE-LoRA SIGIR'24 ref impl | [Applied-ML-Lab/MOELoRA-peft](https://github.com/Applied-Machine-Learning-Lab/MOELoRA-peft) |
| **mergekit** | Si on veut essayer 8 experts full merge | [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit) |
| **mergekit-moe-qwen3** | Fork mergekit Qwen3 | [llmcompe2025/mergekit-moe-qwen3](https://github.com/llmcompe2025-team-semishigure/mergekit-moe-qwen3) |
| **llama.cpp** | Export GGUF, inference CPU/GPU | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| **vLLM** | Inference avec LoRA switching | [vllm-project/vllm](https://github.com/vllm-project/vllm) |

### Commandes clés

```bash
# 1. Training d'un expert LoRA sur RTX 4090 via Unsloth
ssh kxkm@kxkm-ai
cd ~/micro-kiki
python train_expert.py \
  --base_model Qwen/Qwen3.5-4B \
  --dataset data/expert-python/train.jsonl \
  --lora_rank 64 \
  --output output/expert-python-lora \
  --epochs 3 \
  --lr 2e-4 \
  --quantize 4bit

# 2. Training du routeur MoLoRA
python train_router.py \
  --base_model Qwen/Qwen3.5-4B \
  --experts output/expert-*/lora \
  --num_experts 32 \
  --experts_per_token 2 \
  --router_lr 1e-3 \
  --router_epochs 5 \
  --eval_data data/router-eval.jsonl

# 3. Inference MoLoRA (tout tient en ~6 GB VRAM)
python inference_molora.py \
  --base_model Qwen/Qwen3.5-4B-Q4 \
  --experts output/expert-*/lora \
  --router output/router/best.pt \
  --experts_per_token 2 \
  --prompt "Write an ESP32 firmware for I2C sensor reading"

# 4. Export GGUF pour llama.cpp (optionnel, pour les top experts)
python -m peft.merge_and_unload \
  --base Qwen/Qwen3.5-4B \
  --lora output/expert-python-lora \
  --output output/expert-python-merged
llama-quantize output/expert-python-merged/ggml-model-f16.gguf \
  output/expert-python-q4km.gguf Q4_K_M
```

---

## 7. Risques et Mitigations

| Risque | Probabilité | Impact | Mitigation |
|--------|------------|--------|------------|
| Routeur MoLoRA mal calibré | Moyenne | Fort | Evaluer sur mixture de domaines, itérer |
| Experts trop spécialisés (forgetting) | Haute | Moyen | Inclure 10-20% de données générales dans chaque expert |
| 32 experts = overhead de routing | Basse | Faible | Top-2 routing, latence négligeable avec LoRA |
| Qualité 4B insuffisante sur taches complexes | Moyenne | Fort | Fallback vers Qwen3.5-9B (tient en VRAM) |
| MoLoRA pas nativement supporté par llama.cpp | Haute | Moyen | Custom inference script ou vLLM |
| Datasets spécialisés insuffisants | Moyenne | Fort | Distillation depuis 122B teacher sur Mac Studio |

---

## 8. Recommandation Finale

### Configuration optimale

```
Base:       Qwen3.5-4B (Apache 2.0, 262K ctx, hybrid GatedDeltaNet)
Experts:    32 LoRA rank-64 (chacun ~13M params, ~100 MB)
Routing:    MoLoRA top-2 per-token (MLP 2 couches)
Quant:      Base Q4_K_M, LoRA en fp16 (petits)
VRAM total: ~5.3 GB inference / ~12 GB training par expert
Inférence:  vLLM ou custom script Python
Training:   Unsloth sur RTX 4090 (KXKM-AI) + mlx-lm sur Mac Studio
```

### Phases d'implémentation

1. **Phase 1** (1-2 semaines): Distiller 4-5 experts pilotes (Python, C/embedded, reasoning, french, general) depuis le dataset Opus existant
2. **Phase 2** (2-3 semaines): Entraîner le routeur MoLoRA sur un mix des 5 experts, valider la qualité
3. **Phase 3** (3-4 semaines): Générer les datasets spécialisés pour les 27 experts restants via distillation 122B → 4B
4. **Phase 4** (1-2 semaines): Entraîner tous les experts, re-entraîner le routeur
5. **Phase 5** (1 semaine): Benchmark complet, optimisation, déploiement

### Budget total estimé

- **Training**: ~32 x 2h par expert sur RTX 4090 = ~64h GPU
- **Distillation données**: ~100h sur Mac Studio (122B teacher)
- **Router training**: ~4h sur RTX 4090
- **Stockage**: ~3.2 GB pour tous les LoRA + 2.5 GB base = ~6 GB total

---

## Sources

- [MoLoRA: Composable Specialization via Per-Token Adapter Routing](https://arxiv.org/abs/2603.15965)
- [Brainstacks: Cross-Domain MoE-LoRA Stacks](https://arxiv.org/abs/2604.01152)
- [Drop-Upcycling: Sparse MoE Training](https://arxiv.org/abs/2502.19261)
- [LoRAMoE: MoE-Style Plugin](https://arxiv.org/abs/2312.09979)
- [mergekit-moe documentation](https://github.com/arcee-ai/mergekit/blob/main/docs/moe.md)
- [mergekit-moe-qwen3 fork](https://github.com/llmcompe2025-team-semishigure/mergekit-moe-qwen3)
- [MOELoRA-peft (SIGIR'24)](https://github.com/Applied-Machine-Learning-Lab/MOELoRA-peft)
- [MoE Expert Offloading](https://apxml.com/courses/mixture-of-experts-advanced-implementation/chapter-4-efficient-moe-inference/expert-offloading)
- [FlashMoE: SSD Offloading](https://arxiv.org/html/2601.17063)
- [Qwen 3.5 vs Gemma 4 Benchmarks](https://www.maniac.ai/blog/qwen-3-5-vs-gemma-4-benchmarks-by-size)
- [Qwen 3.5 Hardware Guide](https://www.compute-market.com/blog/qwen-3-5-local-hardware-guide-2026)
- [Small Model Benchmarks (distil labs)](https://www.distillabs.ai/blog/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning/)
- [Progressive Distillation Survey](https://link.springer.com/article/10.1007/s10462-025-11423-3)
- [Adaptive Upcycling (EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.1323.pdf)
