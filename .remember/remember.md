# Handoff

## State
POC v2 multi-turn memory WORKS — Turn 4 successfully recalls inductor values from previous turns. Dynamic memory budget (3000 chars ÷ n_episodes) with format `### Previous conversation context:` and `---` separators. 100% routing accuracy via score-based keyword router (DSP and PlatformIO misroutes fixed). VQC confidence gate at 0.30 threshold prevents untrained VQC from overriding keywords. VQC 8-qubit training running on Studio (multiprocessing 16 workers, 144 parameters, 46% acc at epoch 29). Embedding refactor complete — AeonPalace requires embed_fn or model_path (no hash fallback). MiniLM trained: intra=0.80, inter=0.66. StackExchange scraper added 1,784 examples (embedded 464, spice 457, emc 344, kicad 319, stm32 200). Datasets re-merged: 132,186 examples from 7+ sources across 10 domains. PRD at 15/50 stories done. 723 tests pass, 0 fail (new: 17 anti-bias + 39 SNN/LAS + 9 embedding data pipeline). Paper outline updated with real experimental results (+198 lines). README.md created with architecture diagram, domains table, quick start.

## Next
1. **Check Studio VQC training**: 8-qubit balanced VQC (multiprocessing, 16 workers). Copy results back when done, commit weights.
2. **Continue PRD stories**: story-4 (kicad-dsl LoRA training) is next incomplete story.
3. **Paper**: add multi-turn memory success (Turn 4 inductor recall) to Section 5.4 experimental results.
4. **Embedding integration**: wire MiniLM (intra=0.80, inter=0.66) into Aeon memory pipeline (story-49).

## Context
- **ZERO compute on GrosMac** — all training/inference goes to Studio (SSH `studio`) or kxkm-ai. User was explicit about this.
- VQC 8-qubit training: 144 parameters, 46% acc at epoch 29, multiprocessing with 16 workers on Studio.
- MiniLM embedding model trained and saved: intra-domain similarity 0.80, inter-domain 0.66.
- StackExchange scraper output: data/stackexchange/ (1,784 examples across 5 domains).
- Datasets merged: 132,186 examples total from 7+ sources.
- Commit hook rejects `Co-Authored-By` trailers and subjects > 50 chars.
