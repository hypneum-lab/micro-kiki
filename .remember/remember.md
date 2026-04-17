# Handoff

## State
I completed the embedding refactor (Tasks 1-5): AeonPalace no longer has a hash fallback, requires `embed_fn` or `model_path`. 705 tests pass. PRD has 10/50 stories done. POC v2 gets 100% routing with score-based keyword router. VQC confidence gate (0.30 threshold) prevents untrained VQC from overriding keywords. StackExchange scraper added 1,784 examples. DPO generation running on Studio (480B CPU judge). Paper outline updated with real experimental results.

## Next
1. **Check Studio trainings**: VQC balanced (PID 19146, `outputs/vqc-balanced-training.log`) + MiniLM embeddings (PID 19230, `outputs/embedding-training.log`). Copy results back, commit weights.
2. **Fix max_tokens=500 in POC v2** (`scripts/poc_pipeline_v2.py` line ~163): increase to 1024 so multi-turn Turn 4 can recall inductor values.
3. **Re-merge datasets** after StackExchange data: `python3 scripts/merge_all_sources.py` (scraper output added to merge sources). Then retrain embeddings with enriched data.

## Context
- **ZERO compute on GrosMac** — all training/inference goes to Studio (SSH `studio`) or kxkm-ai. User was explicit about this.
- DPO servers still running on Studio: 480B on port 8481, 35B on port 8200 (PIDs 84904, 86034). Kill when done.
- `train_embeddings.py` needs `sentence-transformers` + `accelerate` (installed in Studio venv via pip, not uv).
- Commit hook rejects `Co-Authored-By` trailers and subjects > 50 chars.
