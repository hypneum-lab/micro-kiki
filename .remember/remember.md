# Handoff

## State
PRD 50/50 complete. All 10 SFT adapters trained via restart wrapper on Studio. POC v2 proven end-to-end: 100% routing, multi-turn memory recall, real MiniLM embeddings. Forgetting check: 4/10 PASS. Stacks vs base eval: 3/10 improved. SNN analysis complete (27B+35B energy benchmark). 800+ tests pass. DPO/GRPO training blocked (no MLX trainer, needs kxkm-ai with Unsloth/trl).

## Next
1. **DPO/GRPO on kxkm-ai**: Install Unsloth, load 35B Q4 on RTX 4090, train DPO with 18 pairs, GRPO with reward functions from `src/eval/reward_functions.py`.
2. **Retrain power adapter**: Current loss 1.95 (worst). Re-run `scripts/train_with_restart.py --domain power --chunk 20` on Studio with more epochs.
3. **Paper submission**: `docs/paper-outline-triple-hybrid.md` has all experimental results. Write final sections.

## Context
- **ZERO compute on GrosMac** — Studio or kxkm-ai only.
- Restart wrapper: `scripts/train_with_restart.py --chunk 20` + `val_batches: 0` + `mlx_lm_fork`.
- 480B + 35B servers killed on Studio (free RAM for training).
- Commit hook: no Co-Authored-By, subject ≤ 50 chars.
