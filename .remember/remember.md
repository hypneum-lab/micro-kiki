# Handoff

## State
POC v2 end-to-end PROVEN with real MiniLM embeddings (dim=384, intra=0.80). Multi-turn Turn 4 recalls inductor values. Dynamic memory budget (3000 chars / n_episodes). VQC 8-qubit trained (47.2% acc, weights at `outputs/vqc-8q-weights.npz`). 16/50 PRD stories done. 723 tests pass. 132K dataset examples across 10 domains. AeonServingHook updated with structured format. SFT training for 6 remaining niches RUNNING on Studio (PID 9054).

## Next
1. **Monitor SFT training** on Studio: `ssh studio "tail -20 ~/micro-kiki/outputs/sft-niches-training.log"` — 6 domains (kicad-dsl, freecad, platformio, power, dsp, electronics), ETA ~4-5h.
2. **After SFT**: run forgetting check (story-14), eval all 10 stacks vs base (story-15).
3. **DPO**: relaunch 480B+35B servers on Studio AFTER SFT completes (RAM conflict), then run `scripts/generate_dpo_pairs.py`.

## Context
- **ZERO compute on GrosMac** — everything on Studio or kxkm-ai.
- Studio DPO servers KILLED to free 206 GB for MLX training. Relaunch after SFT.
- Training uses `~/KIKI-Mac_tunner/.venv/bin/python3` on Studio (has mlx-lm).
- 4 domains already trained from previous session: spice, emc, stm32, embedded.
- Commit hook: no Co-Authored-By, subject ≤ 50 chars.
