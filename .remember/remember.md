# Handoff

## State
I got PRD from 0 to 21/50 stories. POC v2 multi-turn memory works end-to-end with real MiniLM embeddings (Turn 4 recalls inductor values). SFT training running on Studio (PID 86223) via `mlx_lm_fork` — freecad in progress, then platformio→power→dsp→electronics. kicad-dsl FAILED (zombie process stole RAM) — needs solo relaunch after the chain finishes. VQC 8-qubit trained (47.2%, weights at `outputs/vqc-8q-weights.npz`). 134K dataset examples, 728+ tests pass.

## Next
1. **Check SFT chain**: `ssh studio "strings ~/micro-kiki/outputs/sft-all-niches-v3.log | grep -E 'START|DONE|FAIL'"` — then relaunch kicad-dsl solo: `ssh studio "cd ~/micro-kiki && .venv/bin/python3 scripts/train_niches_mlxtune.py --domain kicad-dsl"`
2. **After all 10 adapters done**: story-14 forgetting check, story-15 eval all stacks vs base
3. **DPO**: relaunch 480B+35B servers on Studio AFTER SFT finishes, run `scripts/generate_dpo_pairs.py`

## Context
- **ZERO compute on GrosMac** — user was explicit, everything on Studio (`ssh studio`) or kxkm-ai.
- Training MUST use `mlx_lm_fork` at `~/KIKI-Mac_tunner/lib/mlx_lm_fork` (not standard mlx_lm — Metal OOM).
- Always `pkill -f _train.py` before relaunching training (zombie subprocesses steal 96GB Metal).
- Commit hook: no `Co-Authored-By`, subject ≤ 50 chars.
