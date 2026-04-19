# C2: Downstream LLM Evaluation — Partial Results

**Status**: BLOCKED after 1 second of runtime on 2026-04-19. The kxkm-ai llama-server (Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf on port 8000) that was reachable during setup crashed during the first few bench queries. All 600 downstream LLM calls (gen + judge) returned `Connection refused`. Partial data captured below; full run pending server restart.

## What did work

- **VQC training on C3 real corpus**: 400 held-out samples, train_acc = **0.497** (up from 0.25 on C1 synthetic). Confirms C3's corpus-quality finding.
- **Routing accuracy of 3 routers on 100 held-out eval queries**:
  - VQC: **0.340** (vs C3 full-run 0.410 — consistent within seed noise)
  - Random: **0.070** (≈ chance 0.10)
  - Oracle: **1.000** (correct by construction)
- **Harness + self-judge wiring + pipeline** committed (`59e4e6d`), dry-run verified.

## What did NOT work

- The gen+judge LLM calls returned `Connection refused` (errno 61). Mean score = 0 everywhere (the harness treated every call as an empty answer scored 0).
- Cause: kxkm-ai llama-server process died between T6 probe (200 OK) and T7 launch (000). Root cause unknown — possibly OOM, VRAM conflict, or external kill.

## Partial findings still interpretable

1. **Routing accuracy on held-out eval is VQC > random**, confirming the VQC classifier learned something useful on real data (0.340 vs 0.070 = ~5× chance).
2. **Training accuracy 0.497** is the C3-real VQC result; eval on a harder held-out split drops to 0.340 — natural generalization gap.
3. **Score quality signal is ZERO**: cannot conclude anything about whether routing improves downstream answer quality. The main C2 question is unanswered.

## To resume C2

1. Restart kxkm-ai llama-server:
   ```bash
   ssh kxkm-ai "cd ~/llama.cpp && nohup ./build/bin/llama-server -m ~/models/Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf -c 8192 --host 0.0.0.0 --port 8000 > ~/llama-server.log 2>&1 &"
   ```
   (Actual command depends on the prior launch — check `~/.zsh_history` or `~/start-llama.sh` if exists.)

2. Re-run the bench:
   ```bash
   uv run python scripts/bench_downstream_c2.py --per-domain 10 --vqc-epochs 300 --output results/c2-downstream.json
   ```

3. Expected runtime: ~20-40 min wall-clock on kxkm-ai for 600 LLM calls.

## Implications for Paper A

- **Routing accuracy result** (VQC 0.340 vs random 0.070) can be reported immediately — it does not depend on the judge.
- **Downstream quality claim** REQUIRES the judge to work. Paper A §5 should be marked DRAFT until the real run completes.
- Alternative judge path if kxkm-ai stays unstable: use Claude API (requires ANTHROPIC_API_KEY) or a running model on the Studio (requires stopping the in-progress chat-fr training). Both are out-of-session escalations.
