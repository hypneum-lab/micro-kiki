# C2-LoRA — Blocked by Infrastructure Inconsistency

**Status**: BLOCKED at Task 3 (pre-deploy). Not executed.

## What happened

The plan assumed that the 10 per-domain LoRA adapters trained per PRD stories 4-13 (`~/micro-kiki/outputs/stacks/stack-<domain>/` on Studio) shared a consistent training configuration, allowing a single `linear_to_lora_layers(model, num_layers, lora_config)` call at server startup followed by in-place adapter swaps. Inspection of the actual `adapter_config.json` files refutes this.

## Measured reality (sample of 5 stacks)

| Stack | rank | alpha | iters | Base model recorded in config |
|---|---|---|---|---|
| spice | 16 | 32 | 11,542 | `qwen3.5-35b-a3b` |
| stm32 | 8 | 16 | 11,542 | `qwen3.5-35b-a3b` |
| dsp | 8 | 16 | **20** | `qwen3.5-35b-a3b` |
| freecad | 4 | 8 | 394 | `qwen3.5-35b-a3b` |
| platformio | 4 | 8 | **20** | `qwen3.5-35b-a3b` |

Three blocking issues emerge:

1. **Base model is Qwen3.5-35B-A3B, not Qwen3.6-35B-A3B.** The project `CLAUDE.md` invariants state the base is Qwen3.6; the actual training output disagrees. Either `CLAUDE.md` was updated speculatively before the base upgrade, or the adapters are stale.

2. **LoRA configuration varies per adapter** (rank ∈ {4, 8, 16}, alpha ∈ {8, 16, 32}). The plan's single `linear_to_lora_layers(rank=16, alpha=32, …)` shell conversion at server startup produces LoRALinear layers incompatible with adapters trained at rank 4 or 8 — `model.load_weights` would error on tensor-shape mismatch when swapping to a lower-rank adapter.

3. **Half the adapters are effectively untrained.** `dsp` and `platformio` ran for only 20 iterations (vs 11,542 for `spice` and `stm32`). At 1 sample per iteration, that is at most 20 gradient steps on a 35B model — functionally a smoke test, not a trained adapter. Results from those stacks would be noise, invalidating any per-domain comparison.

## What would be required to unblock

Roughly in increasing cost:

1. **Scope reduction (~1 day)**: run C2-LoRA only on the adapters with rank 16 + iters > 1000. A quick grep confirms this set is likely ≤ 3 domains, not enough for comparable measurement against the 10-domain C2 baseline.

2. **Retrain undertrained adapters (~10-20h Studio time)**: rerun `train_stack.py` for the 5 under-trained domains (dsp, platformio, freecad, ...) at a uniform rank/alpha matching the best-trained ones (spice's 16/32). This is feasible but conflicts with whatever the previous training session was doing.

3. **Port adapters to llama.cpp GGUF LoRA format (uncertain)**: would enable inference on kxkm-ai and bypass the MLX swap constraints, but the conversion path for rank-varied adapters is non-standard and the shape mismatches persist.

## Why this is documented instead of fixed in-session

- This session already landed four complete Phase C sub-projects (C1, C2, C3, C5) plus a diagnostic (C2-diag) plus a torch-vqc tool release.
- The C2-LoRA experiment's core purpose — distinguishing "prompt-based persona-refusal" from "weight-level specialisation failure" — remains a useful open question, but investigating it requires training work outside the session's scope.
- Documenting the infrastructure state accurately is itself a finding: the project's adapter collection is **not in a consistent state** despite the PRD marking stories 4-13 as `done`. Paper A § Discussion should note this as a caveat on the project's reproducibility rather than pretend uniform adapters exist.

## Impact on Paper A

No change to numbers or structure. Add one sentence to §5 "Discussion and Limitations":

> Real LoRA adapters corresponding to the 10 domains exist in the project's output directory but were trained with heterogeneous ranks and training-iteration counts, preventing direct comparison of the C2 prompt-based pseudo-adapter result against a weight-level-specialisation baseline. Resolving this requires uniform retraining outside the scope of this work.

## Plan status

`docs/superpowers/plans/2026-04-20-c2-lora-experiment.md` is marked **BLOCKED** with a pointer to this file. Tasks 1-2 (tests + server impl) remain **committed and useful** for any future attempt — the server code is correct pending uniform adapters.

## References

- Original plan: `docs/superpowers/plans/2026-04-20-c2-lora-experiment.md`
- C2 baseline results: `results/c2-downstream.json` + `docs/paper-a/c2-downstream-results.md`
- Diagnostic that motivated C2-LoRA: `docs/paper-a/c2-diagnostic.md`
- Adapter configs on Studio: `~/micro-kiki/outputs/stacks/stack-<domain>/adapter_config.json`
