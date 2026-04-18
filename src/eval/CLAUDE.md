# src/eval/ — forgetting gate, benchmarks, MAP harness

Gatekeeper for every new stack + continuous quality monitoring.

## Modules

- `forgetting.py` — angle(base, adapted) + held-out win-rate. Rollback trigger: angle < 30° AND win-rate drop > 0.03.
- `stack_eval.py` — per-domain accuracy on classified test splits.
- `reward_functions.py` — shared reward primitives (used by DPO/GRPO scripts).
- `map_harness.py`, `map_dispatcher_bench.py`, `map_negotiator_bench.py` — MAP paper benchmarks (see `docs/specs/map-paper-spec.md`).

## Benchmarks

- `bofenghuang/mt-bench-french`
- `manu/FrenchBench` collection
- Per-domain classified splits from the pipeline

## Adaptive judge

- Fast pass: **Qwen3.5-35B** on kxkm-ai (cheap, default).
- Escalate to **Mistral-Large-Opus** (Studio) only if judge confidence < 0.5.
- Never skip the cheap pass to "save time" — the escalation rate is the signal you care about.

## Bias monitoring

- KnowBias neuron debiasing: measured before AND after each stack.
- RBD runtime detector fires on **every** response; flagged outputs are re-generated via DeFrame (see `src/cognitive/`).

## Forgetting gate workflow

1. Stack trained → artifact in `checkpoints/<domain>/`.
2. Run `uv run python src/eval/forgetting.py --stack <domain>`.
3. If gate fails → rollback, do **not** register the adapter with the router.
4. Stacks 02–03 get a retroactive baseline check the first time the framework activates on a newer stack.

## Anti-patterns (eval-specific)

- Don't evaluate on training data — reuses the split loader to enforce disjointness.
- Don't skip the bias check — it's per-response, not per-stack.
- Don't compare stacks across quantization levels (Q4 vs BF16) — the delta is noise.
- Don't run the expensive judge first; always fast-pass then escalate.
- Don't treat a passing forgetting gate as "no regression" — also check domain accuracy on the OTHER stacks.
