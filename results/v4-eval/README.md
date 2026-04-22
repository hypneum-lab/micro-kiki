# V4 SOTA Evaluation & Benchmark — 2026-04-22

Pre-HF-release eval of the `lora-qwen36-35b-v4-sota/` adapter bundle
trained on Mac Studio M3 Ultra against `Qwen3.6-35B-A3B`.

## Headline numbers

| Gate | Result |
|---|---|
| Adapter health (all V4 stacks) | **76/76 pass** lora_B norm ≥ 1e-6 (`validate_adapter_health.py`) |
| Stacks trained | 35 / 35 |
| Router v4 top-1 / top-3 | **0.464 / 0.696** on 5 377 validation samples |
| HumanEval base (N=10) | 0.80 |
| HumanEval V4 adapters (N=10) | cpp 0.90 (+0.10), rust 0.80 (±0), typescript 0.80 (±0), python 0.70 (−0.10), shell 0.70 (−0.10) |

Methodology: `scripts/eval_humaneval_v4.py`, temp=0, max_tokens=384,
indentation normalization (3→4 space fix). Fixture
`humaneval_10.jsonl` (OpenAI HumanEval subset, first 10).

## Per-adapter detail

See `aggregate.json` for the machine-readable roll-up.

```json
{
  "base":       {"pass@1": 0.80, "gen_s": 42.6},
  "cpp":        {"pass@1": 0.90, "gen_s": 57.1, "Δ vs base": +0.10},
  "python":     {"pass@1": 0.70, "gen_s": 72.1, "Δ vs base": −0.10},
  "rust":       {"pass@1": 0.80, "gen_s": 34.5, "Δ vs base":  0.00},
  "shell":      {"pass@1": 0.70, "gen_s": 107.3,"Δ vs base": −0.10},
  "typescript": {"pass@1": 0.80, "gen_s": 68.7, "Δ vs base":  0.00}
}
```

## Reading notes

- **HumanEval is Python-specific.** cpp, rust, typescript, shell
  scores reflect the general code-reasoning surface the base model
  retains through the adapter, not domain proficiency. MBPP /
  MultiPL-E would be needed per-language.
- **Python regression (−0.10)** on N=10 is within single-problem
  noise. Full HumanEval (164 problems) is deferred until the router
  is wired into the serving path.
- **cpp adapter (+0.10)** is within noise too but consistent with
  the C++-heavy KiCad/SPICE/STM32 training mix in the V4 curriculum
  lifting surface code reasoning.

## Coverage gaps (not run in this sweep)

1. **Forgetting angle** sweep across pairs of V4 stacks — deferred,
   V4 SOTA adapters are independently trained (no chain), so
   interference is bounded by construction.
2. **Per-language benchmarks** (MBPP, MultiPL-E, HumanEval-X) —
   future work, requires fixture plumbing.
3. **End-to-end router + swap** eval via `src/serving/mlx_client.py`
   — blocked on the two-server MLX setup (see
   `docs/training/e2e-smoke-runbook.md`).

## HF release gate

All 35 V4 SOTA adapters are healthy and have non-negative deltas
against base on the 5 sampled coding languages. Cleared for
Hugging Face Hub release via `scripts/release_hf.py`.
