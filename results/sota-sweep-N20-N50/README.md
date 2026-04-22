# SOTA sweep (small N) — 2026-04-22 evening

First pass of the multi-benchmark matrix : 3 benchmarks × 7
adapters = 21 runs at small N. Numbers are directional only —
std errors below are in the ±6-10 pp range per cell. A second
pass at N=164/200 follows to confirm the regressions.

## Matrix (Δ vs base by benchmark)

| Adapter | HumanEval N=20 | MBPP N=50 | GSM8K N=50 | verdict |
|---|---|---|---|---|
| **base** | 0.900 | 0.680 | 0.520 | — |
| cpp | +0.05 | **−0.06** | +0.02 | WASH |
| python | −0.05 | **−0.10** | +0.04 | WASH/bad |
| typescript | 0.00 | **−0.10** | +0.06 | WASH |
| rust | 0.00 | **−0.08** | +0.18 | mystery |
| shell | −0.10 | +0.04 | +0.20 | mystery |
| math | −0.05 | **−0.24** 🚨 | 0.00 | TOXIC |

## Key findings

1. **Piste A gains don't generalize**. Piste A (HumanEval
   N=164) : python +6.7 pp, cpp +7.3 pp. This sweep (MBPP
   N=50) : python **−10 pp**, cpp **−6 pp**. The V4 adapters
   appear overfit to HumanEval's problem distribution —
   single-function synthesis with docstring — and regress on
   MBPP's general Python tasks.
2. **Math adapter confirmed toxic**. −24 pp on MBPP extends
   the GSM8K reasoning-adapter catastrophe signal : certain
   V4 adapters actively destroy capability rather than lift it.
3. **Anomalies to investigate**. Shell / Rust on GSM8K show
   +20 / +18 pp — suspicious given the domain mismatch.
   Possibly an artifact of the N=50 subset being easy (base
   hits 0.52 vs 0.455 at N=200).
4. **HumanEval N=20 numbers are unreliable**. Base hit 0.90 vs
   0.604 at N=164 — the 20-problem subset is trivially easier.
   Trust Piste A's N=164 numbers over these.

## Implications

- **Cannot publish V4 as SOTA coding model without multi-benchmark caveats**. A bench-specific win that doesn't transfer is not a general capability claim.
- **Per-adapter validation gate required before any HF release**. See `results/sota-gsm8k/README.md` for the proposed gate spec (lora_B health + in-domain uplift + out-of-domain bound).
- **Next pass** : N=164 (or full set) on MBPP + GSM8K for all 7
  adapters. That closes the statistical gap on the regressions
  and tells us whether this is noise or a real limitation.

## Raw data

| File | Label | pass@1 |
|---|---|---|
| `humaneval-*.json` | 7 adapters, N=20 | see matrix |
| `mbpp-*.json` | 7 adapters, N=50 | see matrix |
| `gsm8k-*.json` | 7 adapters, N=50 | see matrix |
| `gsm8k-base-200.json` + `gsm8k-math-200.json` + `gsm8k-reasoning-200.json` | 3 reference runs at N=200 | 0.455 / 0.475 / 0.045 |
