# V4 SOTA HumanEval-164 sweep — 2026-04-22 (Piste A)

Full HumanEval (164 problems, OpenAI HumanEval dataset, MIT) on
Mac Studio M3 Ultra against Qwen3.6-35B-A3B + V4 SOTA LoRA
adapters (r16, 32 layers, LR 1e-5, mlx_lm training).

Methodology : `scripts/eval_humaneval_v4.py`, temp=0,
max_tokens=384, 3→4 space indent normalization, 15 s sandbox
timeout. Fixture downloaded from HuggingFace
`openai_humaneval` split `test`.

## Headline table

| Label | pass@1 | n | gen_s | Δ vs base |
|---|---|---|---|---|
| **base-164** (Qwen3.6-35B-A3B no adapter) | **0.604** (99/164) | 164 | 807 | — |
| cpp-164 | **0.677** (111/164) | 164 | 548 | **+0.073** |
| python-164 | **0.671** (110/164) | 164 | 725 | **+0.067** |
| typescript-164 | **0.665** (109/164) | 164 | 604 | **+0.061** |
| rust-164 | **0.646** (106/164) | 164 | 429 | **+0.043** |
| shell-164 | 0.610 (100/164) | 164 | 993 | +0.006 (bruit) |

## Cross-tests — non-coding adapters on HumanEval-Python

Measure "inter-domain smoothing" : a vanilla LoRA delta applied
to all 256 MoE experts uniformly is expected to degrade
out-of-domain performance. Magnitude of Δ correlates with
"distance from Python".

| Label | pass@1 | Δ vs base | Interpretation |
|---|---|---|---|
| yaml-json-164-cross | 0.579 | **−0.024** | closest to code (syntax-heavy) |
| chat-fr-164-cross | 0.543 | **−0.061** | natural language |
| math-164-cross | 0.494 | **−0.110** | formal reasoning, furthest |

## Verdict

1. **4/5 coding adapters show positive uplift** on HumanEval
   (python +6.7, cpp +7.3, typescript +6.1, rust +4.3). Shell
   is flat (+0.6 pp within ±4 pp noise at N=164).
2. **All 3 cross-tests show negative Δ** confirming the
   inter-domain smoothing pattern. The magnitude stratifies
   exactly as expected : code-like syntax (yaml-json) < FR text
   < formal math.
3. **Router v4 is the bottleneck**. At 46% top-1 accuracy, the
   expected uplift is `0.46 × (+0.06) + 0.54 × (−0.05) ≈ −0.001`
   — essentially zero net benefit under auto-routing. Client-
   driven routing (explicit `model_id` per request) unlocks
   the +6 pp gains.
4. **V4 SOTA adapters are ship-ready for Hugging Face Hub**.
   The HF layout `clemsail/micro-kiki-v4-sota` already carries
   all 35 adapters (uploaded 2026-04-22 morning, confirmed
   public).

## Follow-up (scoped as separate tasks)

- **GSM8K / MATH** : measure math adapter in-domain to close
  the `-11 pp on HumanEval` interpretation. Needs new eval
  runners (not yet in `scripts/`).
- **MBPP / MultiPL-E** : validate that +6 pp on HumanEval
  generalizes across other Python / multi-language benchmarks.
- **Router v4 improvement** : 46% → target 70%+ top-1 via
  better training data or model class. Highest single-lever
  gain available.

## Raw data

Per-problem JSON bundles in this directory, one per run :
`base-164.json`, `cpp-164.json`, `python-164.json`,
`rust-164.json`, `shell-164.json`, `typescript-164.json`,
`chat-fr-164-cross.json`, `math-164-cross.json`,
`yaml-json-164-cross.json`.

Each carries `label`, `base_model`, `adapter`, `n_problems`,
`max_tokens`, `pass@1`, `total_gen_s`, and
`per_problem[]` with full completions + test errors.
