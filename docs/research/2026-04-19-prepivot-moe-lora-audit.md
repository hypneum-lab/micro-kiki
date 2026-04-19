# Pre-pivot MoE-LoRA audit — 2026-04-19

Empirical evidence that the custom `_moe_lora` training path in
`stacks-v3-r16/` (pre-2026-04-16 pivot) silently failed to train
its LoRA `B` matrices.

## Method

1. Extended `measure_forgetting` to recognize MoE-LoRA keys
   (`...experts.N.lora_{a,b}`).
2. Ran pairwise forgetting sweep across all **35 pre-pivot adapters**
   in `/Users/clems/KIKI-Mac_tunner/output/micro-kiki/stacks-v3-r16/`
   (1 190 ordered pairs).
3. Every pair reported mean angle **0.00°** — and so did every
   per-module angle.
4. Triaged: loaded 6 adapters (chat-fr, cpp, python, reasoning, power,
   rust) and measured individual tensor norms.

## Finding

| Tensor class   | Count per adapter | Max norm | Non-zero count |
|----------------|------------------:|---------:|---------------:|
| `lora_a`       | 512               | ~4.04    | 512            |
| `lora_b`       | 512               | **0.000000** | **0**      |
| `router_w*.weight` | 256           | ~4.63    | 256            |

**`lora_b` tensors are exactly zero** on every pre-pivot adapter
tested. The delta `A @ B = A @ 0 = 0` on every module-layer, which
is why every forgetting pair trivially reports 0°.

The custom MoE-LoRA router heads (`router_w1/w2.{weight,bias}`) did
receive gradients — only the per-expert LoRA pairs were skipped.

## Root cause (hypothesis)

The pre-pivot `_moe_lora` layer likely used a routing formulation
that gated the gradient path to the expert LoRA pairs behind the
router's top-k decision, and the backward path never flowed through
the `B` matrices (possibly because `B` starts at zero and the
routed forward contribution is then `A·x @ B = A·x @ 0 = 0`,
yielding zero gradient wrt `B`). The routers alone could still
update because their loss path was independent of `B`.

Full root-cause analysis would require reading the pre-pivot
`src/stacks/moe_lora.py` implementation (now archived). This
document stops at the empirical observation.

## Consequence

1. The ~35 pre-pivot adapters are **dead weights** — their
   contribution to the base model is exactly zero.
2. The **pivot to standard LoRA on 2026-04-16 was necessary, not
   cosmetic**. Post-pivot sanity: `lora-qwen36-35b/chat-fr` has
   204/204 `lora_b` non-zero (max norm 0.9976). Standard PEFT/MLX
   LoRA paths work.
3. Our forgetting gate pipeline is sound — it correctly reports
   0° when fed zero-delta adapters.
4. The full pre-pivot matrix is kept at
   `results/forgetting-matrix-prepivot.json` (with a `_diagnostic`
   block) as a reproducible artifact of the bug.

## Not in scope

- Re-training any pre-pivot stack (dead end — the pivot supersedes).
- Forensic reading of archived `scripts/legacy/*moe_lora*` — the
  lineage is already clear.
- Fixing the tool to "handle" zero deltas specially — zeros are
  already a pass-through (0°, no crash); adding a warning in the
  CLI might help future investigators but doesn't change the math.

## Follow-ups (optional)

- Add a sentinel check in `scripts/measure_forgetting.py`: if any
  input adapter has **all** `lora_b` norms at machine-zero, emit a
  `loguru.warning("adapter appears untrained — all lora_b zero")`
  before computing angles. Cheap safety net.
- Consider a `validate_adapter_health.py` validator that scans any
  newly-produced adapter for this class of training-path failure
  (at least one non-zero `lora_b`).
