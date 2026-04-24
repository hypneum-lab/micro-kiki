# SpikingKiki Phase C — Qwen3.6 MoE incompatibility finding

**Date:** 2026-04-21
**Status:** Phase C **BLOCKED** pending C2 refactor; V4 adapters (Phase D input) all 35 done.

## What we hit

Running `scripts/convert_spikingkiki_35b.py` against Qwen3.6-35B-A3B produces
```
TypeError: object of type 'Qwen3_5MoeExperts' has no len()
```
on every MoE block (40/40 layers).

## Root cause

Qwen3.6-35B-A3B uses HuggingFace Transformers' **fused MoE expert layout**:
- `block.mlp.experts` is a single `Qwen3_5MoeExperts` module (not a ModuleList).
- Fused weights: `experts.gate_up_proj.weight` shape `(num_experts, 2·intermediate, hidden)` and `experts.down_proj.weight` shape `(num_experts, hidden, intermediate)`.
- The old per-expert access pattern `experts[expert_id].gate_proj.weight` (script lines 227, 438) raises because there is no `__getitem__` nor `__len__`.

The script was designed for Qwen3.5 MoE where each expert was a separate `nn.Linear` triplet inside a `ModuleList`. Qwen3.6 collapsed that into fused 3D weight tensors for speed.

## Options for Phase C2

1. **Slice fused tensors** — replace per-expert loop with per-expert slicing:
   ```python
   gate_w = experts.gate_up_proj.weight[expert_id, :intermediate, :]
   up_w   = experts.gate_up_proj.weight[expert_id, intermediate:, :]
   down_w = experts.down_proj.weight[expert_id]
   ```
   Fits LAS per-expert conversion. ~30 lines of diff. Safest.

2. **Convert fused blocks as-is** — keep the 3D tensor, let LAS handle batched conversion. Requires LAS library modifications (treat experts as a batch axis). Riskier.

3. **Defer SpikingKiki to V5** — skip the 35B SpikingKiki conversion until someone refactors the LAS pipeline for fused MoE. Use spikingkiki-27b (already done, Qwen3.5 dense) for proof-of-concept demos.

## Related prior work done

- Phase A: script patched for Qwen3.6 config ingest (commit `8233d16`)
- Phase C1: linear_attn hybrid support for Qwen3.6 (commit `1d59ad7`)
- Phase D prototype: `convert_lora_to_snn.py` with `--self-test` (rel_l2=0.048 — math correct on standalone LoRA)
- Router bias guard: `getattr(module, "bias", None)` patch for missing router bias

## Recommendation

- **Short term**: defer Phase C. V4 adapters (35/35 done) are the product deliverable; SpikingKiki is research/speculative.
- **Medium term**: Phase C2 = implement fused-expert slicing. ~1 day of focused work.
- **Validation**: re-run against Qwen3.6 after C2 lands; all 40 blocks should convert cleanly.

## Artifacts cleaned up

- `/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4/` partial output deleted.
- `/tmp/spiking-c.log` kept for forensic reference.
