# micro-kiki v0.2 — Quantum-Inspired Implementation Plan

**Scope**: Overlay three quantum-inspired techniques onto the v0.1 classical pipeline — tensor-network base compression (CompactifAI), tensor-network adapters (QTHA), and tensor-network routing (MPS-gated router). 100% classical execution.

**Derived from**: `docs/specs/2026-04-15-micro-kiki-v0.2-quantum-inspired.md`.

**Conventions**:
- Each step is ONE shippable unit of work — can be committed independently.
- Steps run sequentially; step 2 depends on step 1, step 3 is independent.
- Acceptance = what "done" looks like (metric / artifact / test).
- Dependencies = which prior step MUST be complete first.
- **No QPU calls** — classical simulation only. See `.ralph-v0.2/guardrails.md`.

---

## Implementation Steps

### Phase Q — Quantum-inspired overlay

1. **CompactifAI base compression**
   - Files to touch: `src/compress/compactifai.py`, `src/compress/__init__.py`, `scripts/compress_base.py`, `tests/test_compactifai.py`, `configs/compactifai.yaml`
   - Implement tensor-network truncation over self-attn (Q/K/V/O projections) and MLP (gate/up/down) layers of Qwen3.5-4B BF16 checkpoint. Use `quimb` for MPS/MPO decomposition and `opt_einsum` for contraction paths.
   - Expose CLI: `uv run python scripts/compress_base.py --bond-dim 32 --input models/qwen3.5-4b/bf16 --output models/qwen3.5-4b-compact/chi32`.
   - Progressive chi schedule: produce chi in {64, 32, 16, 8} artifacts; evaluate each on HumanEval-sample + MMLU-Pro-sample.
   - Commands:
     - `uv add quimb opt_einsum tensornetwork`
     - `uv run pytest tests/test_compactifai.py`
     - `uv run python scripts/compress_base.py --bond-dim 32`
   - Acceptance: chi=32 artifact exists at `models/qwen3.5-4b-compact/chi32/`, on-disk size <= 1 GB, loader smoke test yields non-empty response, accuracy drop on HumanEval-sample < 5 pp vs BF16 baseline.
   - Dependencies: v0.1 story-1 (BF16 base downloaded).

2. **QTHA adapter for stack-02 (reasoning)**
   - Files to touch: `src/stacks/qtha.py`, `src/stacks/__init__.py`, `scripts/train_qtha_stack.py`, `tests/test_qtha.py`, `configs/qtha-stack-02.yaml`
   - Implement `QTHAAdapter(nn.Module)`: decomposes the frozen base weights into a tensor-network factor + a trainable low-bond-dim correction. Parameter count target: ~500 K (vs 2 M MoLoRA rank-16). Training hook compatible with PEFT-style attach/detach.
   - Reuse the v0.1 stack-02 reasoning dataset (`data/dedup/02-reasoning/`). Train on RTX 4090 via Unsloth-compatible path.
   - Eval: win_rate_vs_base on reasoning eval slice; compare to v0.1 MoLoRA stack-02 baseline dumped from main branch.
   - Commands:
     - `uv run pytest tests/test_qtha.py`
     - `uv run python scripts/train_qtha_stack.py --config configs/qtha-stack-02.yaml`
     - `uv run python scripts/eval_stack.py --stack stack-02-reasoning-qtha --compare stack-02-reasoning`
   - Acceptance: adapter saved to `outputs/stacks/stack-02-reasoning-qtha/`, param count logged <= 600 K, win_rate_vs_base >= v0.1 MoLoRA stack-02 - 1 pp, eval report at `results/qtha-vs-molora.md`.
   - Dependencies: step 1 (for compact base path) or v0.1 BF16 base (baseline comparison).

3. **Tensor-network router prototype**
   - Files to touch: `src/routing/tn_router.py`, `src/routing/__init__.py`, `scripts/train_tn_router.py`, `tests/test_tn_router.py`, `configs/tn-router.yaml`
   - Implement `TNRouter(nn.Module)` gating function using matrix product states (MPS) over embedding input; produce per-domain scores via interference amplitudes.
   - Train on v0.1 3-domain router eval set (chat-fr + reasoning + python) already prepared in v0.1 step ~50. Fallback guard: if macro-F1 drops > 2 pp vs sigmoid, flag in report and do not merge into serving path.
   - Commands:
     - `uv run pytest tests/test_tn_router.py`
     - `uv run python scripts/train_tn_router.py --config configs/tn-router.yaml`
     - `uv run python scripts/eval_router.py --router tn --compare sigmoid`
   - Acceptance: `outputs/routers/tn-v0/` artifact exists, comparison report at `results/tn-vs-sigmoid.md`, macro-F1 delta reported (+/- vs sigmoid), decision line ("merge" | "fallback") included.
   - Dependencies: v0.1 3-domain router eval set (available from main branch).
