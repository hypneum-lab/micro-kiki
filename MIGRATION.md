# Migration Guide — micro-kiki v0.2

## Breaking Changes from v0.1

### Router extended to 37 outputs
- 32 domain outputs + 5 capability flags (web_search, self_critique_token, self_critique_response, self_critique_task, deep_eval)
- Threshold config in `configs/capabilities.yaml`

### OPLoRA default for stacks >= 04
- Stacks 01-03 use PiSSA initialization
- Stacks 04+ use OPLoRA (orthogonal projection) for forgetting prevention
- Config key: `init_lora_weights: oplora`

### DiffAttn fork as default base
- Base model: `models/qwen3.5-4b-diffattn/` (13 full-attn layers modified)
- Fallback: `models/qwen3.5-4b/bf16` if DiffAttn degrades perplexity > 3%
- Rollback documented in `docs/specs/diffattn-integration.md`

### Adapter format
- MoLoRA: 4 experts per projection, rank 16, top-2 softmax routing
- Saved as PEFT-compatible .safetensors in `outputs/stacks/`
- Compatible with vLLM dynamic LoRA loading

### New cognitive layer
- Aeon Memory Palace: `src/memory/` (Atlas vector + Trace graph)
- Negotiator: `src/cognitive/` (CAMP + Catfish + adaptive judge)
- Anti-bias: KnowBias double-application + RBD runtime detector

### Quantum-inspired overlay (optional)
- CompactifAI: tensor-network base compression (`src/compress/`)
- QTHA: hybrid adapter pilot (`src/stacks/qtha.py`)
- TN-router: MPS-based routing (`src/routing/tn_router.py`)

## Serving
- vLLM: `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, port 8100
- MLX: adapter switching via `mlx-lm`, port 8100
- Deploy files staged in `deploy/` (not auto-installed)

## Config freeze
All configs in `configs/` are hash-locked at v0.2.0 tag.
