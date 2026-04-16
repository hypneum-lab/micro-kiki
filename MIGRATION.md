# Migration Guide — micro-kiki v0.2

## Breaking Changes from v0.1

### Base Model
- v0.1: N/A (no prior release)
- v0.2: Qwen3.5-4B with Differential Attention on full-attention layers

### Adapter Format
- Standard LoRA rank 16 via PEFT
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Init: PiSSA (default), OPLoRA (stacks >= 04)
- Adapters saved as PEFT `adapter_model.safetensors`

### Router Protocol
- Sigmoid meta-router: 32 domain outputs, threshold 0.12
- Chat floor: 0.20 (always activates chat-fr stack)
- Max 4 stacks active simultaneously
- HTTP endpoint: `POST /v1/route` with `{"prompt": "..."}` → `{"stacks": [...], "meta_intent": "..."}`

### Serving
- vLLM (kxkm-ai): port 8100, `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`
- MLX (Studio): port 8200, adapter hot-swap via restart (~200ms)
- Aeon memory hook: pre-inference recall + post-inference write

### Memory
- Aeon dual-index: Atlas (vector) + Trace (graph)
- Backends: native (dev), Qdrant + Neo4j (production)
- Compression daemon: episodes > 30 days auto-compressed

### Cognitive Layer
- Negotiator: CAMP + Catfish, adaptive judge
- Anti-bias: KnowBias double-application + RBD runtime
- Dispatcher: 7 meta-intents (training-free YAML)

### Quantum-Inspired (Phase XIII)
- CompactifAI: structured pruning (classical simulator)
- QTHA: quantum-inspired adapter (classical fallback)
- Tensor-network router: bond dimension routing

## Upgrade Path

1. Download Qwen3.5-4B base
2. Apply DiffAttn fork: `uv run python scripts/fork_qwen_diffattn.py`
3. Train stacks in curriculum order
4. Deploy with service units in `deploy/`

## Known Limitations

- Max 4 concurrent stacks (VRAM budget)
- Forgetting check required after each stack
- No QLoRA support for MoE base variants
- Phase XIII quantum techniques are classical-only (no QPU)
