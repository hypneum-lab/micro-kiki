<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# routing

## Purpose
Routing layer that selects which 35B-A3B + LoRA-adapter combination answers a query, composed of four cooperating routers plus a dispatcher. `hybrid_pipeline.HybridPipeline` is the top-level coroutine: it runs the VQC quantum router, falls back to the classical `ModelRouter` if confidence is low, wraps the call in Aeon pre/post-inference memory, and optionally hands multi-candidate responses to the CAMP `Negotiator`. `dispatcher.py` maps the 11-dim sigmoid vector to one of 7 meta-intents (quick-reply, coding, reasoning, creative, research, agentic, tool-use) at zero training cost.

## Key Files
| File | Description |
|------|-------------|
| `router.py` | `MetaRouter` (torch.nn.Module) — sigmoid head over 35 domain outputs (34 niches + "base") and 5 capability outputs (web_search, self_critique_{token,response,task}, deep_eval). Threshold 0.12, max 4 active stacks. Exposes `NICHE_DOMAINS` frozenset and `CAPABILITY_NAMES` as torch-free constants. An 11-output backward-compat path (`num_domains=11`) is still exercised by `tests/routing/test_router_11.py`. |
| `model_router.py` | `ModelRouter.select(query, domain_hint, require_deep)` → `RouteDecision(model_id, adapter, reason)`. Priority: deep-reasoning → qwen480b; niche hint → qwen35b + `stack-<domain>`; code hint (`_CODE_HINTS = {code, coding, debug, firmware, c++, python, rust}`) → devstral; else qwen35b base. |
| `quantum_router.py` | `QuantumRouter` — 4-qubit PennyLane VQC with `AngleEmbedding` + `StronglyEntanglingLayers` (6 layers) + classical linear head to 11 classes. `route(embedding)` returns `RouteDecision`. Parameter-shift-rule training in `train(embeddings, labels, epochs)`. `save/load` to `.npz`. Raises `ImportError` at init if PennyLane missing. |
| `tn_router.py` | `TNRouterConfig` + `estimate_tn_router_params` — MPS-chain tensor-network router scaffolding; not yet a full module. |
| `hybrid_pipeline.py` | `HybridPipeline.route_and_infer(query, context)` orchestrates quantum → classical fallback → Aeon pre-inject → stub inference → optional Negotiator → Aeon post-inject. `HybridPipelineConfig` toggles quantum/memory/negotiator with a `quantum_confidence_threshold` (default 0.7). Ships `_stub_infer` — replace with MLX/vLLM dispatch for production. |
| `dispatcher.py` | `MetaIntent` enum + `load_intent_mapping` (reads `configs/meta_intents.yaml`) + `validate_mapping` (exactly-one-bucket-per-domain) + `dispatch(router_logits, mapping)` → `DispatchResult`. Training-free. |
| `CLAUDE.md` | Project notes — 35 sigmoid outputs (34 niches + base), threshold 0.12 general / 0.20 chat-floor, max 4 active stacks, retest every 4 stacks. |

## For AI Agents

### Working In This Directory
- **Respect the 2026-04-17 expansion**: `NICHE_DOMAINS` is a 34-element frozenset. Index 34 is the implicit "base" output, total 35. Legacy 32-domain and 11-output modes remain reachable via `MetaRouter(num_domains=32|11)` for backward compat.
- **Max 4 active stacks** is enforced in `MetaRouter.get_active_domains` via `topk` — honour the project-level "Don't route > 4 stacks simultaneously" rule.
- **Sigmoid, not softmax**: domains are not mutually exclusive (`CLAUDE.md` explicitly forbids softmax in `router.py`). Only the QuantumRouter's classical head uses softmax because it outputs a single class.
- **Confidence encoding contract**: `QuantumRouter.route` stamps `"conf=0.xxx"` inside the `RouteDecision.reason` string. `hybrid_pipeline._extract_confidence` parses exactly 5 chars after `conf=`. Don't change the format without updating both sides.
- **Adapter path convention**: `RouteDecision.adapter` is `"stack-<domain>"` or `None`. `HybridPipeline` strips the `stack-` prefix before calling `AeonServingHook.post_inference(domain=...)`.
- **Torch is optional** for `router.py` constants — `NICHE_DOMAINS` and `CAPABILITY_NAMES` import without torch; `MetaRouter` instantiation requires it. Keep that split.
- **Do not train router and stacks simultaneously** (CLAUDE.md anti-pattern). Retest the router after every 4 new stacks.

### Testing Requirements
- `tests/routing/test_hybrid_pipeline.py` — full coroutine path, confidence fallback, memory injection counts, negotiator wiring.
- `tests/routing/test_quantum_router.py` — circuit forward, parameter-shift gradient step, save/load roundtrip.
- `tests/routing/test_model_router.py` — priority ordering on representative queries.
- `tests/routing/test_router_11.py` — 11-output MetaRouter acceptance.
- `tests/routing/test_router_37.py` — legacy 37-output path (kept green for migration).
- `tests/test_dispatcher.py` — YAML mapping validation, seven-bucket assignment.

### Common Patterns
- `@dataclass(frozen=True)` configs: `HybridPipelineConfig`, `QuantumRouterConfig`, `TNRouterConfig`, `RouteDecision`.
- `PipelineResult` is a regular dataclass (mutable) because it aggregates telemetry (latency, memories_injected).
- Async-first for `HybridPipeline.route_and_infer` and `Negotiator.negotiate`.
- `TYPE_CHECKING` guards for heavy-circular imports (`QuantumRouter`, `AeonServingHook`, `Negotiator` inside hybrid_pipeline).
- Loguru / stdlib logging per CLAUDE.md — no `print()`.

## Dependencies

### Internal
- `src.routing.router.NICHE_DOMAINS` is reused by `src.routing.quantum_router`, `src.routing.model_router`, and `src.routing.hybrid_pipeline._domain_hint_from_query`.
- `src.routing.model_router.RouteDecision` is the shared contract returned by both classical and quantum routers.
- `src.routing.hybrid_pipeline` wires in `src.serving.aeon_hook.AeonServingHook` and `src.cognitive.negotiator.Negotiator`.
- Consumed by `src.serving.*`, `src.ralph.autonomous` (indirectly via inference), and scripts under `scripts/run_eval_*.py`.

### External
- `pennylane` (quantum_router, optional — module degrades to ImportError at construction if absent).
- `torch`, `torch.nn` (MetaRouter only).
- `numpy` (quantum circuit I/O, hybrid pipeline embedding hash).
- `pyyaml` (dispatcher intent-map loader).
- stdlib: `asyncio`, `logging`, `uuid`, `time`, `pathlib`, `dataclasses`, `enum`.

<!-- MANUAL: -->
