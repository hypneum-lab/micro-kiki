# src/routing/ — meta-router + dispatcher

Selects which LoRA stacks to activate per query, then maps the active-stack set to a high-level intent.

## Modules

- `router.py` — canonical sigmoid meta-router (35 outputs, not softmax).
- `model_router.py` — lightweight prod wrapper used by serving.
- `dispatcher.py` — training-free YAML mapping from router output to 7 meta-intents (config at `configs/dispatcher_capabilities.yaml` + `configs/meta_intents.yaml`).
- `hybrid_pipeline.py` — glue between router → dispatcher → adapter activation.
- `quantum_router.py`, `tn_router.py` — experimental VQC / tensor-network routers (benchmarked via `scripts/benchmark_quantum_router.py`). Do not ship these to prod without an A/B win.

## Thresholds

- General: 0.12. Chat-mode floor: 0.20.
- Thresholds **must** be loaded from config, never hardcoded.
- A stack fires only if its sigmoid crosses threshold AND it fits in the 4-active budget (top-k by score).

## Testing discipline

- Retest the router after every 4 new stacks — threshold drift accumulates.
- Golden queries per domain; flag regressions >1% as blocking.
- Never train router and stacks in the same job — the router must see the final adapters.

## Anti-patterns (routing-specific)

- Don't use softmax — domains overlap (e.g. `embedded` + `stm32` + `platformio`).
- Don't exceed 4 simultaneously active stacks (VRAM + adapter interference).
- Don't hardcode thresholds or the 35-domain count — both come from configs.
- Don't route through the experimental VQC/TN routers by default.
