<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# tests/routing

## Purpose
Verifies the routing layer of micro-kiki — including the classical `ModelRouter` (model + adapter selection from a prompt), the neural `MetaRouter` (11-domain and legacy 37-output variants), the quantum VQC-based `QuantumRouter` (PennyLane simulator), and the triple-hybrid `HybridPipeline` that composes quantum routing, Aeon memory injection, and CAMP negotiation. These tests lock in router thresholds, fallback semantics, and stack selection invariants as prescribed in `tests/CLAUDE.md`.

## Key Files
| File | Description |
|------|-------------|
| `test_model_router.py` | Unit tests for `ModelRouter.select()`: verifies base routing to `qwen35b`, deep-reasoning override to `qwen480b`, niche `domain_hint` to `stack-<domain>`, code-domain fallback to `devstral`, and `RouteDecision` frozen semantics. |
| `test_router_11.py` | Tests `MetaRouter` with 11 outputs (10 niche + base): confirms `NICHE_DOMAINS` size, that `chat-fr` is excluded, `kicad-dsl` is present, base fallback below threshold 0.12, and legacy 32-domain output still works. Skipped if torch missing. |
| `test_router_37.py` | Tests the 37-output variant (32 domains + 5 capabilities): sigmoid range, domain/capability split shape, `max_active=4` cap, and threshold-driven active-capability dict. |
| `test_quantum_router.py` | Tests `QuantumRouter` (VQC, PennyLane simulator): config defaults (4 qubits, 6 layers, 11 classes), circuit output shape and `[-1, 1]` PauliZ range, `route()` producing valid `RouteDecision`, 10-epoch training loss decrease, save/load `.npz` roundtrip, and `ImportError` when PennyLane absent. Uses `@pennylane_required` skip marker. |
| `test_hybrid_pipeline.py` | Tests `HybridPipeline` composition: full-pipeline with quantum + memory + negotiator all active, low-confidence quantum fallback to classical, per-component disable flags, `route_only()` (sync), and default-disabled negotiator. Uses `_make_quantum_router` / `_make_model_router` / `_make_aeon_hook` / `_make_negotiator` mock factories. |

## For AI Agents

### Working In This Directory
Run just this suite: `uv run python -m pytest tests/routing/ -q`.
- `test_quantum_router.py` auto-skips any test decorated `@pennylane_required` if `pennylane` is not importable (install with `uv add pennylane`). `test_import_without_pennylane_raises_import_error` runs unconditionally — it monkeypatches `sys.modules["pennylane"] = None` and toggles `qr_mod._PENNYLANE_AVAILABLE`.
- `test_router_11.py` and `test_router_37.py` use `pytest.importorskip("torch")` at module load.
- No tests in this directory are `@pytest.mark.integration` — everything runs on classical CPU simulators and mocks.
- `HybridPipeline` tests rely on `AsyncMock` for `negotiate`, `MagicMock` for `QuantumRouter.route` / `ModelRouter.select` / `AeonServingHook`.

### Testing Requirements
- `ModelRouter` invariants: `require_deep=True` must override any `domain_hint`; unknown `domain_hint` must degrade silently to `qwen35b` base; `RouteDecision` is frozen (mutation raises).
- `MetaRouter` invariants: niche set has exactly 10 elements; base fallback engages when all sigmoid outputs < 0.12; capability thresholds produce a dict of booleans.
- `QuantumRouter` invariants: PauliZ expectations in `[-1, 1]`; weight tensor shape `(n_layers, n_qubits, 3)`; linear head shapes `(n_qubits, n_classes)` and `(n_classes,)`; training loss non-increasing (tolerance 0.5); save/load preserves routing decision.
- `HybridPipeline` invariants: `quantum_used=True` iff confidence ≥ `quantum_confidence_threshold`; `memories_injected` counts exactly the `[Memory]` lines pre-pended to the prompt; negotiator only runs when `use_negotiator=True`.

### Common Patterns
- `pytest-asyncio` with `asyncio_mode = "auto"` — async tests use bare `async def` without the `@pytest.mark.asyncio` decorator for `test_hybrid_pipeline.py` where the decorator is also explicitly present.
- Module-scoped fixtures for heavy quantum state (`router`, `config` in `test_quantum_router.py`).
- Mock factory functions (`_make_*`) rather than fixtures to allow per-test parameterization of confidence and domain.
- Frozen-dataclass mutation test pattern: `with pytest.raises(Exception): obj.field = ...`.

## Dependencies

### Internal
- `src.routing.model_router` — `ModelRouter`, `RouteDecision`
- `src.routing.router` — `MetaRouter`, `NICHE_DOMAINS`
- `src.routing.quantum_router` — `QuantumRouter`, `QuantumRouterConfig`, `_ALL_DOMAINS`
- `src.routing.hybrid_pipeline` — `HybridPipeline`, `HybridPipelineConfig`, `PipelineResult`, `_extract_confidence`, `_count_memory_lines`
- `src.cognitive.negotiator` — `NegotiationResult` (imported for mock typing)

### External
- `pytest`, `pytest-asyncio`
- `numpy`
- `torch` (via `importorskip`)
- `pennylane` (optional, guarded by `_pennylane_available`)

<!-- MANUAL: -->
