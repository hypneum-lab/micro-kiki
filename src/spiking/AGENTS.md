<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# spiking — Lossless ANN→SNN conversion (LAS framework)

## Purpose
Implements the LAS (Lossless ANN→SNN) conversion pipeline for the v0.3 neuroscience branch: drop-in replacement of transformer `nn.Linear` / MoE / attention layers with rate-coded spiking equivalents that recover ReLU activations in the large-T limit. Numpy-first on the hot path so unit tests never need torch; torch wrappers live in `las_converter.SpikingLinear` for integration with the live 35B-A3B conversion. This module is the runtime kernel of the v0.3 story — it is *not* a skeleton. Central references: `README.md`, `MODEL_CARD-v0.3.md`, `docs/cookbook-v0.3.md`, `docs/specs/las-conversion-framework.md`. Total ≈1078 lines of production code across 4 files.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | 24 lines. Package header + re-exports: `LASConverter, SpikingLinear, convert_linear, verify_equivalence` from `las_converter`; `LIFNeuron, rate_encode` from `lif_neuron`. Explicit `__all__`. |
| `lif_neuron.py` | 127 lines. `LIFNeuron` dataclass (`threshold=1.0`, `tau=1.0`, `v_init=0.0`) — soft-reset leaky-integrate-and-fire neuron, element-wise over any trailing tensor shape. `simulate(currents)` returns `(spikes, membrane)`; asserts `threshold > 0` and `tau ∈ [0,1]` in `__post_init__`. `rate_encode(activations, timesteps, max_rate=1.0)` broadcasts clipped activations to a `(T, *shape)` per-step current. Pure numpy, float64 membrane, soft-reset subtracts threshold where `fired`. |
| `spikingformer_adapter.py` | 269 lines. Spikingformer drop-in attention wrapper. Lazy-imports `spikingjelly.activation_based.{neuron, layer}` at first use so the package stays importable without the optional dep. Wraps Spikingformer as an attention block compatible with the 35B-A3B conversion pipeline. |
| `las_converter.py` | 658 lines — the core. `SpikingLinear` (frozen dataclass; owns an `LIFNeuron` via `field(init=False)`; forward calls `rate_encode(z, self.timesteps, self.max_rate)` then integrates). `convert_linear(linear, timesteps, max_rate) -> SpikingLinear` module-level factory. `SpikingMLP` (sequential stack). `LASConverter` class: `convert_layer`, `convert_model`, `verify_equivalence` (cosine + MSE vs ANN sample-input forward, configurable `tol`). `SpikingMoELayer` (`router: SpikingLinear` + `experts: list[SpikingLinear]`, absolute-magnitude routing per spec). `SpikingMistralBlock` (full-attention dense block: combined `attn_qkv`, `attn_out`, plus SwiGLU `mlp_gate`/`mlp_up`/`mlp_down`). Module-level `verify_equivalence()` convenience wrapper instantiates a default `LASConverter()`. |

## For AI Agents

### Working In This Directory
- **Do not delete anything here on a "stub" hunch.** All four files are live; an earlier AGENTS.md pass mislabelled them because a cached reader returned only line 1. Always verify with `wc -l` before any deletion PR.
- **Numpy is the contract on the hot path.** Tests import only numpy. Adding a torch dependency at module import will break the `tests/test_lif_neuron.py` / `tests/test_las_converter.py` promise. Torch wrappers must live behind a function that imports torch locally (see `spikingformer_adapter.py` for the lazy pattern).
- **Soft reset, not hard reset.** `LIFNeuron.simulate` decrements the membrane by `threshold` on fire — preserving residual is what makes the rate code lossless in the large-T limit. Do not "simplify" this to `v = 0` on fire.
- **`tau=1.0` is the default for LAS conversion** (pure IF, no leak). Leaky variants (`tau < 1.0`) are allowed but drift from the lossless guarantee; flag any such change in the PR.
- **Timestep budget**: cookbook examples use `timesteps=64` for quick sanity runs and `timesteps=256` for production-quality conversions. `verify_equivalence` tolerance should scale with T — bigger T, tighter tol.
- **MoE conversion uses absolute-magnitude routing** (not softmax top-K) per `docs/specs/las-conversion-framework.md`. `SpikingMoELayer.experts` are standard ReLU `SpikingLinear`s; the router is a `SpikingLinear` too, not a plain torch layer.
- **Shape discipline**: `SpikingLinear` expects `(batch, features)` — `SpikingMistralBlock.forward` flattens before the MLP path. If you extend to sequence-aware layers, flatten/unflatten explicitly rather than relying on numpy broadcasting.
- **Spikingjelly import is lazy**: `spikingformer_adapter.py` imports `spikingjelly.activation_based.{neuron, layer}` inside the first call, not at module top. Keep it that way — it is an optional dep in `pyproject.toml`.

### Testing Requirements
- `tests/test_lif_neuron.py` — dataclass validation, rate-code recovery of ReLU, shape invariants.
- `tests/test_las_converter.py` — `convert_linear` numerical equivalence vs ANN forward (inside configured tol), `SpikingMLP` stack, `LASConverter.verify_equivalence` gating.
- `tests/test_las_mistral.py` — `SpikingMistralBlock` end-to-end for the dense attention + SwiGLU path.
- `tests/test_las_moe.py` — `SpikingMoELayer` router + experts; asserts absolute-magnitude routing behaviour.
- `tests/test_spikingformer.py` — `spikingformer_adapter.py` wiring with the spikingjelly optional import.
- `tests/test_e2e_neuro.py` — multi-layer LAS conversion regression.
- `tests/test_convert_spikingkiki.py` — conversion harness integration used by `scripts/convert_spikingkiki_35b.py` (calls `LASConverter` at lines 216 and 496 of that script).
- Combined test surface ≈1200 lines. Run the full suite after any change to `lif_neuron.simulate` or `SpikingLinear.forward` — both are load-bearing for the lossless guarantee.

### Common Patterns
- `@dataclass` with `field(init=False)` to hide derived members (e.g. the internal `LIFNeuron` inside `SpikingLinear`).
- Pure-numpy forward, optional torch wrapper, spikingjelly lazy import — three-layer strategy so the package is importable in minimal test environments.
- Explicit `__post_init__` validation with `ValueError` for out-of-range LIF params.
- Float64 membrane dynamics for numerical stability, cast back on return where needed.

## Dependencies

### Internal
- `src.spiking.lif_neuron` → `src.spiking.las_converter` (LIFNeuron + rate_encode are the primitives).
- Consumed by `scripts/convert_spikingkiki_35b.py` (lines 216, 496) — the production conversion harness for the v0.3 branch.

### External
- `numpy` — required, hot path.
- `torch` — only used inside `SpikingLinear` torch-compat helpers; do not import at module top.
- `spikingjelly` — optional, lazy-imported by `spikingformer_adapter.py`.

<!-- MANUAL: -->
