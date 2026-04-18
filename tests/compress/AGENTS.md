<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# tests/compress

## Purpose
Tests Phase XIII tensor-network compression: the CompactifAI framework entry point (`compress_model`), the QTHA (Quantised Tensor Hybrid Adapter) config + parameter estimator, and the TN-Router parameter estimator. These are lightweight structural tests — no real MPS/MPO decomposition is exercised in CI.

## Key Files
| File | Description |
|------|-------------|
| `test_compactifai.py` | `TestCompactifAI.test_compress_model_framework` — writes a fake 1 KiB safetensors blob, runs `compress_model(input, output, bond_dim=32)`, asserts a `CompressionResult` with `bond_dim == 32` and that the output directory exists. `TestQTHA` — `QTHAConfig` defaults (`bond_dim=8`, `q_proj` in `target_modules`), `estimate_qtha_params(3072, 4, 13) < 700_000`, and monotonic scaling with `bond_dim`. `TestTNRouter` — `TNRouterConfig` defaults (32 domains), `estimate_tn_router_params(cfg) < 200_000`. |

## For AI Agents

### Working In This Directory
Run the suite: `uv run python -m pytest tests/compress/ -q`.
- No `@pytest.mark.integration` tests; the "model" is a fake byte blob — no torch weights are loaded.
- `tmp_path` is used for both input and output directories.
- Tests are synchronous — no `pytest.mark.asyncio`.

### Testing Requirements
- `compress_model` contract: returns a `CompressionResult` with `bond_dim` equal to the kwarg, and creates the output directory.
- `estimate_qtha_params` must be strictly increasing in `bond_dim` (with other dims fixed) and well under 700 K params for the default 13-layer / bond_dim=4 / hidden=3072 configuration.
- `estimate_tn_router_params` must stay under 200 K params for default config — the TN-Router is intended to be lightweight.
- `QTHAConfig.target_modules` must contain `q_proj` by default (attention projections only, consistent with the "don't LoRA-tune MoE FFN layers" rule).

### Common Patterns
- Byte-level fake artifacts (`b"\x00" * 1024`) rather than real safetensors — avoids torch/safetensors dependency for structural tests.
- Parameter-count assertions as ceilings (`< 700_000`), not exact equality — leaves room for formula tweaks.

## Dependencies

### Internal
- `src.compress.compactifai` — `compress_model`, `CompressionResult`
- `src.stacks.qtha` — `QTHAConfig`, `estimate_qtha_params`
- `src.routing.tn_router` — `TNRouterConfig`, `estimate_tn_router_params`

### External
- `pytest`

<!-- MANUAL: -->
