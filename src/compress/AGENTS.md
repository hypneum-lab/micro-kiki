<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# compress — CompactifAI Tensor-Network Compression

## Purpose
Compresses the Qwen3.5-35B-A3B base via MPS/MPO (matrix product state / operator) decomposition over attention and MLP weight matrices. Complements Q4_K_M quantisation: quantisation cuts bit-width, tensor networks cut rank. Target is serving-side memory reduction without dropping below the project Q4 quality floor. Framework-only on this repo — actual compression runs on the GPU machine.

## Key Files
| File | Description |
|------|-------------|
| `compactifai.py` | `CompressionResult` dataclass (input/output size, ratio, bond_dim, output path), `compress_layer_tn(weight_matrix, bond_dim=32)` — single-layer MPS truncation via `quimb.tensor.MatrixProductState.from_dense` + `.compress(max_bond=bond_dim)`, and `compress_model(input_dir, output_dir, bond_dim=32)` — directory-level entry point that estimates sizes and writes a result struct. Requires `quimb` + `opt_einsum`; raises `RuntimeError("quimb required: uv add quimb opt_einsum")` otherwise. |
| `__init__.py` | Empty (`from __future__ import annotations`). |

## For AI Agents

### Working In This Directory
- **Full compression runs on the GPU machine** (kxkm-ai RTX 4090). `compress_model()` in this repo is a framework shim with a size-estimate placeholder — the actual MPS truncation math needs torch + quimb with real weights loaded.
- **Don't stack compression and aggressive quantisation blindly**: the Q4_K_M floor from project `Don't` applies at the serving stage. MPS compression at `bond_dim=32` on top of Q4 is the target combo; drop to bond_dim < 16 only with a forgetting check + MAP bench sanity pass.
- **Bond dimension is the primary knob**: `compression_ratio = 128 / bond_dim` per the current placeholder heuristic. Real ratios depend on per-layer rank structure; validate empirically per stack.
- **Float32 compute for the decomposition** — `weight_matrix.detach().cpu().numpy().astype(np.float32)`. Do not cast the MPS factors back to BF16 before truncation; rounding dominates the bond-dim truncation error.
- **quimb is optional at import time**: the runtime `ImportError` guard means this module can be imported on machines without the dependency. Don't hoist the import to module top.

### Testing Requirements
- Mirror tests in `tests/compress/`. Unit-test `compress_layer_tn()` on a small rank-deficient matrix (known truncation floor).
- Full-model compression is not unit-testable — integration test gated behind a `@pytest.mark.gpu` or `@pytest.mark.slow` marker.

### Common Patterns
- `@dataclass(frozen=True)` for results.
- Lazy third-party imports behind `try/except ImportError` with actionable error text (`uv add quimb opt_einsum`).
- Log input size + bond_dim at INFO before doing work (`logger.info("Compressing %s (%.0f MB) with bond_dim=%d", ...)`).

## Dependencies

### Internal
- Reads base weights through `src/base/loader.py`.
- Compressed output is consumed by `src/orchestrator/` at load time.

### External
- `quimb.tensor.MatrixProductState` (MPS decomposition + compress).
- `opt_einsum` (tensor contraction backend for quimb).
- `numpy`, `torch` (at runtime on the GPU machine).

<!-- MANUAL: -->
