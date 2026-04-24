"""Brainstacks V4: implicit null-space projection in LoRA parameter space.

Projects training gradients orthogonally to the subspace spanned by
previously frozen LoRA adapters.  Operates in LoRA param space
(dim = in_features * rank + rank * out_features) not weight space
(in_features * out_features), keeping projectors at ~2 KB per module.

Reference: Brainstacks (arXiv 2604.01152), adapted for vanilla LoRA
on Qwen3.6-35B-A3B MoE (17 module kinds, 3D switch_mlp experts).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def collect_lora_vectors(
    adapter: dict[str, np.ndarray],
    layer: int,
    module: str,
) -> np.ndarray:
    """Extract flattened LoRA parameter vector(s) for one module in one layer.

    For 2D modules: returns shape (param_dim,)
    For 3D switch_mlp modules (256 experts): returns shape (256, per_expert_dim)
    """
    prefix = f"layers.{layer}.{module}"
    # Handle key formats: may have "language_model.model." prefix
    a_key = None
    b_key = None
    for k in adapter:
        if k.endswith(f"{prefix}.lora_a") or k.endswith(f".{prefix}.lora_a"):
            a_key = k
        if k.endswith(f"{prefix}.lora_b") or k.endswith(f".{prefix}.lora_b"):
            b_key = k
    if a_key is None or b_key is None:
        raise KeyError(f"LoRA keys not found for {prefix}")

    A = adapter[a_key]
    B = adapter[b_key]

    if A.ndim == 3:
        # 3D: (num_experts, rank, dim) — flatten per expert
        n_experts = A.shape[0]
        vectors = []
        for e in range(n_experts):
            vec = np.concatenate([A[e].flatten(), B[e].flatten()])
            vectors.append(vec)
        return np.stack(vectors)  # (n_experts, per_expert_dim)
    else:
        # 2D: (in, rank) and (rank, out)
        return np.concatenate([A.flatten(), B.flatten()])


def collect_all_frozen_vectors(
    adapter_paths: list[str | Path],
    layer: int,
    module: str,
) -> np.ndarray:
    """Load multiple frozen adapters and stack their parameter vectors.

    Returns:
        2D modules: (n_frozen, param_dim)
        3D modules: (n_frozen * 256, per_expert_dim)
    """
    from safetensors.numpy import load_file

    vectors = []
    for path in adapter_paths:
        adapter = load_file(str(Path(path) / "adapters.safetensors"))
        vec = collect_lora_vectors(adapter, layer, module)
        if vec.ndim == 1:
            vectors.append(vec)
        else:
            # 3D: stack all experts from this adapter
            vectors.extend(vec)
    return np.array(vectors, dtype=np.float32)


def build_projector(
    frozen_vectors: np.ndarray,
    top_k: int = 32,
    n_oversampling: int = 10,
    n_iter: int = 3,
) -> np.ndarray:
    """Compute top-K principal directions via randomized SVD.

    Args:
        frozen_vectors: (n_frozen, param_dim) stacked parameter vectors
        top_k: number of directions to protect

    Returns:
        V_keep: (min(top_k, n_frozen), param_dim) — orthonormal rows
    """
    n_frozen, param_dim = frozen_vectors.shape
    effective_k = min(top_k, n_frozen, param_dim)

    if effective_k == 0:
        return np.zeros((0, param_dim), dtype=np.float32)

    # Randomized SVD (Halko-Martinsson-Tropp)
    rng = np.random.default_rng(42)
    k_hat = min(effective_k + n_oversampling, param_dim)
    Omega = rng.standard_normal((param_dim, k_hat)).astype(np.float32)

    Y = frozen_vectors @ Omega  # (n_frozen, k_hat)

    # Power iteration for better approximation
    for _ in range(n_iter):
        Y = frozen_vectors @ (frozen_vectors.T @ Y)

    Q, _ = np.linalg.qr(Y)  # (n_frozen, k_hat)
    B = Q.T @ frozen_vectors  # (k_hat, param_dim)
    _, S, Vt = np.linalg.svd(B, full_matrices=False)

    V_keep = Vt[:effective_k]  # (effective_k, param_dim)

    logger.debug(
        "null-space projector: %d frozen → %d directions (top singular: %.4f)",
        n_frozen, effective_k, S[0] if len(S) > 0 else 0,
    )
    return V_keep.astype(np.float32)


def project_gradient(
    grad: np.ndarray,
    V_keep: np.ndarray,
) -> np.ndarray:
    """Project gradient orthogonally to the protected subspace.

    projected = grad - V_keep^T @ (V_keep @ grad)

    This is the implicit form — no dense P matrix needed.
    """
    if V_keep.shape[0] == 0:
        return grad
    coeffs = V_keep @ grad  # (K,)
    return grad - V_keep.T @ coeffs


class NullSpaceRegistry:
    """Per-layer, per-module projector registry for a training run.

    Usage:
        registry = NullSpaceRegistry.from_frozen_adapters(
            adapter_paths=["path/to/stack-0", "path/to/stack-1"],
            layers=range(32),
            modules=MODULE_KINDS,
            top_k=32,
        )
        # During training:
        projected_grad = registry.project(layer=5, module="self_attn.q_proj", grad=grad_array)
    """

    def __init__(self) -> None:
        self._projectors: dict[tuple[int, str], np.ndarray] = {}  # (layer, module) -> V_keep

    @classmethod
    def from_frozen_adapters(
        cls,
        adapter_paths: list[str | Path],
        layers: range,
        modules: list[str],
        top_k: int = 32,
    ) -> NullSpaceRegistry:
        registry = cls()
        if not adapter_paths:
            logger.info("No frozen adapters — null-space projection disabled")
            return registry

        total = len(list(layers)) * len(modules)
        built = 0
        for layer in layers:
            for module in modules:
                try:
                    frozen = collect_all_frozen_vectors(adapter_paths, layer, module)
                    if frozen.shape[0] > 0:
                        V_keep = build_projector(frozen, top_k=top_k)
                        registry._projectors[(layer, module)] = V_keep
                        built += 1
                except KeyError:
                    pass  # module not present in this layer type

        logger.info(
            "Built %d/%d null-space projectors from %d frozen adapters",
            built, total, len(adapter_paths),
        )
        return registry

    def project(self, layer: int, module: str, grad: np.ndarray) -> np.ndarray:
        """Project a gradient for a specific (layer, module)."""
        V_keep = self._projectors.get((layer, module))
        if V_keep is None:
            return grad
        if grad.ndim == 1:
            return project_gradient(grad, V_keep)
        # 3D: project each expert independently
        result = np.empty_like(grad)
        for e in range(grad.shape[0]):
            result[e] = project_gradient(grad[e], V_keep)
        return result


# The 17 module kinds in Qwen3.6-35B-A3B V4 adapters
MODULE_KINDS: list[str] = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "linear_attn.in_proj_a", "linear_attn.in_proj_b", "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z", "linear_attn.out_proj",
    "mlp.gate", "mlp.shared_expert_gate",
    "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj",
    "mlp.switch_mlp.gate_proj", "mlp.switch_mlp.up_proj", "mlp.switch_mlp.down_proj",
]
