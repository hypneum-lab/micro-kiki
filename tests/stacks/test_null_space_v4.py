"""Tests for V4 null-space projector in LoRA parameter space."""
import numpy as np
import pytest


def test_collect_lora_vectors_from_single_adapter():
    """Load one adapter's A,B matrices and flatten to parameter vector."""
    from src.stacks.null_space_v4 import collect_lora_vectors
    # Create fake adapter dict mimicking V4 format
    adapter = {
        "layers.0.self_attn.q_proj.lora_a": np.random.randn(2048, 16).astype(np.float32),
        "layers.0.self_attn.q_proj.lora_b": np.random.randn(16, 4096).astype(np.float32),
    }
    vectors = collect_lora_vectors(adapter, layer=0, module="self_attn.q_proj")
    # Vector = concat(flatten(A), flatten(B))
    expected_dim = 2048 * 16 + 16 * 4096
    assert vectors.shape == (expected_dim,)


def test_collect_lora_vectors_3d_switch_mlp():
    """3D switch_mlp: flatten per-expert, return (256, param_dim)."""
    from src.stacks.null_space_v4 import collect_lora_vectors
    adapter = {
        "layers.0.mlp.switch_mlp.gate_proj.lora_a": np.random.randn(256, 16, 2048).astype(np.float32),
        "layers.0.mlp.switch_mlp.gate_proj.lora_b": np.random.randn(256, 512, 16).astype(np.float32),
    }
    vectors = collect_lora_vectors(adapter, layer=0, module="mlp.switch_mlp.gate_proj")
    # Per-expert: concat(flatten(A[e]), flatten(B[e])) for e in 256
    per_expert_dim = 16 * 2048 + 512 * 16
    assert vectors.shape == (256, per_expert_dim)


def test_build_projector_orthogonal():
    """V_keep directions are orthogonal to projected gradient."""
    from src.stacks.null_space_v4 import build_projector, project_gradient
    # 3 frozen stacks, param_dim=64
    frozen_vectors = np.random.randn(3, 64).astype(np.float32)
    V_keep = build_projector(frozen_vectors, top_k=2)
    assert V_keep.shape == (2, 64)

    # Project a random gradient
    grad = np.random.randn(64).astype(np.float32)
    projected = project_gradient(grad, V_keep)

    # projected should be orthogonal to V_keep rows
    for i in range(V_keep.shape[0]):
        dot = np.abs(np.dot(projected, V_keep[i]))
        assert dot < 1e-5, f"Not orthogonal: dot={dot}"

    # projected should not be zero (grad had components outside V_keep span)
    assert np.linalg.norm(projected) > 0.01


def test_build_projector_preserves_orthogonal_component():
    """If grad is already orthogonal to frozen directions, projection is identity."""
    from src.stacks.null_space_v4 import build_projector, project_gradient
    # Frozen in directions [1,0,0,...] and [0,1,0,...]
    frozen = np.eye(64, dtype=np.float32)[:2]  # first 2 basis vectors
    V_keep = build_projector(frozen, top_k=2)

    # Grad purely in direction [0,0,1,0,...] — orthogonal to frozen
    grad = np.zeros(64, dtype=np.float32)
    grad[2] = 1.0
    projected = project_gradient(grad, V_keep)
    np.testing.assert_allclose(projected, grad, atol=1e-5)


def test_project_gradient_zero_when_fully_in_frozen_span():
    """If grad lies entirely in the frozen subspace, projection is zero."""
    from src.stacks.null_space_v4 import build_projector, project_gradient
    frozen = np.eye(64, dtype=np.float32)[:2]
    V_keep = build_projector(frozen, top_k=2)

    grad = np.zeros(64, dtype=np.float32)
    grad[0] = 3.0
    grad[1] = -2.0  # entirely in span of frozen dirs 0,1
    projected = project_gradient(grad, V_keep)
    assert np.linalg.norm(projected) < 1e-5
