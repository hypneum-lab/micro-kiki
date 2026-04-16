from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.base.diff_attention import DifferentialAttention, init_lambda


class TestDifferentialAttention:
    def test_output_shape(self):
        attn = DifferentialAttention(d_model=768, num_heads=12)
        out = attn(torch.randn(2, 16, 768))
        assert out.shape == (2, 16, 768)

    def test_lambda_is_learnable(self):
        attn = DifferentialAttention(d_model=768, num_heads=12)
        assert attn.lambda_param.requires_grad is True

    def test_init_lambda_scales_with_depth(self):
        assert init_lambda(0, 13, 0.8) < init_lambda(12, 13, 0.8)

    def test_nonzero_output(self):
        torch.manual_seed(42)
        attn = DifferentialAttention(d_model=256, num_heads=4)
        assert attn(torch.randn(1, 8, 256)).abs().sum() > 0
