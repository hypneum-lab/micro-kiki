"""Differential Attention (arxiv 2410.05258, ICLR 2025).

Applied only to the 13 full_attention layers of Qwen3.5-4B.
scores = softmax(Q1*K1) - lambda * softmax(Q2*K2)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_lambda(layer_idx: int, num_layers: int, reinit_lambda: float = 0.8) -> float:
    return reinit_lambda * ((layer_idx + 1) / num_layers)


class DifferentialAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, layer_idx: int = 0,
                 num_layers: int = 13, reinit_lambda: float = 0.8) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q1_proj = nn.Linear(d_model, d_model, bias=False)
        self.k1_proj = nn.Linear(d_model, d_model, bias=False)
        self.q2_proj = nn.Linear(d_model, d_model, bias=False)
        self.k2_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        lam_init = init_lambda(layer_idx, num_layers, reinit_lambda)
        self.lambda_param = nn.Parameter(torch.full((num_heads,), lam_init))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q1, k1 = reshape(self.q1_proj(x)), reshape(self.k1_proj(x))
        q2, k2 = reshape(self.q2_proj(x)), reshape(self.k2_proj(x))
        v = reshape(self.v_proj(x))
        scale = math.sqrt(self.head_dim)

        attn1 = F.softmax(torch.matmul(q1, k1.transpose(-2, -1)) / scale, dim=-1)
        attn2 = F.softmax(torch.matmul(q2, k2.transpose(-2, -1)) / scale, dim=-1)

        lam = self.lambda_param.view(1, self.num_heads, 1, 1)
        out = torch.matmul(attn1 - lam * attn2, v)
        return self.o_proj(out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model))
