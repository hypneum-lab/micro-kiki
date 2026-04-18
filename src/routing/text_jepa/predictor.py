"""Tiny predictor — MLP 128 → 32 → 128. Kept small per V-JEPA 2 asymmetry recipe."""
from __future__ import annotations

import torch
from torch import nn


class Predictor(nn.Module):
    """Asymmetric narrow-bottleneck MLP predictor."""

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., latent_dim) → (..., latent_dim)."""
        return self.net(x)
