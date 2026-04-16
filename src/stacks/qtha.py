"""QTHA: Quantum-inspired Tensor Hybrid Adapter.

Decomposes frozen weights into tensor-network factor + trainable low-bond correction.
Target: ~500K params (vs 2M MoLoRA rank-16).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QTHAConfig:
    bond_dim: int = 8
    target_modules: list[str] | None = None

    def __post_init__(self):
        if self.target_modules is None:
            object.__setattr__(self, "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])


def estimate_qtha_params(hidden_dim: int, bond_dim: int, num_layers: int, num_modules: int = 4) -> int:
    """Estimate total trainable parameters for QTHA adapter."""
    # Per module: bond_dim * hidden_dim (correction matrix)
    per_module = bond_dim * hidden_dim
    total = per_module * num_modules * num_layers
    return total
