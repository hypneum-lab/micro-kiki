from __future__ import annotations

import torch
import torch.nn as nn

CAPABILITY_NAMES = [
    "web_search",
    "self_critique_token",
    "self_critique_response",
    "self_critique_task",
    "deep_eval",
]


class MetaRouter(nn.Module):
    """Sigmoid meta-router with domain + capability outputs."""

    def __init__(
        self,
        input_dim: int = 768,
        num_domains: int = 32,
        num_capabilities: int = 5,
    ) -> None:
        super().__init__()
        self.num_domains = num_domains
        self.num_capabilities = num_capabilities
        total = num_domains + num_capabilities
        self.linear = nn.Linear(input_dim, total)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.linear(x))

    def get_domains(self, output: torch.Tensor) -> torch.Tensor:
        return output[:, : self.num_domains]

    def get_capabilities(self, output: torch.Tensor) -> torch.Tensor:
        return output[:, self.num_domains :]

    def get_active_domains(
        self,
        output: torch.Tensor,
        threshold: float = 0.12,
        max_active: int = 4,
    ) -> list[list[int]]:
        domains = self.get_domains(output)
        results = []
        for row in domains:
            mask = row > threshold
            indices = mask.nonzero(as_tuple=True)[0].tolist()
            if len(indices) > max_active:
                scores = row[indices]
                top_k = scores.topk(max_active).indices
                indices = [indices[i] for i in top_k.tolist()]
            results.append(indices)
        return results

    def get_active_capabilities(
        self,
        output: torch.Tensor,
        thresholds: dict[str, float],
    ) -> dict[str, bool]:
        caps = self.get_capabilities(output)[0]
        return {
            name: caps[i].item() > thresholds.get(name, 0.5)
            for i, name in enumerate(CAPABILITY_NAMES)
        }
