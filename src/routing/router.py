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

# 10 niche domain names (pivot 2026-04-16: 32 → 10 niche + 1 base = 11 outputs)
NICHE_DOMAINS: frozenset[str] = frozenset({
    "kicad-dsl",
    "spice",
    "emc",
    "stm32",
    "embedded",
    "freecad",
    "platformio",
    "power",
    "dsp",
    "electronics",
})


class MetaRouter(nn.Module):
    """Sigmoid meta-router with domain + capability outputs."""

    def __init__(
        self,
        input_dim: int = 768,
        num_domains: int = 11,
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

    # Ordered list of niche domain names for 11-output mode (index 0-9).
    # Index 10 is the implicit "base" output (fallback to raw 35B, no adapter).
    _NICHE_DOMAIN_LIST: list[str] = sorted(NICHE_DOMAINS)

    def get_active_domains_named(
        self,
        output: torch.Tensor,
        threshold: float = 0.12,
        max_active: int = 4,
    ) -> list[str]:
        """Return active niche domain names, or ["base"] when none exceed threshold.

        Only valid when num_domains == 11 (10 niche + 1 base output).
        """
        domains = self.get_domains(output)[0]  # first batch item
        niche_scores = domains[: len(self._NICHE_DOMAIN_LIST)]
        active = [
            self._NICHE_DOMAIN_LIST[i]
            for i, score in enumerate(niche_scores)
            if score.item() > threshold
        ]
        if len(active) > max_active:
            scored = sorted(
                active,
                key=lambda name: niche_scores[self._NICHE_DOMAIN_LIST.index(name)].item(),
                reverse=True,
            )
            active = scored[:max_active]
        return active if active else ["base"]

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
