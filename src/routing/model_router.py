from __future__ import annotations

import logging
from dataclasses import dataclass

from src.routing.router import NICHE_DOMAINS

logger = logging.getLogger(__name__)

_CODE_HINTS = {"code", "coding", "debug", "firmware", "c++", "python", "rust"}


@dataclass(frozen=True)
class RouteDecision:
    model_id: str        # "qwen35b" | "qwen480b" | "devstral"
    adapter: str | None  # "stack-<domain>" or None
    reason: str


class ModelRouter:
    """Stateless router: maps a query + hints to a (model, adapter) pair."""

    def select(
        self,
        query: str,
        domain_hint: str | None = None,
        require_deep: bool = False,
    ) -> RouteDecision:
        """Select model and optional LoRA adapter for the query.

        Priority order:
        1. require_deep=True → qwen480b (no adapter — too large for LoRA inference)
        2. domain_hint in NICHE_DOMAINS → qwen35b + stack adapter
        3. query tokens match code hints → devstral (no adapter)
        4. Default → qwen35b base (no adapter)
        """
        if require_deep:
            logger.debug("deep reasoning requested → qwen480b")
            return RouteDecision(
                model_id="qwen480b",
                adapter=None,
                reason="deep reasoning requested",
            )

        if domain_hint is not None and domain_hint in NICHE_DOMAINS:
            adapter = f"stack-{domain_hint}"
            logger.debug("niche domain %s → qwen35b + %s", domain_hint, adapter)
            return RouteDecision(
                model_id="qwen35b",
                adapter=adapter,
                reason=f"niche domain: {domain_hint}",
            )

        query_lower = query.lower()
        if any(hint in query_lower for hint in _CODE_HINTS):
            logger.debug("code hint detected → devstral")
            return RouteDecision(
                model_id="devstral",
                adapter=None,
                reason="code domain detected",
            )

        logger.debug("no niche signal → qwen35b base")
        return RouteDecision(
            model_id="qwen35b",
            adapter=None,
            reason="default: qwen35b base",
        )
