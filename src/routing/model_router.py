from __future__ import annotations

import logging
from dataclasses import dataclass

from src.routing.router import NICHE_DOMAINS

logger = logging.getLogger(__name__)

_CODE_HINTS = {"code", "coding", "debug", "firmware", "c++", "python", "rust"}

# Sub-category routing: source domain -> parent domain.
# When a sub-category domain is detected, the router first tries the parent
# domain's stack, then falls back to the sub-category-specific adapter if available.
SUBCATEGORY_MAP: dict[str, str] = {
    "stm32": "embedded",
}


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
            # Sub-category routing: try parent domain stack first if configured
            parent = SUBCATEGORY_MAP.get(domain_hint)
            if parent and parent in NICHE_DOMAINS:
                adapter = f"stack-{parent}"
                logger.debug(
                    "sub-category %s → parent %s stack (stack-%s), "
                    "fallback to stack-%s if available",
                    domain_hint, parent, parent, domain_hint,
                )
                return RouteDecision(
                    model_id="qwen35b",
                    adapter=adapter,
                    reason=f"sub-category: {domain_hint} -> {parent} stack "
                           f"(fallback: stack-{domain_hint})",
                )
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
