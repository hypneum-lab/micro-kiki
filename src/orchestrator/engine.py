from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Awaitable

from src.critique.best_of_n import BestOfN
from src.critique.self_refine import SelfRefine
from src.critique.agentic_loop import AgenticLoop
from src.search.base import SearchResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchestrationResult:
    response: str
    search_results: list[SearchResult]
    critique_applied: bool
    iterations: int = 1


class OrchestrationEngine:
    """Main orchestration engine: routes queries through capabilities."""

    def __init__(
        self,
        capabilities_config: dict,
        best_of_n_config: dict,
        agentic_max_iterations: int = 5,
    ) -> None:
        self._caps_config = capabilities_config
        self._bon = BestOfN(**best_of_n_config)
        self._agentic_max_iterations = agentic_max_iterations

    async def _search(self, query: str) -> list[SearchResult]:
        return []

    def _format_search_context(self, results: list[SearchResult]) -> str:
        if not results:
            return ""
        lines = ["## Search Results"]
        for r in results:
            lines.append(f"- **{r.title}** ({r.url}): {r.snippet}")
        return "\n".join(lines)

    async def process(
        self,
        query: str,
        active_capabilities: dict[str, bool],
        generate_fn: Callable[[str], Awaitable[tuple[str, float]]],
        router_confidence: float,
    ) -> OrchestrationResult:
        search_results: list[SearchResult] = []
        critique_applied = False
        augmented_query = query

        if active_capabilities.get("web_search"):
            search_results = await self._search(query)
            context = self._format_search_context(search_results)
            if context:
                augmented_query = f"{context}\n\n## Query\n{query}"

        if active_capabilities.get("self_critique_task"):
            async def gen_text(prompt: str) -> str:
                text, _ = await generate_fn(prompt)
                return text

            loop = AgenticLoop(
                generate_fn=gen_text,
                max_iterations=self._agentic_max_iterations,
            )
            result = await loop.run(task=augmented_query, tools=["search_web", "search_papers"])
            return OrchestrationResult(
                response=result.final_output,
                search_results=search_results,
                critique_applied=True,
                iterations=result.iterations,
            )

        if active_capabilities.get("self_critique_token"):
            candidate = await self._bon.run(
                prompt=augmented_query,
                generate_fn=generate_fn,
                router_confidence=router_confidence,
            )
            response_text = candidate.text
        else:
            response_text, _ = await generate_fn(augmented_query)

        if active_capabilities.get("self_critique_response"):
            async def gen_text_refine(prompt: str) -> str:
                text, _ = await generate_fn(prompt)
                return text

            refine = SelfRefine(generate_fn=gen_text_refine)
            result = await refine.run(query=query, response=response_text)
            response_text = result.final_response
            critique_applied = result.corrected

        return OrchestrationResult(
            response=response_text,
            search_results=search_results,
            critique_applied=critique_applied,
        )
