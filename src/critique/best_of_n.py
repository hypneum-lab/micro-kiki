from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Awaitable


@dataclass(frozen=True)
class ScoredCandidate:
    text: str
    log_prob: float


class BestOfN:
    """Adaptive best-of-N sampling driven by router confidence."""

    def __init__(
        self,
        high_threshold: float = 0.8,
        mid_threshold: float = 0.5,
        mid_n: int = 3,
        low_n: int = 5,
    ) -> None:
        self._high_threshold = high_threshold
        self._mid_threshold = mid_threshold
        self._mid_n = mid_n
        self._low_n = low_n

    def select_n(self, confidence: float) -> int:
        if confidence > self._high_threshold:
            return 1
        if confidence > self._mid_threshold:
            return self._mid_n
        return self._low_n

    async def generate_candidates(
        self,
        prompt: str,
        generate_fn: Callable[[str], Awaitable[tuple[str, float]]],
        n: int,
    ) -> list[ScoredCandidate]:
        tasks = [generate_fn(prompt) for _ in range(n)]
        results = await asyncio.gather(*tasks)
        return [ScoredCandidate(text=text, log_prob=lp) for text, lp in results]

    @staticmethod
    def select_best(candidates: list[ScoredCandidate]) -> ScoredCandidate:
        return max(candidates, key=lambda c: c.log_prob)

    async def run(
        self,
        prompt: str,
        generate_fn: Callable[[str], Awaitable[tuple[str, float]]],
        router_confidence: float,
    ) -> ScoredCandidate:
        n = self.select_n(router_confidence)
        candidates = await self.generate_candidates(prompt, generate_fn, n)
        return self.select_best(candidates)
