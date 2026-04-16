"""Catfish dissent injection (arxiv 2505.21503).

Injects devil's advocate when high agreement + weak arguments detected.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

CATFISH_SYSTEM_PROMPT = """You are a devil's advocate. Challenge the consensus position.
Find weaknesses, missing perspectives, and potential errors.
Provide a substantively different viewpoint backed by reasoning."""


@dataclass(frozen=True)
class CatfishResult:
    triggered: bool
    dissent_response: str | None
    reason: str


class CatfishModule:
    """Inject dissent when consensus is suspiciously strong but arguments weak."""

    def __init__(self, generate_fn=None, agreement_threshold: float = 0.95,
                 quality_threshold: float = 0.6) -> None:
        self._generate = generate_fn
        self._agreement_threshold = agreement_threshold
        self._quality_threshold = quality_threshold

    def should_trigger(self, agreement_score: float, avg_argument_quality: float) -> bool:
        return agreement_score > self._agreement_threshold and avg_argument_quality < self._quality_threshold

    async def generate_dissent(self, prompt: str, consensus_response: str) -> CatfishResult:
        if self._generate is None:
            return CatfishResult(triggered=True, dissent_response=None, reason="No generate_fn configured")

        dissent_prompt = f"{CATFISH_SYSTEM_PROMPT}\n\nOriginal prompt: {prompt}\nConsensus answer: {consensus_response}\n\nProvide your dissenting view:"
        dissent = await self._generate(dissent_prompt)
        logger.info("Catfish generated dissent for prompt: %.50s", prompt)
        return CatfishResult(triggered=True, dissent_response=dissent, reason="High agreement + weak arguments")

    async def maybe_dissent(self, prompt: str, consensus_response: str,
                            agreement_score: float, avg_quality: float) -> CatfishResult:
        if not self.should_trigger(agreement_score, avg_quality):
            return CatfishResult(triggered=False, dissent_response=None, reason="Thresholds not met")
        return await self.generate_dissent(prompt, consensus_response)
