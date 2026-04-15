"""Extract structured arguments from candidate responses for Negotiator."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extract the key argument from this response.

## Response
{response}

## Return JSON:
{{"claim": "main claim", "evidence": "supporting evidence", "reasoning": "chain of reasoning"}}"""


@dataclass(frozen=True)
class Argument:
    claim: str
    evidence: str
    reasoning: str
    quality_score: float = 0.0


class ArgumentExtractor:
    def __init__(self, generate_fn=None) -> None:
        self._generate = generate_fn

    async def extract(self, response: str) -> Argument:
        if self._generate is None:
            return Argument(claim=response[:100], evidence="", reasoning="", quality_score=0.5)

        raw = await self._generate(EXTRACTION_PROMPT.format(response=response))
        try:
            data = json.loads(raw)
            quality = min(1.0, (
                (0.4 if data.get("evidence") else 0) +
                (0.4 if data.get("reasoning") else 0) +
                (0.2 if data.get("claim") else 0)
            ))
            return Argument(
                claim=data.get("claim", ""),
                evidence=data.get("evidence", ""),
                reasoning=data.get("reasoning", ""),
                quality_score=quality,
            )
        except (json.JSONDecodeError, KeyError):
            return Argument(claim=response[:100], evidence="", reasoning="", quality_score=0.1)

    async def extract_batch(self, responses: list[str]) -> list[Argument]:
        return [await self.extract(r) for r in responses]
