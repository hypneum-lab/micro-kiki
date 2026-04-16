"""Reasoning-based Bias Detector (arxiv 2505.17100).

Post-inference check: flags potential biases via reasoning pass.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

RBD_PROMPT = """Analyze this response for potential biases.

## Prompt
{prompt}

## Response
{response}

Check for: confirmation bias, stereotyping, framing effects, authority bias, anchoring.

Return JSON:
{{"biased": true/false, "bias_type": "type or null", "explanation": "brief", "confidence": 0-1}}"""


@dataclass(frozen=True)
class BiasDetection:
    biased: bool
    bias_type: str | None
    explanation: str
    confidence: float


class ReasoningBiasDetector:
    def __init__(self, generate_fn=None) -> None:
        self._generate = generate_fn

    async def detect(self, prompt: str, response: str) -> BiasDetection:
        if self._generate is None:
            return BiasDetection(biased=False, bias_type=None, explanation="No detector configured", confidence=0)

        raw = await self._generate(RBD_PROMPT.format(prompt=prompt, response=response))
        try:
            data = json.loads(raw)
            return BiasDetection(
                biased=data.get("biased", False),
                bias_type=data.get("bias_type"),
                explanation=data.get("explanation", ""),
                confidence=data.get("confidence", 0),
            )
        except (json.JSONDecodeError, KeyError):
            return BiasDetection(biased=False, bias_type=None, explanation="Parse error", confidence=0)
