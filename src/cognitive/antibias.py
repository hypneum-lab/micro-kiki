"""Anti-bias orchestrator: RBD check + DeFrame re-generation."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from src.cognitive.rbd import ReasoningBiasDetector, BiasDetection

logger = logging.getLogger(__name__)

DEFRAME_PROMPT = """The following response was flagged for potential {bias_type} bias.
Rewrite it to be fair, balanced, and free of the identified bias while preserving accuracy.

Original prompt: {prompt}
Original response: {response}
Bias detected: {explanation}

Rewrite:"""


@dataclass(frozen=True)
class AntiBiasResult:
    original_response: str
    final_response: str
    bias_detected: bool
    detection: BiasDetection
    rewritten: bool


class AntiBiasOrchestrator:
    def __init__(self, detector: ReasoningBiasDetector, generate_fn=None) -> None:
        self._detector = detector
        self._generate = generate_fn

    async def check_and_fix(self, prompt: str, response: str) -> AntiBiasResult:
        detection = await self._detector.detect(prompt, response)

        if not detection.biased or detection.confidence < 0.5:
            return AntiBiasResult(
                original_response=response, final_response=response,
                bias_detected=False, detection=detection, rewritten=False,
            )

        if self._generate is None:
            return AntiBiasResult(
                original_response=response, final_response=response,
                bias_detected=True, detection=detection, rewritten=False,
            )

        rewrite = await self._generate(DEFRAME_PROMPT.format(
            bias_type=detection.bias_type or "unknown",
            prompt=prompt, response=response, explanation=detection.explanation,
        ))

        logger.info("DeFrame rewrite triggered for %s bias", detection.bias_type)
        return AntiBiasResult(
            original_response=response, final_response=rewrite,
            bias_detected=True, detection=detection, rewritten=True,
        )
