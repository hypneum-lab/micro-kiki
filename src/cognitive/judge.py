"""Adaptive judge: fast (Qwen35B) or deep (Mistral-Large) based on agreement.

Reference: CAMP arbitration (arxiv 2604.00085).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgeResult:
    winner_idx: int
    confidence: float
    rationale: str
    backend_used: str


class AdaptiveJudge:
    """Routes to fast or deep judge based on agreement score."""

    def __init__(self, fast_client=None, deep_client=None,
                 fast_model: str = "qwen35", deep_model: str = "mistral-large") -> None:
        self._fast = fast_client
        self._deep = deep_client
        self._fast_model = fast_model
        self._deep_model = deep_model

    def _select_backend(self, agreement_score: float) -> str:
        if agreement_score > 0.9:
            return "skip"
        if agreement_score >= 0.5:
            return "fast"
        return "deep"

    async def judge(self, candidates: list[str], arguments: list, agreement_score: float) -> JudgeResult:
        backend = self._select_backend(agreement_score)

        if backend == "skip":
            return JudgeResult(winner_idx=0, confidence=agreement_score, rationale="High agreement — trust consensus", backend_used="skip")

        prompt = "Compare these candidates and pick the best:\n"
        for i, (c, a) in enumerate(zip(candidates, arguments)):
            prompt += f"\n## Candidate {i}\n{c[:300]}\nClaim: {a.claim}\nEvidence: {a.evidence}\n"
        prompt += "\nReturn JSON: {\"winner_idx\": N, \"confidence\": 0-1, \"rationale\": \"reason\"}"

        client = self._fast if backend == "fast" else self._deep
        model = self._fast_model if backend == "fast" else self._deep_model

        if client is None:
            return JudgeResult(winner_idx=0, confidence=0.5, rationale=f"No {backend} client configured", backend_used=backend)

        raw = await client.generate(prompt=prompt, model=model)
        try:
            data = json.loads(raw)
            return JudgeResult(
                winner_idx=data.get("winner_idx", 0),
                confidence=data.get("confidence", 0.5),
                rationale=data.get("rationale", ""),
                backend_used=backend,
            )
        except (json.JSONDecodeError, KeyError):
            return JudgeResult(winner_idx=0, confidence=0.5, rationale="Parse error", backend_used=backend)
