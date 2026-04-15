"""Negotiator: K candidates -> argument extraction -> judge -> catfish -> response."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from src.cognitive.argument_extractor import ArgumentExtractor
from src.cognitive.judge import AdaptiveJudge, JudgeResult
from src.cognitive.catfish import CatfishModule, CatfishResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NegotiationResult:
    winner_response: str
    winner_idx: int
    judge_result: JudgeResult
    catfish_result: CatfishResult | None
    num_candidates: int


class Negotiator:
    def __init__(self, extractor: ArgumentExtractor, judge: AdaptiveJudge,
                 catfish: CatfishModule) -> None:
        self._extractor = extractor
        self._judge = judge
        self._catfish = catfish

    async def negotiate(self, prompt: str, candidates: list[str]) -> NegotiationResult:
        arguments = await self._extractor.extract_batch(candidates)

        # Compute agreement (simple: embedding similarity proxy via text overlap)
        if len(set(c[:50] for c in candidates)) == 1:
            agreement = 1.0
        else:
            agreement = 1.0 / len(set(c[:50] for c in candidates))

        avg_quality = sum(a.quality_score for a in arguments) / len(arguments) if arguments else 0

        judge_result = await self._judge.judge(candidates, arguments, agreement)

        catfish_result = await self._catfish.maybe_dissent(
            prompt, candidates[judge_result.winner_idx], agreement, avg_quality,
        )

        if catfish_result.triggered and catfish_result.dissent_response:
            candidates.append(catfish_result.dissent_response)
            new_args = await self._extractor.extract_batch(candidates)
            judge_result = await self._judge.judge(candidates, new_args, agreement * 0.5)

        return NegotiationResult(
            winner_response=candidates[judge_result.winner_idx],
            winner_idx=judge_result.winner_idx,
            judge_result=judge_result,
            catfish_result=catfish_result,
            num_candidates=len(candidates),
        )
