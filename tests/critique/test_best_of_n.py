from __future__ import annotations

import pytest
from src.critique.best_of_n import BestOfN, ScoredCandidate


class TestBestOfN:
    def test_select_n_from_confidence(self):
        bon = BestOfN(high_threshold=0.8, mid_threshold=0.5, mid_n=3, low_n=5)
        assert bon.select_n(confidence=0.9) == 1
        assert bon.select_n(confidence=0.6) == 3
        assert bon.select_n(confidence=0.3) == 5

    @pytest.mark.asyncio
    async def test_generate_and_score(self):
        async def mock_generate(prompt: str) -> tuple[str, float]:
            return "response", -1.5

        bon = BestOfN()
        candidates = await bon.generate_candidates(prompt="test", generate_fn=mock_generate, n=3)
        assert len(candidates) == 3
        assert all(isinstance(c, ScoredCandidate) for c in candidates)

    @pytest.mark.asyncio
    async def test_best_candidate_is_highest_score(self):
        bon = BestOfN()
        candidates = [
            ScoredCandidate(text="bad", log_prob=-3.0),
            ScoredCandidate(text="best", log_prob=-0.5),
            ScoredCandidate(text="ok", log_prob=-1.5),
        ]
        best = bon.select_best(candidates)
        assert best.text == "best"

    @pytest.mark.asyncio
    async def test_single_candidate_when_confident(self):
        call_count = 0

        async def mock_generate(prompt: str) -> tuple[str, float]:
            nonlocal call_count
            call_count += 1
            return "response", -1.0

        bon = BestOfN(high_threshold=0.8)
        result = await bon.run(prompt="test", generate_fn=mock_generate, router_confidence=0.9)
        assert call_count == 1
        assert result.text == "response"
