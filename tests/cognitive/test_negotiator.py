"""Tests for Phase IX Negotiator + Phase X Anti-bias."""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock
from src.cognitive.argument_extractor import ArgumentExtractor, Argument
from src.cognitive.judge import AdaptiveJudge, JudgeResult
from src.cognitive.catfish import CatfishModule
from src.cognitive.rbd import ReasoningBiasDetector, BiasDetection
from src.cognitive.antibias import AntiBiasOrchestrator


class TestArgumentExtractor:
    @pytest.mark.asyncio
    async def test_extract_without_generate(self):
        ext = ArgumentExtractor()
        arg = await ext.extract("Test response about circuits")
        assert isinstance(arg, Argument)
        assert arg.claim != ""

    @pytest.mark.asyncio
    async def test_extract_with_generate(self):
        async def mock_gen(prompt):
            return json.dumps({"claim": "Main point", "evidence": "Data", "reasoning": "Because"})
        ext = ArgumentExtractor(generate_fn=mock_gen)
        arg = await ext.extract("Response")
        assert arg.claim == "Main point"
        assert arg.quality_score > 0.5


class TestAdaptiveJudge:
    @pytest.mark.asyncio
    async def test_skip_on_high_agreement(self):
        judge = AdaptiveJudge()
        result = await judge.judge(["a", "a"], [Argument("c", "e", "r", 0.8)] * 2, 0.95)
        assert result.backend_used == "skip"

    @pytest.mark.asyncio
    async def test_fast_on_mid_agreement(self):
        fast = AsyncMock()
        fast.generate.return_value = json.dumps({"winner_idx": 0, "confidence": 0.7, "rationale": "Better"})
        judge = AdaptiveJudge(fast_client=fast)
        result = await judge.judge(["a", "b"], [Argument("c", "", "", 0.3)] * 2, 0.7)
        assert result.backend_used == "fast"


class TestCatfish:
    def test_trigger_conditions(self):
        cat = CatfishModule()
        assert cat.should_trigger(0.98, 0.4) is True
        assert cat.should_trigger(0.80, 0.4) is False
        assert cat.should_trigger(0.98, 0.8) is False

    @pytest.mark.asyncio
    async def test_no_trigger_returns_false(self):
        cat = CatfishModule()
        result = await cat.maybe_dissent("prompt", "response", 0.5, 0.8)
        assert result.triggered is False


class TestRBD:
    @pytest.mark.asyncio
    async def test_detect_no_bias(self):
        async def mock_gen(prompt):
            return json.dumps({"biased": False, "bias_type": None, "explanation": "Clean", "confidence": 0.1})
        rbd = ReasoningBiasDetector(generate_fn=mock_gen)
        result = await rbd.detect("prompt", "response")
        assert result.biased is False

    @pytest.mark.asyncio
    async def test_detect_bias(self):
        async def mock_gen(prompt):
            return json.dumps({"biased": True, "bias_type": "stereotyping", "explanation": "Gender", "confidence": 0.9})
        rbd = ReasoningBiasDetector(generate_fn=mock_gen)
        result = await rbd.detect("prompt", "biased response")
        assert result.biased is True
        assert result.bias_type == "stereotyping"


class TestAntiBias:
    @pytest.mark.asyncio
    async def test_no_rewrite_when_clean(self):
        async def mock_gen(prompt):
            return json.dumps({"biased": False, "bias_type": None, "explanation": "OK", "confidence": 0.1})
        rbd = ReasoningBiasDetector(generate_fn=mock_gen)
        ab = AntiBiasOrchestrator(detector=rbd)
        result = await ab.check_and_fix("prompt", "clean response")
        assert result.rewritten is False
        assert result.final_response == "clean response"

    @pytest.mark.asyncio
    async def test_rewrite_when_biased(self):
        call_count = 0
        async def mock_gen(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({"biased": True, "bias_type": "framing", "explanation": "Biased framing", "confidence": 0.9})
            return "Rewritten fair response"
        rbd = ReasoningBiasDetector(generate_fn=mock_gen)
        ab = AntiBiasOrchestrator(detector=rbd, generate_fn=mock_gen)
        result = await ab.check_and_fix("prompt", "biased response")
        assert result.rewritten is True
        assert "Rewritten" in result.final_response
