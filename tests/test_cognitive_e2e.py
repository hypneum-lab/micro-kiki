"""Story-23: Full cognitive layer E2E test.

Proves the chain Router -> Aeon Memory recall -> Inference (mocked) ->
Negotiator -> Anti-bias -> Aeon Memory write works end-to-end with no
real LLM and no GPU.
"""
from __future__ import annotations

import asyncio
import hashlib
from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.cognitive.antibias import AntiBiasPipeline
from src.cognitive.argument_extractor import ArgumentExtractor
from src.cognitive.catfish import CatfishModule
from src.cognitive.judge import AdaptiveJudge
from src.cognitive.negotiator import Negotiator
from src.cognitive.rbd import ReasoningBiasDetector
from src.memory.aeon import AeonPalace
from src.routing.model_router import ModelRouter, RouteDecision
from src.serving.aeon_hook import AeonServingHook


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_embed(dim: int = 64):
    def fn(text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec = rng.randn(dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)
    return fn


DOMAINS = ["spice", "stm32", "power", "embedded", "kicad-dsl"]

TURNS = [
    ("Simulate an RC filter in SPICE with 10kHz cutoff", "spice"),
    ("Configure STM32 ADC for 12-bit continuous mode", "stm32"),
    ("Design a 5V buck converter with LM2596", "power"),
    ("How to debounce a GPIO on STM32?", "stm32"),
    ("Run transient analysis on MOSFET switching circuit", "spice"),
    ("PCB layout for a 4-layer power supply", "power"),
    ("Write SPI driver for STM32F4 with DMA", "stm32"),
    ("SPICE model for a Zener diode clamp", "spice"),
    ("Calculate inductor ripple current for buck converter", "power"),
    ("Optimize embedded interrupt latency on Cortex-M4", "embedded"),
]


def _build_negotiator() -> Negotiator:
    """Build a Negotiator with no LLM backends (graceful fallback)."""
    extractor = ArgumentExtractor(generate_fn=None)
    judge = AdaptiveJudge(fast_client=None, deep_client=None)
    catfish = CatfishModule(generate_fn=None)
    return Negotiator(extractor=extractor, judge=judge, catfish=catfish)


def _build_antibias_pipeline() -> AntiBiasPipeline:
    """Build an AntiBias pipeline with no LLM (detector returns not-biased)."""
    detector = ReasoningBiasDetector(generate_fn=None)
    return AntiBiasPipeline(detector=detector, generate_fn=None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCognitiveE2E:
    """Full cognitive pipeline integration tests (mock-based, no GPU)."""

    @pytest.mark.asyncio
    async def test_full_pipeline_10_turn_session(self) -> None:
        """Simulate a 10-turn hardware design session touching 3+ domains.

        For each turn: route -> recall -> infer (mock) -> write -> verify.
        """
        router = ModelRouter()
        palace = AeonPalace(dim=64, embed_fn=_mock_embed(64))
        hook = AeonServingHook(palace)
        negotiator = _build_negotiator()
        antibias = _build_antibias_pipeline()

        route_decisions: list[RouteDecision] = []
        memory_writes: list[str] = []

        for i, (query, domain) in enumerate(TURNS):
            # 1. Route
            decision = router.select(query, domain_hint=domain)
            assert decision is not None
            assert decision.model_id is not None
            route_decisions.append(decision)

            # 2. Recall via AeonServingHook (prepends memories)
            augmented_prompt = hook.pre_inference(query)
            assert augmented_prompt is not None

            # 3. Mock inference
            mock_response = f"[Turn {i}] Answer for: {query[:40]}..."

            # 4. Negotiate between 2 mock candidates
            candidates = [mock_response, f"Alt: {mock_response}"]
            neg_result = await negotiator.negotiate(augmented_prompt, candidates)
            assert neg_result is not None
            assert neg_result.winner_response is not None

            # 5. Anti-bias check
            ab_result = await antibias.process(query, neg_result.winner_response)
            assert ab_result is not None
            final_response = ab_result.final_response

            # 6. Write back to memory
            hook.post_inference(query, final_response, domain, f"turn-{i}")
            memory_writes.append(f"turn-{i}")

        # Verify all layers activated
        assert len(route_decisions) == 10
        assert len(memory_writes) == 10
        assert palace.stats["episodes"] == 10
        assert palace.stats["vectors"] == 10

        # Verify 3+ domains touched
        domains_seen = {TURNS[i][1] for i in range(10)}
        assert len(domains_seen) >= 3

    @pytest.mark.asyncio
    async def test_memory_accumulates_across_turns(self) -> None:
        """Write 5 turns, verify recall on turn 6 returns previous episodes."""
        palace = AeonPalace(dim=64, embed_fn=_mock_embed(64))

        # Write 5 episodes about SPICE simulation
        for i in range(5):
            palace.write(
                content=f"Turn {i}: SPICE simulation of RC filter variant {i}",
                domain="spice",
                source=f"turn-{i}",
            )

        assert palace.stats["episodes"] == 5

        # Turn 6: recall should find previous episodes
        recalled = palace.recall("SPICE RC filter simulation", top_k=10)
        assert len(recalled) > 0
        assert len(recalled) <= 5

        # Verify recalled episodes come from our writes
        for ep in recalled:
            assert ep.domain == "spice"
            assert "SPICE simulation" in ep.content

    @pytest.mark.asyncio
    async def test_negotiator_in_pipeline(self) -> None:
        """Mock 2 candidate responses, verify Negotiator picks one."""
        negotiator = _build_negotiator()

        candidates = [
            "Use a 100nF ceramic capacitor for decoupling.",
            "Use a 10uF tantalum capacitor for bulk decoupling.",
        ]
        result = await negotiator.negotiate(
            "What capacitor for STM32 decoupling?", candidates,
        )

        assert result is not None
        assert result.winner_response in candidates
        assert result.winner_idx in (0, 1)
        assert result.num_candidates >= 2
        assert result.judge_result is not None

    @pytest.mark.asyncio
    async def test_antibias_in_pipeline(self) -> None:
        """Pass a response through AntiBias, verify it runs without error."""
        pipeline = _build_antibias_pipeline()

        prompt = "Compare STM32 vs ESP32 for battery-powered IoT"
        response = "STM32 is always better because it has lower power consumption."

        result = await pipeline.process(prompt, response)

        assert result is not None
        assert isinstance(result.final_response, str)
        assert len(result.final_response) > 0
        # With no generate_fn, detector returns not-biased
        assert result.detection is not None

    @pytest.mark.asyncio
    async def test_no_component_silently_skipped(self) -> None:
        """Run full pipeline, check routing, memory, and negotiator all
        produce non-None results."""
        router = ModelRouter()
        palace = AeonPalace(dim=64, embed_fn=_mock_embed(64))
        hook = AeonServingHook(palace)
        negotiator = _build_negotiator()

        query = "Design a SPICE testbench for LDO voltage regulator"
        domain = "spice"

        # Routing
        decision = router.select(query, domain_hint=domain)
        assert decision is not None
        assert decision.model_id is not None
        assert decision.adapter is not None  # spice is a niche domain
        assert decision.reason is not None

        # Memory recall (empty palace, but must not error)
        augmented = hook.pre_inference(query)
        assert augmented is not None
        assert isinstance(augmented, str)

        # Mock inference
        mock_response = "LDO testbench: VIN sweep 3.0-5.5V, load step 0-500mA"

        # Negotiator
        candidates = [mock_response, "Alternative: use AC analysis for stability"]
        neg_result = await negotiator.negotiate(query, candidates)
        assert neg_result is not None
        assert neg_result.winner_response is not None
        assert neg_result.judge_result is not None
        assert neg_result.catfish_result is not None

        # Memory write
        hook.post_inference(query, neg_result.winner_response, domain, "turn-0")
        assert palace.stats["episodes"] == 1
        assert palace.stats["vectors"] == 1

        # Verify recall now returns the written episode
        recalled = palace.recall(query, top_k=5)
        assert len(recalled) == 1
        assert recalled[0].domain == domain
