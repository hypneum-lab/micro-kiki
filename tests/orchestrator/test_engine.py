from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.orchestrator.engine import OrchestrationEngine


@pytest.fixture
def engine():
    return OrchestrationEngine(
        capabilities_config={
            "web_search": {"threshold": 0.15},
            "self_critique_token": {"threshold": 0.10},
            "self_critique_response": {"threshold": 0.20},
            "self_critique_task": {"threshold": 0.35},
            "deep_eval": {"threshold": 0.25},
        },
        best_of_n_config={
            "high_threshold": 0.8, "mid_threshold": 0.5,
            "mid_n": 3, "low_n": 5,
        },
    )


class TestOrchestrationEngine:
    @pytest.mark.asyncio
    async def test_simple_query_no_capabilities(self, engine):
        active_caps = {k: False for k in [
            "web_search", "self_critique_token", "self_critique_response",
            "self_critique_task", "deep_eval",
        ]}
        result = await engine.process(
            query="Hello",
            active_capabilities=active_caps,
            generate_fn=AsyncMock(return_value=("Hello back", -1.0)),
            router_confidence=0.95,
        )
        assert result.response == "Hello back"
        assert result.search_results == []
        assert result.critique_applied is False

    @pytest.mark.asyncio
    async def test_web_search_injects_context(self, engine):
        active_caps = {k: False for k in [
            "web_search", "self_critique_token", "self_critique_response",
            "self_critique_task", "deep_eval",
        ]}
        active_caps["web_search"] = True

        mock_search = AsyncMock(return_value=[
            MagicMock(title="Result 1", url="https://a.com", snippet="snippet 1"),
        ])

        with patch.object(engine, "_search", mock_search):
            result = await engine.process(
                query="What is MoE?",
                active_capabilities=active_caps,
                generate_fn=AsyncMock(return_value=("Response with sources", -1.0)),
                router_confidence=0.9,
            )

        assert len(result.search_results) == 1
        assert result.response == "Response with sources"
