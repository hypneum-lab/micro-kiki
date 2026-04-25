from __future__ import annotations

"""Tests for the RLM recursive domain router.

All HTTP calls are mocked — no running server required.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.routing.rlm_router import (
    RLMRouter,
    _extract_content,
    _parse_decomposition,
    _subs_to_route,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _openai_response(content: str) -> dict:
    """Build a minimal OpenAI-style response envelope."""
    return {
        "choices": [
            {"message": {"content": content}, "finish_reason": "stop"}
        ],
    }


def _mock_post(content: str, *, status_code: int = 200):
    """Return a mock ``httpx.Client.post`` that returns *content*."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = _openai_response(content)
    resp.raise_for_status.return_value = None
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


# ---------------------------------------------------------------------------
# Unit: _extract_content
# ---------------------------------------------------------------------------


class TestExtractContent:
    def test_normal_response(self):
        body = _openai_response("hello")
        assert _extract_content(body) == "hello"

    def test_missing_choices_raises(self):
        with pytest.raises(ValueError):
            _extract_content({})


# ---------------------------------------------------------------------------
# Unit: _parse_decomposition
# ---------------------------------------------------------------------------


class TestParseDecomposition:
    def test_valid_json(self):
        text = json.dumps({"sub_queries": [{"domain": "python", "query": "sort a list"}]})
        result = _parse_decomposition(text)
        assert result == [{"domain": "python", "query": "sort a list"}]

    def test_markdown_fenced_json(self):
        text = '```json\n{"sub_queries": [{"domain": "rust", "query": "ownership"}]}\n```'
        result = _parse_decomposition(text)
        assert result == [{"domain": "rust", "query": "ownership"}]

    def test_unknown_domain_skipped(self):
        text = json.dumps(
            {"sub_queries": [
                {"domain": "python", "query": "valid"},
                {"domain": "fortran", "query": "invalid"},
            ]}
        )
        result = _parse_decomposition(text)
        assert len(result) == 1
        assert result[0]["domain"] == "python"

    def test_garbage_raises(self):
        with pytest.raises(ValueError, match="Could not parse"):
            _parse_decomposition("this is not json at all")

    def test_empty_sub_queries_raises(self):
        text = json.dumps({"sub_queries": []})
        with pytest.raises(ValueError, match="Empty"):
            _parse_decomposition(text)

    def test_bare_list_accepted(self):
        text = json.dumps([{"domain": "stm32", "query": "CAN bus init"}])
        result = _parse_decomposition(text)
        assert result == [{"domain": "stm32", "query": "CAN bus init"}]


# ---------------------------------------------------------------------------
# Unit: _subs_to_route
# ---------------------------------------------------------------------------


class TestSubsToRoute:
    def test_single_uses_original_query(self):
        subs = [{"domain": "python", "query": "sub"}]
        result = _subs_to_route(subs, "original query")
        assert result == [("python", 1.0, "original query")]

    def test_multi_uses_sub_queries(self):
        subs = [
            {"domain": "stm32", "query": "CAN bus"},
            {"domain": "power", "query": "battery"},
        ]
        result = _subs_to_route(subs, "original")
        assert len(result) == 2
        assert result[0] == ("stm32", 1.0, "CAN bus")
        assert result[1] == ("power", 1.0, "battery")


# ---------------------------------------------------------------------------
# Integration: RLMRouter.decompose (mocked HTTP)
# ---------------------------------------------------------------------------


class TestDecomposeSingleDomain:
    """Simple query returns a single domain."""

    def test_single_domain(self):
        content = json.dumps(
            {"sub_queries": [{"domain": "python", "query": "How to sort a list in Python"}]}
        )
        router = RLMRouter(server_url="http://test:9200")

        with patch("src.routing.rlm_router.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = _mock_post(content)
            mock_cls.return_value = mock_client

            result = router.decompose("How to sort a list in Python")

        assert len(result) == 1
        assert result[0]["domain"] == "python"


class TestDecomposeMultiDomain:
    """Complex query returns 2-4 domains."""

    def test_multi_domain(self):
        content = json.dumps(
            {"sub_queries": [
                {"domain": "stm32", "query": "Design an STM32 system with CAN bus"},
                {"domain": "power", "query": "Battery power supply for embedded"},
                {"domain": "emc", "query": "EMC compliance strategy"},
                {"domain": "kicad-pcb", "query": "Custom KiCad PCB layout"},
            ]}
        )
        router = RLMRouter(server_url="http://test:9200")

        with patch("src.routing.rlm_router.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = _mock_post(content)
            mock_cls.return_value = mock_client

            result = router.decompose(
                "Design a battery-powered STM32 system with CAN bus, "
                "EMC compliance, and a custom KiCad PCB layout"
            )

        assert len(result) == 4
        domains = {r["domain"] for r in result}
        assert domains == {"stm32", "power", "emc", "kicad-pcb"}


# ---------------------------------------------------------------------------
# Integration: RLMRouter.route (mocked HTTP)
# ---------------------------------------------------------------------------


class TestRouteFallbackOnJsonFailure:
    """Malformed model output triggers the fallback router."""

    def test_fallback_on_parse_failure(self):
        fallback = MagicMock()
        fallback.route.return_value = [("chat-fr", 0.5, "the query")]

        router = RLMRouter(
            server_url="http://test:9200",
            fallback_router=fallback,
        )

        with patch("src.routing.rlm_router.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = _mock_post("this is garbage, not JSON")
            mock_cls.return_value = mock_client

            result = router.route("the query")

        fallback.route.assert_called_once_with("the query")
        assert result == [("chat-fr", 0.5, "the query")]


class TestRouteFallbackOnTimeout:
    """Server timeout triggers the fallback router."""

    def test_fallback_on_timeout(self):
        fallback = MagicMock()
        fallback.route.return_value = [("reasoning", 0.8, "think about X")]

        router = RLMRouter(
            server_url="http://test:9200",
            fallback_router=fallback,
            timeout_s=0.1,
        )

        with patch("src.routing.rlm_router.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = TimeoutError("timed out")
            mock_cls.return_value = mock_client

            result = router.route("think about X")

        fallback.route.assert_called_once_with("think about X")
        assert result == [("reasoning", 0.8, "think about X")]


class TestRouteFallbackOnHttpError:
    """HTTP 500 triggers the fallback router."""

    def test_fallback_on_http_error(self):
        fallback = MagicMock()
        fallback.route.return_value = [("shell", 0.9, "ls -la")]

        router = RLMRouter(
            server_url="http://test:9200",
            fallback_router=fallback,
        )

        with patch("src.routing.rlm_router.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = _mock_post("error", status_code=500)
            mock_cls.return_value = mock_client

            result = router.route("ls -la")

        fallback.route.assert_called_once()


class TestRouteFallbackNone:
    """When no fallback is set, returns default chat-fr entry."""

    def test_default_fallback(self):
        router = RLMRouter(server_url="http://test:9200")

        with patch("src.routing.rlm_router.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = ConnectionError("refused")
            mock_cls.return_value = mock_client

            result = router.route("bonjour")

        assert result == [("chat-fr", 0.0, "bonjour")]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class TestCacheHit:
    """Same query returns cached result without a second HTTP call."""

    def test_cache_hit(self):
        content = json.dumps(
            {"sub_queries": [{"domain": "docker", "query": "Dockerfile multi-stage"}]}
        )
        router = RLMRouter(server_url="http://test:9200")

        with patch("src.routing.rlm_router.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = _mock_post(content)
            mock_cls.return_value = mock_client

            result1 = router.route("Dockerfile multi-stage")
            result2 = router.route("Dockerfile multi-stage")

        # Only one HTTP call — second hit served from cache
        assert mock_client.post.call_count == 1
        assert result1 == result2
        assert result1 == [("docker", 1.0, "Dockerfile multi-stage")]


class TestCacheEviction:
    """Cache evicts oldest entry when full."""

    def test_eviction_at_capacity(self):
        router = RLMRouter(server_url="http://test:9200")

        # Fill cache manually
        from src.routing.rlm_router import _CacheEntry
        import time

        for i in range(256):
            router._cache[f"q{i}"] = _CacheEntry(
                result=(("python", 1.0, f"q{i}"),),
                timestamp=float(i),
            )

        assert len(router._cache) == 256

        # Insert one more via the public path
        content = json.dumps(
            {"sub_queries": [{"domain": "rust", "query": "new query"}]}
        )

        with patch("src.routing.rlm_router.httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = _mock_post(content)
            mock_cls.return_value = mock_client

            router.route("new query")

        # Oldest (q0, timestamp=0.0) should be evicted
        assert "q0" not in router._cache
        assert "new query" in router._cache
        assert len(router._cache) == 256
