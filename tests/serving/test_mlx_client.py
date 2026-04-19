"""Unit tests for :mod:`src.serving.mlx_client`.

All tests stub the network with ``httpx.MockTransport`` — no real
sockets are opened. Runs in <100 ms with ``asyncio_mode = auto``.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from src.serving import mlx_client
from src.serving.mlx_client import MAX_RETRIES, MLXClient


def _chat_completion(text: str) -> dict[str, Any]:
    """Build an OpenAI-shaped response body containing ``text``."""
    return {
        "id": "cmpl-test",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}}
        ],
    }


def _make_client(handler) -> MLXClient:
    transport = httpx.MockTransport(handler)
    return MLXClient(
        host="http://studio-test:8000",
        model="qwen3.6-35b",
        timeout=1.0,
        transport=transport,
    )


# ---------------------------------------------------------------------------
# 1. Base-model path (no adapter)
# ---------------------------------------------------------------------------


async def test_generate_base_model_no_adapter() -> None:
    """When ``adapter`` is None the payload must NOT carry an adapter key."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_chat_completion("hello from base"))

    client = _make_client(handler)
    out = await client.generate("say hi", adapter=None)

    assert out == "hello from base"
    assert captured["url"].endswith("/v1/chat/completions")
    body = captured["body"]
    assert "adapter" not in body, body
    assert body["model"] == "qwen3.6-35b"
    assert body["messages"] == [{"role": "user", "content": "say hi"}]
    # Deterministic by default so win-rate eval is reproducible.
    assert body["temperature"] == 0.0


# ---------------------------------------------------------------------------
# 2. Adapter path (explicit LoRA)
# ---------------------------------------------------------------------------


async def test_generate_with_adapter() -> None:
    """``adapter`` is propagated verbatim as a top-level body field."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_chat_completion("adapted reply"))

    client = _make_client(handler)
    out = await client.generate(
        "explain entropy",
        adapter="/path/to/stack-05/adapters.safetensors",
    )

    assert out == "adapted reply"
    assert captured["body"]["adapter"] == (
        "/path/to/stack-05/adapters.safetensors"
    )


# ---------------------------------------------------------------------------
# 3. Transient 5xx → retry succeeds
# ---------------------------------------------------------------------------


async def test_retry_on_503(monkeypatch: pytest.MonkeyPatch) -> None:
    """MockTransport returns 503 twice then 200; client must retry."""
    # Make backoff a no-op so the test stays fast.
    async def _fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mlx_client.asyncio, "sleep", _fast_sleep)

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="temporary")
        return httpx.Response(200, json=_chat_completion("finally ok"))

    client = _make_client(handler)
    out = await client.generate("ping")

    assert out == "finally ok"
    assert calls["n"] == 3, "client must retry twice then succeed"


# ---------------------------------------------------------------------------
# 4. Permanent 5xx → raises after MAX_RETRIES
# ---------------------------------------------------------------------------


async def test_timeout_raises_after_max_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Permanent 503: client exhausts retries and propagates the error."""
    async def _fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mlx_client.asyncio, "sleep", _fast_sleep)

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="still down")

    client = _make_client(handler)

    with pytest.raises(httpx.HTTPStatusError):
        await client.generate("ping")

    assert calls["n"] == MAX_RETRIES, (
        f"expected {MAX_RETRIES} attempts, got {calls['n']}"
    )


# ---------------------------------------------------------------------------
# 5. Non-5xx HTTP errors are not retried (bonus — cheap sanity)
# ---------------------------------------------------------------------------


async def test_client_error_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 400 Bad Request raises immediately without burning retry budget."""
    async def _fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(mlx_client.asyncio, "sleep", _fast_sleep)

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad")

    client = _make_client(handler)

    with pytest.raises(httpx.HTTPStatusError):
        await client.generate("ping")

    assert calls["n"] == 1
