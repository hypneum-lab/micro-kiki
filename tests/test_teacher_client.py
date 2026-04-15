"""Tests for :mod:`src.distill.teacher_client`.

All network I/O is mocked via an in-process ``httpx.MockTransport`` so
these tests stay deterministic and CPU-only. We cover:

* cache miss then hit (no second HTTP call)
* transient 500 → retry → success
* permanent 400 → no retry, raises
* Qwen3 thinking-mode payload shape
* endpoint resolution error
* cache key determinism (param order irrelevant)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.distill.teacher_client import (
    QWEN3_THINKING_MODELS,
    GenerateParams,
    RetryPolicy,
    TeacherCache,
    TeacherClient,
    TeacherError,
    cache_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_response(content: str, model: str = "mistral-large-opus") -> dict[str, Any]:
    return {
        "id": "resp-1",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }


def _make_client(
    handler: Any,
    *,
    tmp_path: Path,
    retry: RetryPolicy | None = None,
    endpoints: dict[str, str] | None = None,
) -> TeacherClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, timeout=5.0)
    cache = TeacherCache(path=tmp_path / "cache.sqlite")
    retry = retry or RetryPolicy(max_attempts=3, base_delay_s=0.0, jitter=0.0)
    return TeacherClient(
        endpoints=endpoints or {"mistral-large-opus": "http://teacher.local"},
        cache=cache,
        retry=retry,
        http_client=http,
    )


# ---------------------------------------------------------------------------
# cache_key
# ---------------------------------------------------------------------------


def test_cache_key_is_param_order_independent() -> None:
    a = cache_key("hi", "m", {"temperature": 0.7, "top_p": 0.9})
    b = cache_key("hi", "m", {"top_p": 0.9, "temperature": 0.7})
    assert a == b


def test_cache_key_changes_with_model() -> None:
    a = cache_key("hi", "m1", {"temperature": 0.7})
    b = cache_key("hi", "m2", {"temperature": 0.7})
    assert a != b


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------


def test_retry_policy_zero_delay_on_first_attempt() -> None:
    policy = RetryPolicy(max_attempts=3, base_delay_s=1.0, jitter=0.0)
    assert policy.sleep_for(1) == 0.0


def test_retry_policy_exponential_growth() -> None:
    policy = RetryPolicy(max_attempts=4, base_delay_s=1.0, factor=2.0, jitter=0.0)
    assert policy.sleep_for(2) == pytest.approx(1.0)
    assert policy.sleep_for(3) == pytest.approx(2.0)
    assert policy.sleep_for(4) == pytest.approx(4.0)


def test_retry_policy_caps_at_max_delay() -> None:
    policy = RetryPolicy(
        max_attempts=10, base_delay_s=1.0, factor=10.0, max_delay_s=5.0, jitter=0.0
    )
    assert policy.sleep_for(5) == 5.0


# ---------------------------------------------------------------------------
# Cache (unit)
# ---------------------------------------------------------------------------


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache = TeacherCache(path=tmp_path / "c.sqlite")
    assert cache.get("missing") is None
    cache.put("k1", "m", "hello", {"finish_reason": "stop"})
    assert cache.get("k1") == "hello"
    # Overwrite semantics.
    cache.put("k1", "m", "world")
    assert cache.get("k1") == "world"
    cache.close()


# ---------------------------------------------------------------------------
# End-to-end with MockTransport
# ---------------------------------------------------------------------------


def test_generate_sync_hits_http_once_then_cache(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        body = _ok_response("bonjour")
        return httpx.Response(200, json=body)

    client = _make_client(handler, tmp_path=tmp_path)
    try:
        out1 = client.generate_sync("hello", "mistral-large-opus")
        out2 = client.generate_sync("hello", "mistral-large-opus")
    finally:
        asyncio.run(client.aclose())

    assert out1 == "bonjour"
    assert out2 == "bonjour"
    assert len(calls) == 1, "second call must be served from cache"


def test_generate_retries_on_500_then_succeeds(tmp_path: Path) -> None:
    state = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] < 3:
            return httpx.Response(500, json={"error": "transient"})
        return httpx.Response(200, json=_ok_response("ok"))

    client = _make_client(handler, tmp_path=tmp_path)
    try:
        out = client.generate_sync("p", "mistral-large-opus")
    finally:
        asyncio.run(client.aclose())

    assert out == "ok"
    assert state["n"] == 3


def test_generate_raises_on_400_without_retry(tmp_path: Path) -> None:
    state = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        state["n"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    client = _make_client(handler, tmp_path=tmp_path)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            client.generate_sync("p", "mistral-large-opus")
    finally:
        asyncio.run(client.aclose())

    assert state["n"] == 1, "4xx must not be retried"


def test_generate_exhausts_retries_on_persistent_500(tmp_path: Path) -> None:
    state = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        state["n"] += 1
        return httpx.Response(500, json={"error": "down"})

    client = _make_client(handler, tmp_path=tmp_path)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            client.generate_sync("p", "mistral-large-opus")
    finally:
        asyncio.run(client.aclose())

    assert state["n"] == 3, "must try exactly max_attempts=3 times"


def test_qwen3_thinking_false_sets_chat_template_kwargs(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content.decode()))
        return httpx.Response(200, json=_ok_response("short", model="qwen3.5-35b-a3b-opus"))

    client = _make_client(
        handler,
        tmp_path=tmp_path,
        endpoints={"qwen3.5-35b-a3b-opus": "http://kxkm.local"},
    )
    try:
        client.generate_sync(
            "score this",
            "qwen3.5-35b-a3b-opus",
            params=GenerateParams(thinking=False, max_tokens=16),
        )
    finally:
        asyncio.run(client.aclose())

    assert captured, "request should have been sent"
    body = captured[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["model"] == "qwen3.5-35b-a3b-opus"
    assert body["messages"][0]["content"] == "score this"


def test_thinking_toggle_ignored_for_non_qwen3_models(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content.decode()))
        return httpx.Response(200, json=_ok_response("ok"))

    client = _make_client(handler, tmp_path=tmp_path)
    try:
        client.generate_sync(
            "hi",
            "mistral-large-opus",
            params=GenerateParams(thinking=True),
        )
    finally:
        asyncio.run(client.aclose())

    body = captured[0]
    assert "chat_template_kwargs" not in body


def test_endpoint_resolution_error(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response("ok"))

    client = _make_client(handler, tmp_path=tmp_path)
    try:
        with pytest.raises(TeacherError, match="no endpoint configured"):
            client.generate_sync("hi", "unknown-model")
    finally:
        asyncio.run(client.aclose())


def test_malformed_response_raises(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client = _make_client(handler, tmp_path=tmp_path)
    try:
        with pytest.raises(TeacherError, match="malformed"):
            client.generate_sync("p", "mistral-large-opus")
    finally:
        asyncio.run(client.aclose())


def test_cache_disabled_always_hits_http(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=_ok_response(f"r{calls['n']}"))

    client = _make_client(handler, tmp_path=tmp_path)
    try:
        client.generate_sync("p", "mistral-large-opus", use_cache=False)
        client.generate_sync("p", "mistral-large-opus", use_cache=False)
    finally:
        asyncio.run(client.aclose())

    assert calls["n"] == 2


def test_qwen3_model_registry_is_non_empty() -> None:
    assert "qwen3.5-35b-a3b-opus" in QWEN3_THINKING_MODELS
    assert "qwen3.5-122b-a10b-opus" in QWEN3_THINKING_MODELS
