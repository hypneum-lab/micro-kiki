"""OpenAI-compatible async teacher client with on-disk cache + retries.

This module wraps the small set of large "teacher" models micro-kiki
distills from. All teachers expose an OpenAI ``/v1/chat/completions``
compatible HTTP surface (mlx-lm server on Studio, vLLM / llama.cpp on
kxkm-ai).

Features
--------
* Async HTTP via :mod:`httpx`; sync wrapper provided for test ergonomics.
* Exponential backoff retry (max 3 attempts, base 1 s, jitter).
* On-disk SQLite response cache keyed by
  ``SHA256(prompt || model || sorted(params))``.
* Qwen3 "thinking" mode toggle — the client emits
  ``chat_template_kwargs.enable_thinking = False`` when ``thinking`` is
  disabled (short outputs, scoring, classification).
* Endpoint URLs read from env vars (``TEACHER_MISTRAL_URL``,
  ``TEACHER_QWEN122_URL``, ``TEACHER_QWEN35_URL``, ``TEACHER_DEVSTRAL_URL``)
  with sensible cluster defaults. Override in tests via the constructor.

Design rationale
----------------
* SQLite (not JSONL) because the cache is query-heavy (SELECT by hash)
  and we want atomic writes for free. Single-file, stdlib, no server.
* We retry on network errors and HTTP 5xx; 4xx responses fail fast —
  a malformed request won't fix itself.
* The cache stores the full completion text *plus* the response metadata
  (model, usage) so we can reconstruct stats without a replay.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import random
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


#: Canonical teacher model names used inside micro-kiki. Each maps to an
#: env var holding the base URL of the OpenAI-compatible HTTP endpoint.
DEFAULT_ENDPOINTS: dict[str, str] = {
    "mistral-large-opus": "TEACHER_MISTRAL_URL",
    "qwen3.5-122b-a10b-opus": "TEACHER_QWEN122_URL",
    "qwen3.5-35b-a3b-opus": "TEACHER_QWEN35_URL",
    "devstral-v3": "TEACHER_DEVSTRAL_URL",
    "devstral-v4": "TEACHER_DEVSTRAL_URL",
}

#: Models in the Qwen3 family that support the ``enable_thinking`` toggle
#: via ``chat_template_kwargs``. Mistral / Devstral ignore it.
QWEN3_THINKING_MODELS: frozenset[str] = frozenset(
    {
        "qwen3.5-122b-a10b-opus",
        "qwen3.5-35b-a3b-opus",
    }
)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


DEFAULT_CACHE_PATH = Path.home() / ".cache" / "micro-kiki" / "teacher-cache.sqlite"


class TeacherCache:
    """Tiny SQLite-backed key-value cache for teacher responses.

    The cache is intentionally small-surface: one table, one index. We
    don't bother with a schema migration because the hash covers every
    user-visible input.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_CACHE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, isolation_level=None)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            "  hash TEXT PRIMARY KEY,"
            "  model TEXT NOT NULL,"
            "  completion TEXT NOT NULL,"
            "  meta_json TEXT NOT NULL,"
            "  created_at REAL NOT NULL"
            ")"
        )

    def get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT completion FROM responses WHERE hash = ?", (key,)
        ).fetchone()
        return None if row is None else row[0]

    def put(
        self,
        key: str,
        model: str,
        completion: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO responses "
            "(hash, model, completion, meta_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                key,
                model,
                completion,
                json.dumps(meta or {}, sort_keys=True, ensure_ascii=False),
                time.time(),
            ),
        )

    def close(self) -> None:
        self._conn.close()


def cache_key(prompt: str, model: str, params: dict[str, Any]) -> str:
    """Return the SHA-256 cache key for a (prompt, model, params) triple."""

    payload = {
        "prompt": prompt,
        "model": model,
        "params": params,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


@dataclass
class RetryPolicy:
    """Exponential backoff with jitter.

    Defaults: 3 attempts, 1 s base delay, factor 2, up to 30 s, ±25 %
    jitter. Only transient errors trigger a retry (network errors and
    HTTP 5xx responses).
    """

    max_attempts: int = 3
    base_delay_s: float = 1.0
    factor: float = 2.0
    max_delay_s: float = 30.0
    jitter: float = 0.25

    def sleep_for(self, attempt: int) -> float:
        """Return the pre-attempt sleep for ``attempt`` (1-indexed)."""
        if attempt <= 1:
            return 0.0
        raw = self.base_delay_s * (self.factor ** (attempt - 2))
        capped = min(raw, self.max_delay_s)
        if self.jitter > 0:
            spread = capped * self.jitter
            capped = capped + random.uniform(-spread, spread)
        return max(0.0, capped)


class TeacherError(RuntimeError):
    """Raised when a teacher request fails permanently."""


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    return isinstance(exc, (httpx.TransportError, httpx.TimeoutException))


async def _with_retry(
    coro_factory: Callable[[], Awaitable[Any]],
    policy: RetryPolicy,
    *,
    sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Any:
    last_exc: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        delay = policy.sleep_for(attempt)
        if delay > 0:
            await sleeper(delay)
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001 — we classify below
            last_exc = exc
            if not _is_transient(exc) or attempt == policy.max_attempts:
                raise
            logger.warning(
                "teacher call transient failure (attempt %d/%d): %s",
                attempt,
                policy.max_attempts,
                exc,
            )
    # Unreachable: the loop either returns or raises.
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@dataclass
class GenerateParams:
    """Per-request knobs. Mirrors the OpenAI chat-completion schema."""

    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    # Qwen3 thinking toggle. ``None`` lets the model default apply; for
    # non-Qwen3 models this field is silently ignored.
    thinking: bool | None = None
    # Free-form extra fields merged into the JSON payload.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.thinking is not None:
            out["thinking"] = self.thinking
        if self.extra:
            out["extra"] = self.extra
        return out


class TeacherClient:
    """Async OpenAI-compatible teacher client with cache + retries."""

    def __init__(
        self,
        *,
        endpoints: dict[str, str] | None = None,
        cache: TeacherCache | None = None,
        retry: RetryPolicy | None = None,
        timeout_s: float = 120.0,
        http_client: httpx.AsyncClient | None = None,
        api_key: str | None = None,
    ) -> None:
        self._endpoints = dict(endpoints) if endpoints is not None else self._load_env_endpoints()
        self._cache = cache if cache is not None else TeacherCache()
        self._retry = retry or RetryPolicy()
        self._timeout_s = timeout_s
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=timeout_s)
        self._api_key = api_key or os.environ.get("TEACHER_API_KEY", "")

    @staticmethod
    def _load_env_endpoints() -> dict[str, str]:
        out: dict[str, str] = {}
        for model, env_var in DEFAULT_ENDPOINTS.items():
            url = os.environ.get(env_var)
            if url:
                out[model] = url
        return out

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
        self._cache.close()

    async def __aenter__(self) -> "TeacherClient":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_endpoint(self, model: str) -> str:
        try:
            return self._endpoints[model]
        except KeyError as exc:
            raise TeacherError(
                f"no endpoint configured for teacher model {model!r}; "
                f"set one of {sorted(DEFAULT_ENDPOINTS.values())}"
            ) from exc

    def _build_payload(
        self,
        prompt: str,
        model: str,
        params: GenerateParams,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
        }
        if model in QWEN3_THINKING_MODELS and params.thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": bool(params.thinking)}
        if params.extra:
            body.update(params.extra)
        return body

    async def generate(
        self,
        prompt: str,
        model: str,
        params: GenerateParams | None = None,
        *,
        use_cache: bool = True,
    ) -> str:
        """Return a single completion string for ``prompt`` from ``model``.

        Hits the on-disk cache first; on miss, performs an HTTP round
        trip under the retry policy and stores the result.
        """

        params = params or GenerateParams()
        key = cache_key(prompt, model, params.to_dict())
        if use_cache:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        url = self.resolve_endpoint(model).rstrip("/") + "/v1/chat/completions"
        payload = self._build_payload(prompt, model, params)
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async def _do_request() -> httpx.Response:
            resp = await self._client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp

        response = await _with_retry(_do_request, self._retry)
        text, meta = self._extract_completion(response.json())
        self._cache.put(key, model, text, meta)
        return text

    def generate_sync(
        self,
        prompt: str,
        model: str,
        params: GenerateParams | None = None,
        *,
        use_cache: bool = True,
    ) -> str:
        """Synchronous wrapper around :meth:`generate` for tests / scripts."""
        return asyncio.run(
            self.generate(prompt, model, params=params, use_cache=use_cache)
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_completion(
        payload: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Pull the first message's content from an OpenAI-compatible body.

        Qwen3 thinking-mode responses nest reasoning_content alongside
        content; we return only ``content``. Meta carries usage +
        finish_reason for telemetry.
        """
        try:
            choice = payload["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise TeacherError(f"malformed teacher response: {payload!r}") from exc
        content = message.get("content")
        if not isinstance(content, str):
            raise TeacherError(
                f"teacher response missing string content: {message!r}"
            )
        meta = {
            "finish_reason": choice.get("finish_reason"),
            "usage": payload.get("usage", {}),
            "model": payload.get("model"),
        }
        return content, meta


# ---------------------------------------------------------------------------
# Dataclass helper for `src/distill/__init__.py`
# ---------------------------------------------------------------------------


def fields(obj: Any) -> list[str]:
    """Return dataclass field names — small helper kept for test clarity."""
    return [f.name for f in dataclasses.fields(obj)]
