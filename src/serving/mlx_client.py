"""Async HTTP client for the Mac Studio MLX-LM server.

Used by ``scripts/measure_forgetting.py`` for the win-rate half of the
forgetting gate, and by any future eval harness that needs to generate
with / without a LoRA adapter loaded.

Wire contract (target: ``python -m mlx_lm.server`` on studio:8000):

* OpenAI-compatible endpoint: ``POST /v1/chat/completions``.
* Request body follows the standard shape
  ``{"model": str, "messages": [...], "max_tokens": int, "temperature":
  float}``; we add an optional ``adapter`` top-level key when a LoRA
  stack path is given. MLX-LM does not support runtime LoRA hot-swap
  in-process (``src/serving/AGENTS.md`` documents the subprocess-restart
  workaround); this client assumes the serving layer handles the swap
  out-of-band and simply propagates the adapter identifier.
* Response: OpenAI schema — we extract ``choices[0].message.content``.

Entry points
------------
* :class:`MLXClient` for programmatic use (async).
* Module-level :func:`generate` is a **synchronous** wrapper invoked by
  ``--generate-fn-module src.serving.mlx_client:generate``. The call
  site in :mod:`src.eval.forgetting._compute_winrate` expects a sync
  callable returning ``str``, so the wrapper bridges via
  :func:`asyncio.run` (or a fresh event loop when one is already
  running on the current thread).
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
from loguru import logger

DEFAULT_HOST = os.environ.get("MLX_HOST", "http://studio:8000")
DEFAULT_MODEL = os.environ.get("MLX_MODEL", "qwen3.6-35b")
DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 3


def _load_host_map() -> dict[str, str]:
    """Parse ``MLX_ADAPTER_HOST_MAP`` env var → ``{adapter_key: host_url}``.

    Enables the "two servers, one per adapter" production flow used by
    the forgetting gate: ``mlx_lm.server`` 0.31.2 does not hot-swap
    adapters per request, so each adapter must be served by its own
    long-running process on a distinct port. The operator spawns the
    servers, sets ``MLX_ADAPTER_HOST_MAP`` to map adapter identifiers
    to their URLs, and the sync :func:`generate` entry point routes
    automatically.

    Example::

        export MLX_ADAPTER_HOST_MAP='{"chat-fr":"http://studio:8000",
                                      "reasoning":"http://studio:8001"}'

    Empty or malformed → empty dict (falls back to ``DEFAULT_HOST``).
    """
    raw = os.environ.get("MLX_ADAPTER_HOST_MAP", "").strip()
    if not raw:
        return {}
    import json

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except json.JSONDecodeError:
        logger.warning("MLX_ADAPTER_HOST_MAP is not valid JSON, ignoring")
    return {}


def _resolve_host(adapter: str | None) -> str:
    """Pick the host URL for the given adapter identifier.

    1. Env-var map (``MLX_ADAPTER_HOST_MAP``) — match by full adapter
       key, then by basename (``.../chat-fr/adapters.safetensors`` →
       ``chat-fr``).
    2. Fall back to ``DEFAULT_HOST`` / ``MLX_HOST``.
    """
    if adapter is None:
        return DEFAULT_HOST
    host_map = _load_host_map()
    if not host_map:
        return DEFAULT_HOST
    if adapter in host_map:
        return host_map[adapter]
    # Try basename variants: ".../stacks/chat-fr/adapters.safetensors" → "chat-fr"
    from pathlib import Path

    stem_candidates = {Path(adapter).stem, Path(adapter).parent.name}
    for key, url in host_map.items():
        if key in stem_candidates:
            return url
    return DEFAULT_HOST


# Per-host singleton cache (one ``MLXClient`` per host URL).
_CLIENTS: "dict[str, MLXClient]" = {}


class MLXClient:
    """Async client for the MLX-LM OpenAI-compatible server."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._transport = transport  # test hook; None → real network

    def _build_payload(
        self,
        prompt: str,
        adapter: str | None,
        *,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if adapter is not None:
            # MLX-LM extension — echoed by the operator runbook that
            # restarts the subprocess with ``--adapter-path <adapter>``
            # (see ``src/serving/mlx_server.py``). Field is omitted
            # entirely when no adapter is requested (base-model run).
            payload["adapter"] = adapter
        return payload

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        """Pull ``choices[0].message.content`` out of an OpenAI response."""
        try:
            choices = data["choices"]
            first = choices[0]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"MLX server response missing choices: {data!r}") from exc
        # Chat-style
        msg = first.get("message") if isinstance(first, dict) else None
        if isinstance(msg, dict) and "content" in msg:
            return str(msg["content"])
        # Fallback: /v1/completions-style (text field)
        if isinstance(first, dict) and "text" in first:
            return str(first["text"])
        raise ValueError(f"MLX server response missing content: {data!r}")

    async def generate(
        self,
        prompt: str,
        adapter: str | None = None,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Generate a completion for ``prompt``.

        Args:
            prompt: user message to send.
            adapter: optional LoRA identifier. When ``None``, the base
                model is used (no ``adapter`` key in the body).
            max_tokens: generation cap; kept modest for eval batches.
            temperature: deterministic by default (0.0) — win-rate
                scoring does not want sampling noise.

        Retries up to :data:`MAX_RETRIES` on ``httpx.ConnectError`` or
        any HTTP 5xx response with exponential backoff
        (``2 ** attempt`` seconds). Non-5xx HTTP errors raise
        immediately. After the final failure the underlying exception
        propagates to the caller.
        """
        url = f"{self.host}/v1/chat/completions"
        payload = self._build_payload(
            prompt, adapter, max_tokens=max_tokens, temperature=temperature
        )

        client_kwargs: dict[str, Any] = {"timeout": self.timeout}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport

        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                async with httpx.AsyncClient(**client_kwargs) as client:
                    response = await client.post(url, json=payload)
                    if 500 <= response.status_code < 600:
                        # Raise a classified error so the retry loop
                        # below can decide to back off. We do NOT use
                        # ``response.raise_for_status()`` here because
                        # we want to preserve the response object for
                        # ``HTTPStatusError.response.status_code``.
                        raise httpx.HTTPStatusError(
                            f"MLX server returned {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                    response.raise_for_status()
                    return self._extract_text(response.json())
            except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
                # Non-5xx HTTPStatusError: give up immediately.
                if isinstance(exc, httpx.HTTPStatusError) and not (
                    500 <= exc.response.status_code < 600
                ):
                    raise
                last_exc = exc
                if attempt == MAX_RETRIES - 1:
                    logger.debug(
                        "MLX client giving up after {} attempts: {}",
                        MAX_RETRIES,
                        exc,
                    )
                    raise
                backoff = 2 ** attempt
                logger.debug(
                    "MLX client retrying after {} (attempt {}/{}): {}",
                    backoff,
                    attempt + 1,
                    MAX_RETRIES,
                    exc,
                )
                await asyncio.sleep(backoff)
        # Unreachable: the loop either returns or raises.
        raise RuntimeError("MLX client retry loop exited without result") from last_exc


async def agenerate(
    prompt: str,
    adapter: str | None = None,
    *,
    host: str | None = None,
    **kwargs: Any,
) -> str:
    """Async module-level entry point.

    ``host`` override takes precedence over the ``MLX_ADAPTER_HOST_MAP``
    routing; when neither is set we fall back to ``DEFAULT_HOST``.
    """
    resolved = host or _resolve_host(adapter)
    client = _CLIENTS.get(resolved)
    if client is None:
        client = MLXClient(host=resolved)
        _CLIENTS[resolved] = client
    return await client.generate(prompt, adapter=adapter, **kwargs)


def generate(
    prompt: str,
    adapter: str | None = None,
    *,
    host: str | None = None,
    **kwargs: Any,
) -> str:
    """Sync entry point for ``--generate-fn-module src.serving.mlx_client:generate``.

    ``src.eval.forgetting._compute_winrate`` calls ``generate_fn`` in
    synchronous context, so we bridge async→sync here. When an event
    loop is already running on the calling thread we spin up a fresh
    one via ``asyncio.new_event_loop()`` to avoid nesting violations.
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    if running is None:
        return asyncio.run(agenerate(prompt, adapter=adapter, host=host, **kwargs))

    # Already inside an event loop — run the coroutine on a fresh one
    # in a worker thread. This path is mostly defensive; the forgetting
    # CLI is a plain sync main() so the common case is the ``None``
    # branch above.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            agenerate(prompt, adapter=adapter, host=host, **kwargs)
        )
    finally:
        loop.close()
