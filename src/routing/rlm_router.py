from __future__ import annotations

"""RLM (Recursive LM) domain router.

Uses the base 35B model itself to decompose complex queries into
domain-specific sub-queries before adapter routing.  This replaces the
MiniLM sigmoid classifier for multi-domain queries while keeping it as
a fast-path fallback.

The decomposition call uses ``raw_mode=true`` (no cognitive layer),
``max_tokens=200``, and ``temperature=0.1`` for deterministic structured
JSON output.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from src.routing.router import NICHE_DOMAINS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain allow-list (frozen copy for validation)
# ---------------------------------------------------------------------------

_VALID_DOMAINS: frozenset[str] = NICHE_DOMAINS

# ---------------------------------------------------------------------------
# Decomposition system prompt
# ---------------------------------------------------------------------------

_DECOMPOSE_SYSTEM = (
    "You are a query decomposer. Given a user query, identify the technical "
    "domains it spans and split it into domain-specific sub-queries.\n\n"
    "IMPORTANT: Your ENTIRE response must be a single JSON object. "
    "No markdown, no explanation, no code fences. Just the JSON.\n\n"
    "Available domains: " + ", ".join(sorted(_VALID_DOMAINS)) + "\n\n"
    "If the query is simple (single domain), return a single entry. "
    "If complex, decompose into 2-4 sub-queries.\n\n"
    'Output format: {"sub_queries": [{"domain": "...", "query": "..."}]}\n\n'
    "Examples:\n\n"
    'User: "Design a battery-powered STM32 system with CAN bus, EMC compliance, '
    'and a custom KiCad PCB layout"\n'
    'Output: {"sub_queries": [\n'
    '  {"domain": "stm32", "query": "Design an STM32 microcontroller system with CAN bus peripheral configuration"},\n'
    '  {"domain": "power", "query": "Design a battery power supply circuit for an embedded system"},\n'
    '  {"domain": "emc", "query": "EMC compliance strategy for a mixed-signal PCB with CAN bus"},\n'
    '  {"domain": "kicad-pcb", "query": "KiCad PCB layout for a 4-layer STM32 board with CAN and power section"}\n'
    "]}\n\n"
    'User: "Write a Python script that reads I2C sensor data on ESP32 via PlatformIO"\n'
    'Output: {"sub_queries": [\n'
    '  {"domain": "python", "query": "Write a Python script to read I2C sensor data and process the values"},\n'
    '  {"domain": "platformio", "query": "Configure a PlatformIO project for ESP32 with I2C library dependencies"}\n'
    "]}\n\n"
    'User: "Explain the Rust borrow checker"\n'
    'Output: {"sub_queries": [\n'
    '  {"domain": "rust", "query": "Explain the Rust borrow checker with ownership and lifetime examples"}\n'
    "]}"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SERVER_URL = "http://127.0.0.1:9200"
_DEFAULT_TIMEOUT_S = 10.0
_MAX_TOKENS = 200
_TEMPERATURE = 0.1
_MAX_CACHE_SIZE = 256


# ---------------------------------------------------------------------------
# Fallback router protocol
# ---------------------------------------------------------------------------


class FallbackRouter(Protocol):
    """Minimal interface for the MiniLM sigmoid fallback."""

    def route(self, query: str) -> list[tuple[str, float, str]]:
        """Return ``[(domain, confidence, sub_query), ...]``."""
        ...


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CacheEntry:
    result: tuple[tuple[str, float, str], ...]
    timestamp: float


# ---------------------------------------------------------------------------
# RLM Router
# ---------------------------------------------------------------------------


@dataclass
class RLMRouter:
    """Recursive LM router — decomposes queries via the base 35B model.

    Parameters
    ----------
    server_url:
        Base URL of the micro-kiki server (OpenAI-compatible endpoint).
    fallback_router:
        Optional fallback used when decomposition fails or times out.
    timeout_s:
        HTTP timeout in seconds for the decomposition call.
    """

    server_url: str = _DEFAULT_SERVER_URL
    fallback_router: FallbackRouter | None = None
    timeout_s: float = _DEFAULT_TIMEOUT_S
    _cache: dict[str, _CacheEntry] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, query: str) -> list[tuple[str, float, str]]:
        """Route *query* and return ``[(domain, confidence, sub_query), ...]``.

        For single-domain queries the list has one entry.  For multi-domain
        queries it has 2-4 entries, each with ``confidence=1.0``.  Falls back
        to *fallback_router* (or a single ``("chat-fr", 0.0, query)`` entry)
        on any failure.
        """
        cached = self._cache_get(query)
        if cached is not None:
            return list(cached)

        try:
            subs = self.decompose(query)
        except Exception:
            logger.warning("RLM decomposition failed, using fallback", exc_info=True)
            return self._fallback(query)

        if not subs:
            return self._fallback(query)

        result = _subs_to_route(subs, query)
        self._cache_put(query, result)
        return result

    def decompose(self, query: str) -> list[dict[str, str]]:
        """Call the base model to decompose *query* into sub-queries.

        Returns a list of ``{"domain": "...", "query": "..."}`` dicts.
        Raises on HTTP/JSON failure (caller should catch).
        """
        payload = {
            "model": "default",
            "messages": [
                {"role": "system", "content": _DECOMPOSE_SYSTEM},
                {"role": "user", "content": query},
            ],
            "max_tokens": _MAX_TOKENS,
            "temperature": _TEMPERATURE,
            "raw_mode": True,
            "strip_thinking": True,
        }

        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()

        body = resp.json()
        content = _extract_content(body)
        return _parse_decomposition(content)

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback(self, query: str) -> list[tuple[str, float, str]]:
        if self.fallback_router is not None:
            try:
                return self.fallback_router.route(query)
            except Exception:
                logger.warning("Fallback router also failed", exc_info=True)
        return [("chat-fr", 0.0, query)]

    # ------------------------------------------------------------------
    # LRU-ish cache
    # ------------------------------------------------------------------

    def _cache_get(self, query: str) -> tuple[tuple[str, float, str], ...] | None:
        entry = self._cache.get(query)
        if entry is None:
            return None
        return entry.result

    def _cache_put(
        self,
        query: str,
        result: list[tuple[str, float, str]],
    ) -> None:
        if len(self._cache) >= _MAX_CACHE_SIZE:
            oldest_key = min(self._cache, key=lambda k: self._cache[k].timestamp)
            del self._cache[oldest_key]
        self._cache[query] = _CacheEntry(
            result=tuple(result),
            timestamp=time.monotonic(),
        )


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------


def _extract_content(body: dict) -> str:
    """Pull the assistant message content from an OpenAI-style response."""
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected response shape: {body!r}") from exc


def _parse_decomposition(text: str) -> list[dict[str, str]]:
    """Parse the model output into a list of sub-query dicts.

    Tries ``json.loads`` first, then falls back to regex extraction for
    slightly malformed JSON (common with constrained-token generation).
    """
    text = text.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # Attempt 1: direct parse
    try:
        parsed = json.loads(text)
        return _validate_subs(parsed)
    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    # Attempt 2: find the JSON object via regex
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return _validate_subs(parsed)
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    raise ValueError(f"Could not parse decomposition from model output: {text!r}")


def _validate_subs(parsed: object) -> list[dict[str, str]]:
    """Validate and normalise the parsed JSON into canonical sub-query dicts."""
    if isinstance(parsed, dict):
        subs = parsed.get("sub_queries", [])
    elif isinstance(parsed, list):
        subs = parsed
    else:
        raise TypeError(f"Expected dict or list, got {type(parsed).__name__}")

    if not isinstance(subs, list) or len(subs) == 0:
        raise ValueError("Empty sub_queries list")

    result: list[dict[str, str]] = []
    for entry in subs:
        if not isinstance(entry, dict):
            continue
        domain = str(entry.get("domain", "")).strip().lower()
        query = str(entry.get("query", "")).strip()
        if not domain or not query:
            continue
        # Clamp unknown domains to the nearest valid one or skip
        if domain not in _VALID_DOMAINS:
            logger.debug("Unknown domain %r from decomposition, skipping", domain)
            continue
        result.append({"domain": domain, "query": query})

    if not result:
        raise ValueError("No valid sub-queries after validation")
    return result


def _subs_to_route(
    subs: list[dict[str, str]],
    original_query: str,
) -> list[tuple[str, float, str]]:
    """Convert validated sub-query dicts to the route tuple format."""
    if len(subs) == 1:
        return [(subs[0]["domain"], 1.0, original_query)]
    return [(s["domain"], 1.0, s["query"]) for s in subs]
