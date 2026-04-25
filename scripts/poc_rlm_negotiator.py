#!/usr/bin/env python3
"""PoC: RLM recursive decomposition as Negotiator replacement.

Connects to the micro-kiki server (port 9200, OpenAI-compat) and compares:
- Direct inference (single domain, current behavior)
- RLM recursive decomposition (multi-domain, proposed replacement)

Usage:
    uv run python scripts/poc_rlm_negotiator.py
    uv run python scripts/poc_rlm_negotiator.py --base-url http://192.168.0.210:9200
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx
from rlm import RLM
from rlm.core.types import RLMChatCompletion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("poc-rlm")

# ── Test prompts ──────────────────────────────────────────────────────────

PROMPTS: list[dict[str, str]] = [
    {
        "label": "single-domain-spice",
        "expected_domains": "spice",
        "prompt": "Write a SPICE netlist for a Butterworth low-pass filter at 10kHz",
    },
    {
        "label": "multi-domain-stm32-can-emc-kicad",
        "expected_domains": "stm32,power,emc,kicad-pcb",
        "prompt": (
            "Design a battery-powered STM32 system with CAN bus, "
            "EMC compliance, and a custom KiCad PCB layout"
        ),
    },
    {
        "label": "multi-domain-esp32-i2c-mqtt",
        "expected_domains": "platformio,embedded,iot",
        "prompt": (
            "Build a PlatformIO project for ESP32 that reads I2C sensors "
            "and sends data over MQTT with power optimization"
        ),
    },
    {
        "label": "cross-domain-freecad-spice",
        "expected_domains": "freecad,spice-sim",
        "prompt": (
            "Create a FreeCAD parametric model of a heatsink, then simulate "
            "its thermal performance in SPICE"
        ),
    },
    {
        "label": "single-domain-conversational",
        "expected_domains": "electronics",
        "prompt": "Explique les avantages du protocole I2C par rapport au SPI",
    },
]

# ── Domain list (from src/routing/router.py NICHE_DOMAINS) ───────────────

DOMAINS: list[str] = sorted([
    "chat-fr", "components", "cpp", "devops", "docker", "dsp",
    "electronics", "embedded", "emc", "freecad", "html-css", "iot",
    "kicad-dsl", "kicad-pcb", "llm-ops", "llm-orch", "lua-upy", "math",
    "ml-training", "music-audio", "platformio", "power", "python",
    "reasoning", "rust", "security", "shell", "spice", "spice-sim",
    "sql", "stm32", "typescript", "web-backend", "web-frontend",
    "yaml-json",
])

# ── RLM system prompt for recursive decomposition ────────────────────────

RLM_DECOMPOSE_SYSTEM = """You are a multi-domain engineering assistant with access to 35 specialized domain adapters. When given a complex query spanning multiple technical domains, you MUST:

1. Identify the distinct sub-problems (one per domain)
2. For each sub-problem, call yourself with a focused query using rlm_query()
3. Compose the sub-answers into a coherent final response

Available domains: {domains}

If the query is simple (single domain), answer directly without decomposition.

When decomposing, use this pattern in a Python code block:
```python
sub1 = rlm_query("focused sub-question about domain X")
sub2 = rlm_query("focused sub-question about domain Y")
# ... then compose results
```

IMPORTANT: Each sub-query should be self-contained and specify the domain context clearly.""".format(
    domains=", ".join(DOMAINS)
)


@dataclass(frozen=True)
class RouteResult:
    """Result from /v1/route endpoint."""

    query: str
    domains: list[dict[str, Any]]
    fallback: bool
    latency_ms: float


@dataclass(frozen=True)
class InferenceResult:
    """Result from either direct inference or RLM recursive call."""

    method: str  # "direct" | "rlm-recursive"
    response: str
    domains_used: list[str]
    token_estimate: int
    latency_ms: float
    decomposition_depth: int
    metadata: dict[str, Any] | None = None


def route_query(
    client: httpx.Client,
    base_url: str,
    query: str,
    threshold: float = 0.12,
    top_k: int = 4,
) -> RouteResult:
    """Call /v1/route to classify a prompt into domains."""
    t0 = time.perf_counter()
    resp = client.post(
        f"{base_url}/v1/route",
        json={"query": query, "threshold": threshold, "top_k": top_k},
        timeout=10.0,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    data = resp.json()
    return RouteResult(
        query=data.get("query", query),
        domains=data.get("domains", []),
        fallback=data.get("fallback", True),
        latency_ms=elapsed_ms,
    )


def direct_inference(
    client: httpx.Client,
    base_url: str,
    prompt: str,
    model: str = "kiki-meta-coding",
) -> InferenceResult:
    """Single-pass inference via /v1/chat/completions."""
    t0 = time.perf_counter()
    resp = client.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.7,
            "strip_thinking": True,
            "raw_mode": True,
        },
        timeout=120.0,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    data = resp.json()

    content = ""
    if "choices" in data and data["choices"]:
        content = data["choices"][0].get("message", {}).get("content", "")

    usage = data.get("usage", {})
    total_tokens = usage.get("total_tokens", len(content) // 4)

    return InferenceResult(
        method="direct",
        response=content,
        domains_used=[model],
        token_estimate=total_tokens,
        latency_ms=elapsed_ms,
        decomposition_depth=0,
    )


def rlm_recursive_inference(
    base_url: str,
    prompt: str,
    model_name: str = "kiki-meta-coding",
) -> InferenceResult:
    """RLM recursive decomposition via the rlm library."""
    t0 = time.perf_counter()

    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "base_url": f"{base_url}/v1",
            "api_key": "unused",
            "model_name": model_name,
        },
        max_depth=1,
        max_iterations=5,
        custom_system_prompt=RLM_DECOMPOSE_SYSTEM,
        verbose=False,
    )

    result: RLMChatCompletion = rlm.completion(prompt)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    response_text = result.response
    metadata_dict = result.to_dict() if hasattr(result, "to_dict") else None

    # Estimate tokens from usage summary
    usage = result.usage_summary
    total_tokens = usage.total_input_tokens + usage.total_output_tokens

    # Infer decomposition depth from metadata
    depth = 0
    if metadata_dict and metadata_dict.get("metadata"):
        iterations = metadata_dict["metadata"].get("iterations", [])
        for it in iterations:
            if it.get("code_blocks"):
                depth = max(depth, 1)
                break

    return InferenceResult(
        method="rlm-recursive",
        response=response_text,
        domains_used=["rlm-root"],
        token_estimate=total_tokens,
        latency_ms=elapsed_ms,
        decomposition_depth=depth,
        metadata=metadata_dict,
    )


def run_comparison(base_url: str) -> list[dict[str, Any]]:
    """Run all test prompts through both methods and collect results."""
    results: list[dict[str, Any]] = []

    with httpx.Client() as client:
        for case in PROMPTS:
            label = case["label"]
            prompt = case["prompt"]
            expected = case["expected_domains"]

            logger.info("=" * 70)
            logger.info("Test: %s", label)
            logger.info("Prompt: %s", prompt[:80])
            logger.info("Expected domains: %s", expected)

            # Step 1: Route the query
            try:
                route = route_query(client, base_url, prompt)
                detected_domains = [d["name"] for d in route.domains]
                logger.info(
                    "Router: %s (%.1fms, fallback=%s)",
                    detected_domains,
                    route.latency_ms,
                    route.fallback,
                )
            except Exception as exc:
                logger.error("Router failed: %s", exc)
                detected_domains = []
                route = None

            # Use expected_domains as ground truth for multi-domain detection.
            # The MiniLM router is single-label (CrossEntropy) and rarely
            # returns 2+ domains.  The RLM's value is precisely that it can
            # decompose WITHOUT relying on the router's multi-label signal.
            expected_count = len([d for d in expected.split(",") if d.strip()])
            is_multi_domain = expected_count >= 2

            # Step 2a: Direct inference (current behavior)
            try:
                direct = direct_inference(client, base_url, prompt)
                logger.info(
                    "Direct: %d tokens, %.1fms",
                    direct.token_estimate,
                    direct.latency_ms,
                )
            except Exception as exc:
                logger.error("Direct inference failed: %s", exc)
                direct = None

            # Step 2b: RLM recursive (proposed replacement) — only for multi-domain
            rlm_result = None
            if is_multi_domain:
                try:
                    rlm_result = rlm_recursive_inference(base_url, prompt)
                    logger.info(
                        "RLM: %d tokens, %.1fms, depth=%d",
                        rlm_result.token_estimate,
                        rlm_result.latency_ms,
                        rlm_result.decomposition_depth,
                    )
                except Exception as exc:
                    logger.error("RLM recursive failed: %s", exc)
            else:
                logger.info("Single domain — skipping RLM (pass-through)")

            # Step 3: Collect comparison
            entry: dict[str, Any] = {
                "label": label,
                "prompt": prompt,
                "expected_domains": expected,
                "detected_domains": detected_domains,
                "is_multi_domain": is_multi_domain,
                "route_latency_ms": route.latency_ms if route else None,
            }

            if direct is not None:
                entry["direct"] = {
                    "response_preview": direct.response[:200],
                    "tokens": direct.token_estimate,
                    "latency_ms": round(direct.latency_ms, 1),
                }

            if rlm_result is not None:
                entry["rlm"] = {
                    "response_preview": rlm_result.response[:200],
                    "tokens": rlm_result.token_estimate,
                    "latency_ms": round(rlm_result.latency_ms, 1),
                    "decomposition_depth": rlm_result.decomposition_depth,
                }

            results.append(entry)

    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY: Direct Inference vs RLM Recursive Decomposition")
    print("=" * 80)

    for entry in results:
        print(f"\n--- {entry['label']} ---")
        print(f"  Domains detected: {entry['detected_domains']}")
        print(f"  Multi-domain: {entry['is_multi_domain']}")

        if "direct" in entry:
            d = entry["direct"]
            print(f"  Direct:  {d['tokens']} tokens, {d['latency_ms']}ms")
            print(f"           {d['response_preview'][:100]}...")

        if "rlm" in entry:
            r = entry["rlm"]
            print(f"  RLM:     {r['tokens']} tokens, {r['latency_ms']}ms, depth={r['decomposition_depth']}")
            print(f"           {r['response_preview'][:100]}...")
        elif entry["is_multi_domain"]:
            print("  RLM:     FAILED")
        else:
            print("  RLM:     skipped (single domain)")

    # Aggregate stats for multi-domain cases
    multi_cases = [e for e in results if e["is_multi_domain"] and "direct" in e and "rlm" in e]
    if multi_cases:
        print(f"\n{'=' * 80}")
        print("MULTI-DOMAIN AGGREGATE (RLM vs Direct)")
        print(f"{'=' * 80}")
        direct_tokens = sum(e["direct"]["tokens"] for e in multi_cases)
        rlm_tokens = sum(e["rlm"]["tokens"] for e in multi_cases)
        direct_latency = sum(e["direct"]["latency_ms"] for e in multi_cases)
        rlm_latency = sum(e["rlm"]["latency_ms"] for e in multi_cases)

        print(f"  Total tokens — Direct: {direct_tokens}, RLM: {rlm_tokens} "
              f"(ratio: {rlm_tokens / direct_tokens:.2f}x)" if direct_tokens > 0 else "")
        print(f"  Total latency — Direct: {direct_latency:.0f}ms, RLM: {rlm_latency:.0f}ms "
              f"(ratio: {rlm_latency / direct_latency:.2f}x)" if direct_latency > 0 else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="PoC: RLM vs Negotiator comparison")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:9200",
        help="micro-kiki server base URL (default: http://127.0.0.1:9200)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON results (default: stdout summary only)",
    )
    args = parser.parse_args()

    logger.info("micro-kiki RLM vs Negotiator PoC")
    logger.info("Server: %s", args.base_url)

    results = run_comparison(args.base_url)
    print_summary(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
