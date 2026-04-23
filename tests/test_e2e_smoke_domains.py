"""Story-19: E2E smoke test for 35 niche domains + base fallback.

Verifies that:
1. ModelRouter.select() routes each of the 34 niche domain prompts correctly.
2. AeonPalace write + recall round-trip works for every domain.
3. All 35 niche domains get adapter != None.
4. The base query ("What is the weather?") gets adapter == None.
"""
from __future__ import annotations

import hashlib
from datetime import datetime

import numpy as np
import pytest

from src.routing.model_router import ModelRouter, RouteDecision
from src.routing.router import NICHE_DOMAINS
from src.memory.aeon import AeonPalace

# ---------------------------------------------------------------------------
# 35-domain test prompts (one per niche domain)
# ---------------------------------------------------------------------------
DOMAIN_TESTS: dict[str, str] = {
    "chat-fr":      "Explique-moi comment fonctionne un transformateur en francais.",
    "components":   "Find a JLCPCB basic part for a 100nF 0402 MLCC capacitor.",
    "cpp":          "Write a C++ template metaprogram for compile-time Fibonacci.",
    "devops":       "Write a GitHub Actions workflow for CI/CD with Docker deploy.",
    "docker":       "Create a multi-stage Dockerfile for a Python FastAPI application.",
    "dsp":          "Implement a 256-point FFT in fixed-point Q15 for Cortex-M4.",
    "electronics":  "Design an instrumentation amplifier with gain=100 using AD620.",
    "embedded":     "Implement a circular buffer in C for UART RX interrupt handler.",
    "emc":          "Design an EMI filter for USB 3.0 to meet CISPR 32 Class B.",
    "freecad":      "Write a FreeCAD macro for a parametric heatsink with fins.",
    "html-css":     "Create a responsive grid layout with Tailwind CSS for a dashboard.",
    "iot":          "Write ESP-NOW peer-to-peer communication code for sensor mesh.",
    "kicad-dsl":    "Create a KiCad S-expression for a TQFP-48 footprint with thermal pad.",
    "kicad-pcb":    "Design a 4-layer PCB stackup with controlled impedance for USB 3.0.",
    "llm-ops":      "Deploy a vLLM server with GGUF quantized model and KV cache tuning.",
    "llm-orch":     "Build a RAG pipeline with Qdrant vector store and LangChain agents.",
    "lua-upy":      "Write a MicroPython driver for I2C BME280 sensor on ESP32.",
    "math":         "Derive the Fourier transform of a Gaussian pulse analytically.",
    "ml-training":  "Configure LoRA fine-tuning with gradient checkpointing for a 7B model.",
    "music-audio":  "Implement a Web Audio synthesizer with ADSR envelope and LFO.",
    "platformio":   "Write platformio.ini for ESP32-S3 + STM32F407 multi-env build.",
    "power":        "Design a 48V to 12V synchronous buck converter at 5A.",
    "python":       "Write a pytest fixture with session-scoped async database connection.",
    "reasoning":    "Solve this step by step: if 3x + 7 = 22, find x.",
    "rust":         "Implement a lock-free concurrent queue in Rust using atomics.",
    "security":     "Audit this Flask app for OWASP Top 10 vulnerabilities.",
    "shell":        "Write a bash script to find and delete files older than 30 days.",
    "spice":        "Write a SPICE netlist for a current-mode buck converter at 500kHz.",
    "spice-sim":    "Run an AC noise analysis for the op-amp buffer I wrote in ngspice.",
    "sql":          "Write a PostgreSQL query with window functions for running totals.",
    "stm32":        "Write STM32 HAL code for DMA-based ADC on 4 channels.",
    "typescript":   "Create a type-safe React hook for API fetching with generics.",
    "web-backend":  "Build a FastAPI endpoint with Pydantic validation and rate limiting.",
    "web-frontend": "Implement a React 19 component with Suspense and error boundaries.",
    "yaml-json":    "Write an OpenAPI 3.1 schema for a REST API with JSON Schema refs.",
}

BASE_QUERY = "What is the weather?"


# ---------------------------------------------------------------------------
# Mock embed helper (deterministic, hash-based)
# ---------------------------------------------------------------------------
def _mock_embed(dim: int = 64):
    def fn(text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec = rng.randn(dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)
    return fn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def router() -> ModelRouter:
    return ModelRouter()


@pytest.fixture
def palace() -> AeonPalace:
    return AeonPalace(dim=64, embed_fn=_mock_embed(64))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestDomainRouting:
    """Verify ModelRouter routes each niche domain prompt to an adapter."""

    @pytest.mark.parametrize("domain,prompt", list(DOMAIN_TESTS.items()))
    def test_niche_domain_gets_adapter(self, router: ModelRouter, domain: str, prompt: str):
        """Each niche domain prompt should route to a non-None adapter."""
        route = router.select(prompt, domain_hint=domain)
        assert route.adapter is not None, (
            f"Domain '{domain}' should have an adapter, got None. "
            f"Route: {route}"
        )

    @pytest.mark.parametrize("domain,prompt", list(DOMAIN_TESTS.items()))
    def test_niche_domain_adapter_name(self, router: ModelRouter, domain: str, prompt: str):
        """Adapter name should contain 'stack-' prefix."""
        route = router.select(prompt, domain_hint=domain)
        assert route.adapter is not None
        assert route.adapter.startswith("stack-"), (
            f"Adapter for '{domain}' should start with 'stack-', got '{route.adapter}'"
        )

    @pytest.mark.parametrize("domain,prompt", list(DOMAIN_TESTS.items()))
    def test_niche_domain_model_id(self, router: ModelRouter, domain: str, prompt: str):
        """All niche domains should route to qwen35b."""
        route = router.select(prompt, domain_hint=domain)
        assert route.model_id == "qwen35b", (
            f"Domain '{domain}' should use qwen35b, got '{route.model_id}'"
        )

    def test_base_query_no_adapter(self, router: ModelRouter):
        """Base (non-domain) query should get adapter == None."""
        route = router.select(BASE_QUERY, domain_hint=None)
        assert route.adapter is None, (
            f"Base query should have adapter=None, got '{route.adapter}'"
        )

    def test_all_35_domains_covered(self):
        """Ensure DOMAIN_TESTS covers all 35 NICHE_DOMAINS."""
        assert set(DOMAIN_TESTS.keys()) == set(NICHE_DOMAINS), (
            f"DOMAIN_TESTS keys {set(DOMAIN_TESTS.keys())} != "
            f"NICHE_DOMAINS {set(NICHE_DOMAINS)}"
        )


class TestMemoryRoundTrip:
    """Verify AeonPalace write + recall cycle for every domain."""

    @pytest.mark.parametrize("domain,prompt", list(DOMAIN_TESTS.items()))
    def test_write_recall_cycle(self, palace: AeonPalace, domain: str, prompt: str):
        """Write an episode and recall it back by the same query text."""
        content = f"User: {prompt}\nAssistant: [mock response for {domain}]"
        episode_id = palace.write(
            content=content,
            domain=domain,
            timestamp=datetime.now(),
            source="test-e2e-smoke",
        )
        assert episode_id, "write() should return a non-empty episode ID"

        recalled = palace.recall(prompt, top_k=3)
        assert len(recalled) >= 1, (
            f"recall() for domain '{domain}' returned no episodes after write"
        )
        # The written episode should be among the recalled ones
        recalled_ids = [ep.id for ep in recalled]
        assert episode_id in recalled_ids, (
            f"Written episode {episode_id} not found in recalled IDs {recalled_ids}"
        )

    def test_base_query_write_recall(self, palace: AeonPalace):
        """Base (non-domain) query also round-trips through memory."""
        content = f"User: {BASE_QUERY}\nAssistant: I don't have weather data."
        episode_id = palace.write(
            content=content,
            domain="base",
            timestamp=datetime.now(),
            source="test-e2e-smoke",
        )
        recalled = palace.recall(BASE_QUERY, top_k=3)
        assert len(recalled) >= 1
        assert episode_id in [ep.id for ep in recalled]


class TestFullE2ESmoke:
    """Combined routing + memory smoke test for all 35 queries."""

    def test_all_domains_e2e(self, router: ModelRouter, palace: AeonPalace):
        """Run through all 35 niche domains: route, write, recall."""
        for domain, prompt in DOMAIN_TESTS.items():
            # Route
            route = router.select(prompt, domain_hint=domain)
            assert route.adapter is not None, f"No adapter for {domain}"

            # Write
            content = f"User: {prompt}\nAssistant: [response for {domain}]"
            ep_id = palace.write(
                content=content,
                domain=domain,
                timestamp=datetime.now(),
                source="test-e2e-smoke",
            )

            # Recall — use top_k=40 because 35 domain episodes compete
            recalled = palace.recall(prompt, top_k=40)
            assert any(ep.id == ep_id for ep in recalled), (
                f"Episode {ep_id} for {domain} not recalled"
            )

    def test_base_fallback_e2e(self, router: ModelRouter, palace: AeonPalace):
        """Base query: adapter=None, memory round-trip works."""
        route = router.select(BASE_QUERY, domain_hint=None)
        assert route.adapter is None

        ep_id = palace.write(
            content=f"User: {BASE_QUERY}\nAssistant: No weather data.",
            domain="base",
            timestamp=datetime.now(),
            source="test-e2e-smoke",
        )
        recalled = palace.recall(BASE_QUERY, top_k=3)
        assert any(ep.id == ep_id for ep in recalled)

    def test_adapter_none_only_for_base(self, router: ModelRouter):
        """Across all 11 queries, only the base query should get adapter=None."""
        adapters = {}
        for domain, prompt in DOMAIN_TESTS.items():
            route = router.select(prompt, domain_hint=domain)
            adapters[domain] = route.adapter

        base_route = router.select(BASE_QUERY, domain_hint=None)
        adapters["base"] = base_route.adapter

        # All 10 niches should have an adapter
        for domain in DOMAIN_TESTS:
            assert adapters[domain] is not None, f"{domain} should have adapter"

        # Base should not
        assert adapters["base"] is None, "base should have adapter=None"
