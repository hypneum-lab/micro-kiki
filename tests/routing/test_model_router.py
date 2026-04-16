from __future__ import annotations

import pytest

from src.routing.model_router import ModelRouter, RouteDecision


@pytest.fixture()
def router() -> ModelRouter:
    return ModelRouter()


def test_simple_query_routes_to_qwen35b_base(router: ModelRouter):
    decision = router.select("Explique-moi les réseaux de neurones")
    assert decision.model_id == "qwen35b"
    assert decision.adapter is None


def test_require_deep_routes_to_qwen480b(router: ModelRouter):
    decision = router.select("Analyse ce circuit", require_deep=True)
    assert decision.model_id == "qwen480b"
    assert decision.adapter is None


def test_niche_domain_hint_routes_to_qwen35b_with_adapter(router: ModelRouter):
    decision = router.select("Génère un footprint", domain_hint="kicad-dsl")
    assert decision.model_id == "qwen35b"
    assert decision.adapter == "stack-kicad-dsl"


def test_code_domain_routes_to_devstral(router: ModelRouter):
    decision = router.select("Write a Python function to parse CAN frames")
    assert decision.model_id == "devstral"
    assert decision.adapter is None


def test_unknown_domain_hint_routes_to_qwen35b_base(router: ModelRouter):
    decision = router.select("Quelque chose", domain_hint="unknown-domain-xyz")
    assert decision.model_id == "qwen35b"
    assert decision.adapter is None


def test_route_decision_is_frozen():
    d = RouteDecision(model_id="qwen35b", adapter=None, reason="test")
    with pytest.raises(Exception):
        d.model_id = "other"  # type: ignore[misc]


def test_deep_reasoning_overrides_niche_domain(router: ModelRouter):
    """require_deep takes priority even when a niche domain_hint is given."""
    decision = router.select("Analyse profonde", domain_hint="stm32", require_deep=True)
    assert decision.model_id == "qwen480b"
    assert decision.adapter is None
