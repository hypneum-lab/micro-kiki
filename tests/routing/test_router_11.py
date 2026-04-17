from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from src.routing.router import MetaRouter, NICHE_DOMAINS


def test_router_has_35_outputs_by_default():
    router = MetaRouter()
    assert router.num_domains == 35


def test_niche_domains_has_34_elements():
    assert len(NICHE_DOMAINS) == 34


def test_new_domains_in_niche_domains():
    assert "components" in NICHE_DOMAINS
    assert "llm-ops" in NICHE_DOMAINS
    assert "ml-training" in NICHE_DOMAINS


def test_chat_fr_in_niche_domains():
    assert "chat-fr" in NICHE_DOMAINS


def test_kicad_dsl_in_niche_domains():
    assert "kicad-dsl" in NICHE_DOMAINS


def test_output_shape_35_domains():
    router = MetaRouter(num_domains=35)
    assert router.num_domains == 35
    x = torch.zeros(1, 768)
    out = router(x)
    assert out.shape == (1, 35 + 5)


def test_base_fallback_activates_for_general_query():
    """All niche outputs below threshold -> fallback to ["base"]."""
    router = MetaRouter()
    # Craft output tensor directly with all domain scores below threshold.
    crafted = torch.full((1, 35 + 5), 0.05)  # all below 0.12
    active = router.get_active_domains_named(crafted, threshold=0.12)
    assert active == ["base"]


def test_niche_domain_activates_above_threshold():
    """Force a known niche domain output above threshold -> it appears in result."""
    router = MetaRouter()
    # Manually craft output tensor: all low except first niche index
    crafted = torch.full((1, 35), 0.05)
    crafted[0, 0] = 0.9  # index 0 = first sorted niche domain
    active = router.get_active_domains_named(crafted, threshold=0.12)
    assert active != ["base"]
    assert len(active) >= 1
