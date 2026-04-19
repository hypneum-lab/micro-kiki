"""Integration test for src/routing/downstream_harness.py with mocked LLM + judge."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np


def test_run_downstream_routes_llm_then_judges():
    from src.routing.downstream_harness import run_downstream_eval

    queries = [
        {"question": "What is a Schmitt trigger?", "domain": "electronics", "domain_idx": 0},
        {"question": "How do I debounce a switch?", "domain": "electronics", "domain_idx": 0},
        {"question": "What is a dsp filter?", "domain": "dsp", "domain_idx": 1},
    ]
    n_classes = 2

    router_fn = MagicMock(side_effect=lambda emb: 0)
    llm_fn = MagicMock(return_value="mock answer")

    def judge_fn(question: str, answer: str, domain: str) -> int:
        return 5 if domain == "electronics" else 1

    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((len(queries), 16))

    result = run_downstream_eval(
        queries=queries,
        embeddings=embeddings,
        router_fn=router_fn,
        llm_fn=llm_fn,
        judge_fn=judge_fn,
        domain_names=["electronics", "dsp"],
    )

    assert "per_query" in result
    assert "mean_score" in result
    assert len(result["per_query"]) == 3
    # 2/3 queries are electronics (score 5), 1/3 is dsp (score 1)
    assert abs(result["mean_score"] - 11 / 3) < 1e-6
    assert router_fn.call_count == 3
    assert llm_fn.call_count == 3


def test_run_downstream_reports_routing_accuracy():
    """Harness should surface how often router picked the correct domain."""
    from src.routing.downstream_harness import run_downstream_eval

    queries = [
        {"question": "q1", "domain": "a", "domain_idx": 0},
        {"question": "q2", "domain": "b", "domain_idx": 1},
        {"question": "q3", "domain": "b", "domain_idx": 1},
    ]
    router_fn = MagicMock(side_effect=[0, 0, 1])  # correct, wrong, correct
    llm_fn = MagicMock(return_value="x")
    judge_fn = MagicMock(return_value=3)

    result = run_downstream_eval(
        queries=queries,
        embeddings=np.zeros((3, 4)),
        router_fn=router_fn,
        llm_fn=llm_fn,
        judge_fn=judge_fn,
        domain_names=["a", "b"],
    )
    assert result["routing_accuracy"] == 2 / 3
