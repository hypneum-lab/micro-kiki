"""Unit tests for the MAP harness (story-1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.map_harness import (
    MAPHarness,
    MAPModule,
    MockAgent,
    SCHEMA_VERSION,
    edit_distance,
    gen_conflict_prompts,
    gen_judge_prompts,
    gen_meta_prompts,
    gen_plan_prompts,
    gen_trajectory_prompts,
    spearman,
    token_cosine,
)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def test_gen_conflict_prompts_shape():
    items = gen_conflict_prompts(n=12, seed=1)
    assert len(items) == 12
    for it in items:
        assert it.module is MAPModule.CONFLICT_MONITOR
        assert "intents" in it.payload
        assert isinstance(it.reference, float)
        assert 0.0 <= it.reference <= 1.0


def test_gen_trajectory_prompts_shape():
    items = gen_trajectory_prompts(n=10, seed=2)
    assert len(items) == 10
    for it in items:
        assert it.module is MAPModule.STATE_PREDICTOR
        assert "history" in it.payload
        assert isinstance(it.reference, str)


def test_gen_judge_prompts_balanced():
    items = gen_judge_prompts(n=50, seed=3)
    assert len(items) == 50
    winners = [it.reference for it in items]
    assert set(winners) == {"a", "b"}


def test_gen_plan_prompts_shape():
    items = gen_plan_prompts(n=8, seed=4)
    for it in items:
        assert it.module is MAPModule.DECOMPOSER
        assert isinstance(it.reference, list)
        assert all(isinstance(s, str) for s in it.reference)


def test_gen_meta_prompts_labels():
    items = gen_meta_prompts(n=40, seed=5)
    labels = {it.reference for it in items}
    assert labels.issubset({"act", "plan", "reflect"})


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def test_spearman_rank_identity():
    xs = [0.1, 0.2, 0.3, 0.4]
    ys = [1.0, 2.0, 3.0, 4.0]
    assert spearman(xs, ys) == pytest.approx(1.0)


def test_spearman_reverse():
    xs = [0.1, 0.2, 0.3, 0.4]
    ys = [4.0, 3.0, 2.0, 1.0]
    assert spearman(xs, ys) == pytest.approx(-1.0)


def test_edit_distance_basic():
    assert edit_distance(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert edit_distance(["a", "b"], ["a", "c"]) == 1
    assert edit_distance([], ["x", "y"]) == 2


def test_token_cosine_bounds():
    assert 0.0 <= token_cosine("hello world", "hello there") <= 1.0
    assert token_cosine("abc", "") == 0.0
    assert token_cosine("hello world", "hello world") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Harness end-to-end on MockAgent
# ---------------------------------------------------------------------------


def test_harness_runs_all_modules():
    harness = MAPHarness(seed=7)
    report = harness.run_all(MockAgent())

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["agent"] == "MockAgent"
    assert report["seed"] == 7

    modules = report["modules"]
    expected = {m.value for m in MAPModule}
    assert set(modules.keys()) == expected

    # Every module produces at least one numeric metric + n>0.
    for name, sub in modules.items():
        assert sub["n"] > 0, name
        numeric = [
            v for v in sub.values() if isinstance(v, (int, float))
        ]
        assert len(numeric) >= 2, name


def test_harness_writes_valid_json(tmp_path: Path):
    harness = MAPHarness(seed=1)
    report = harness.run_all(MockAgent())
    out = tmp_path / "sub" / "report.json"
    written = harness.write_json(report, out)
    assert written.exists()
    loaded = json.loads(written.read_text())
    assert loaded["schema_version"] == SCHEMA_VERSION
    assert "modules" in loaded


def test_mock_agent_reasonable_conflict_score():
    # Entropy-based mock should be near 1.0 for a flat distribution.
    agent = MockAgent()
    flat = [("a", 0.25), ("b", 0.25), ("c", 0.25), ("d", 0.25)]
    assert agent.conflict_monitor(flat) == pytest.approx(1.0, abs=1e-6)
    # And near 0 for a one-hot distribution.
    spike = [("a", 0.97), ("b", 0.01), ("c", 0.01), ("d", 0.01)]
    assert agent.conflict_monitor(spike) < 0.3


def test_mock_agent_coordinator_thresholds():
    agent = MockAgent()
    assert agent.coordinate({"uncertainty": 0.1}) == "act"
    assert agent.coordinate({"uncertainty": 0.5}) == "plan"
    assert agent.coordinate({"uncertainty": 0.9}) == "reflect"


def test_harness_deterministic():
    h1 = MAPHarness(seed=11).run_all(MockAgent())
    h2 = MAPHarness(seed=11).run_all(MockAgent())
    # Drop latency fields which depend on wall clock.
    for side in (h1, h2):
        for sub in side["modules"].values():
            sub.pop("latency_ms", None)
    assert h1["modules"] == h2["modules"]
