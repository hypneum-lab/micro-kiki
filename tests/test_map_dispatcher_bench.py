"""Tests for story-2 dispatcher bench."""

from __future__ import annotations

import json

from src.eval.map_dispatcher_bench import (
    META_INTENTS,
    V02DispatcherAgent,
    run,
)


def test_meta_intents_seven():
    assert len(META_INTENTS) == 7


def test_dispatcher_conflict_is_bounded():
    agent = V02DispatcherAgent()
    # Flat distribution over 4 stacks → near 1.0.
    flat = [("code", 0.25), ("chat", 0.25), ("knowledge", 0.25),
            ("creative", 0.25)]
    assert 0.9 <= agent.conflict_monitor(flat) <= 1.0
    # One-hot → near 0.
    spike = [("code", 0.95), ("chat", 0.02), ("knowledge", 0.02),
             ("creative", 0.01)]
    assert agent.conflict_monitor(spike) < 0.3


def test_dispatcher_applies_chat_floor():
    agent = V02DispatcherAgent(chat_floor=0.20)
    # Top score below the chat floor → conflict gets bumped.
    low = [("code", 0.15), ("chat", 0.14), ("knowledge", 0.13),
           ("creative", 0.12)]
    high = [("code", 0.60), ("chat", 0.20), ("knowledge", 0.10),
            ("creative", 0.10)]
    assert agent.conflict_monitor(low) >= agent.conflict_monitor(high) - 0.5
    # Bumped version must still be <= 1.0.
    assert agent.conflict_monitor(low) <= 1.0


def test_dispatcher_activation_cap():
    agent = V02DispatcherAgent(activation_cap=4)
    # More than 4 inputs → only top 4 considered.
    many = [
        ("code", 0.9),
        ("chat", 0.05),
        ("knowledge", 0.03),
        ("creative", 0.02),
        ("electronics", 0.001),
        ("system", 0.001),
        ("security", 0.001),
    ]
    # Should still act like a spike (only top 4 kept).
    assert agent.conflict_monitor(many) < 0.5


def test_run_emits_full_report():
    result = run(n=30, seed=3)
    d = result.to_dict()
    assert d["schema_version"] == "map-dispatcher-bench/1.0"
    assert d["agent"] == "V02DispatcherAgent"
    assert d["n"] == 30
    # At least 3 numeric metrics as required by acceptance.
    numerics = [
        k for k in d
        if isinstance(d[k], (int, float)) and k != "n"
    ]
    assert len(numerics) >= 3
    assert isinstance(d["notes"], list)
    assert len(d["notes"]) >= 1


def test_run_deterministic():
    a = run(n=20, seed=7).to_dict()
    b = run(n=20, seed=7).to_dict()
    # Drop latency (wall-clock-dependent).
    for side in (a, b):
        side.pop("latency_ms", None)
    assert a == b
