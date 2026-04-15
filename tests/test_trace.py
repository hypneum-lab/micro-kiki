"""Tests for the Trace neuro-symbolic graph (story-7)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.memory.trace import EDGE_KINDS, TraceGraph


def _ts(offset_minutes: int) -> datetime:
    base = datetime(2026, 4, 16, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def test_add_node_and_edge_roundtrip() -> None:
    g = TraceGraph()
    g.add_node("a", ts=_ts(0), topic="init")
    g.add_node("b", ts=_ts(1), topic="init")
    g.add_edge("a", "b", kind="temporal")
    assert g.has_node("a") and g.has_node("b")
    assert g.successors("a") == ["b"]
    assert g.predecessors("b") == ["a"]
    st = g.stats()
    assert st["n_nodes"] == 2
    assert st["n_edges"] == 1
    assert st["by_kind"]["temporal"] == 1


def test_unknown_kind_raises() -> None:
    g = TraceGraph()
    with pytest.raises(ValueError):
        g.add_node("a", kind="alien")
    g.add_node("a")
    g.add_node("b")
    with pytest.raises(ValueError):
        g.add_edge("a", "b", kind="alien")


def test_missing_endpoints_raise() -> None:
    g = TraceGraph()
    g.add_node("a")
    with pytest.raises(KeyError):
        g.add_edge("a", "ghost", kind="causal")
    with pytest.raises(KeyError):
        g.add_edge("ghost", "a", kind="causal")


def test_ancestors_chain() -> None:
    g = TraceGraph()
    for i in range(5):
        g.add_node(f"n{i}", ts=_ts(i))
    for i in range(4):
        g.add_edge(f"n{i}", f"n{i+1}", kind="causal")
    anc = g.ancestors("n3", kind="causal")
    assert anc == {"n0", "n1", "n2"}
    desc = g.descendants("n1", kind="causal")
    assert desc == {"n2", "n3", "n4"}


def test_edge_kind_filter() -> None:
    g = TraceGraph()
    g.add_node("a")
    g.add_node("b")
    g.add_edge("a", "b", kind="temporal")
    g.add_edge("a", "b", kind="topical")
    assert len(g.successors("a", kind="temporal")) == 1
    assert len(g.successors("a", kind="topical")) == 1
    assert len(g.successors("a")) == 2


def test_time_range_sorted() -> None:
    g = TraceGraph()
    g.add_node("a", ts=_ts(2))
    g.add_node("b", ts=_ts(0))
    g.add_node("c", ts=_ts(5))
    hits = g.time_range(_ts(-1), _ts(3))
    assert [n.id for n in hits] == ["b", "a"]


def test_remove_node_is_clean() -> None:
    g = TraceGraph()
    for name in "abc":
        g.add_node(name)
    g.add_edge("a", "b", kind="causal")
    g.add_edge("b", "c", kind="causal")
    assert g.remove_node("b") is True
    assert g.invariants_ok()
    assert g.successors("a") == []
    assert g.predecessors("c") == []
    assert g.remove_node("b") is False  # idempotent


def test_summary_kind_supported() -> None:
    g = TraceGraph()
    g.add_node("raw1", kind="raw")
    g.add_node("raw2", kind="raw")
    g.add_node("sum", kind="summary")
    g.add_edge("sum", "raw1", kind="summary_of")
    g.add_edge("sum", "raw2", kind="summary_of")
    assert set(
        n.id for n in g.nodes(kind="summary")
    ) == {"sum"}
    assert set(g.successors("sum", kind="summary_of")) == {
        "raw1",
        "raw2",
    }


def test_edge_kinds_constant_covers_spec() -> None:
    assert set(EDGE_KINDS) == {
        "temporal",
        "causal",
        "topical",
        "summary_of",
    }
