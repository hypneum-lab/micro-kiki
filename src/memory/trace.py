"""Trace neuro-symbolic graph.

Ported from the v0.2 cognitive-layer design spec. Stores episodes as
typed nodes with typed edges (``temporal``, ``causal``, ``topical``,
``summary_of``). Implemented as a lightweight directed multi-edge
graph in pure Python to avoid an external dependency; the surface is
API-compatible with what networkx would offer for the subset we need
(add_node, add_edge, ancestors, neighbors, predecessors).

If networkx is available we expose ``to_networkx`` for interop.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Iterator

EDGE_KINDS = ("temporal", "causal", "topical", "summary_of")
NODE_KINDS = ("raw", "summary")


@dataclass
class Node:
    """Trace node (episode or summary)."""

    id: str
    kind: str = "raw"
    ts: datetime | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """Directed typed edge."""

    src: str
    dst: str
    kind: str
    attrs: dict[str, Any] = field(default_factory=dict)


class TraceGraph:
    """Directed typed multi-graph keyed by node id."""

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._out: dict[str, list[Edge]] = defaultdict(list)
        self._in: dict[str, list[Edge]] = defaultdict(list)

    # ------------------------------------------------------------------ add

    def add_node(
        self,
        node_id: str,
        *,
        kind: str = "raw",
        ts: datetime | None = None,
        **attrs: Any,
    ) -> Node:
        if kind not in NODE_KINDS:
            raise ValueError(f"unknown node kind: {kind}")
        node = Node(id=node_id, kind=kind, ts=ts, attrs=dict(attrs))
        self._nodes[node_id] = node
        # Initialise buckets so has_node + empty-queries work cleanly.
        self._out.setdefault(node_id, [])
        self._in.setdefault(node_id, [])
        return node

    def add_edge(
        self,
        src: str,
        dst: str,
        kind: str,
        **attrs: Any,
    ) -> Edge:
        if kind not in EDGE_KINDS:
            raise ValueError(f"unknown edge kind: {kind}")
        if src not in self._nodes:
            raise KeyError(f"missing src node: {src}")
        if dst not in self._nodes:
            raise KeyError(f"missing dst node: {dst}")
        edge = Edge(src=src, dst=dst, kind=kind, attrs=dict(attrs))
        self._out[src].append(edge)
        self._in[dst].append(edge)
        return edge

    # ---------------------------------------------------------------- query

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_node(self, node_id: str) -> Node:
        return self._nodes[node_id]

    def nodes(self, *, kind: str | None = None) -> Iterator[Node]:
        if kind is None:
            return iter(self._nodes.values())
        return (n for n in self._nodes.values() if n.kind == kind)

    def edges(self, *, kind: str | None = None) -> Iterator[Edge]:
        for bucket in self._out.values():
            for edge in bucket:
                if kind is None or edge.kind == kind:
                    yield edge

    def successors(
        self, node_id: str, *, kind: str | None = None
    ) -> list[str]:
        return [
            e.dst for e in self._out.get(node_id, [])
            if kind is None or e.kind == kind
        ]

    def predecessors(
        self, node_id: str, *, kind: str | None = None
    ) -> list[str]:
        return [
            e.src for e in self._in.get(node_id, [])
            if kind is None or e.kind == kind
        ]

    def ancestors(
        self, node_id: str, *, kind: str | None = None
    ) -> set[str]:
        """Transitive predecessors along edges (optionally typed)."""
        if node_id not in self._nodes:
            raise KeyError(node_id)
        seen: set[str] = set()
        frontier: deque[str] = deque([node_id])
        while frontier:
            current = frontier.popleft()
            for parent in self.predecessors(current, kind=kind):
                if parent not in seen:
                    seen.add(parent)
                    frontier.append(parent)
        return seen

    def descendants(
        self, node_id: str, *, kind: str | None = None
    ) -> set[str]:
        if node_id not in self._nodes:
            raise KeyError(node_id)
        seen: set[str] = set()
        frontier: deque[str] = deque([node_id])
        while frontier:
            current = frontier.popleft()
            for child in self.successors(current, kind=kind):
                if child not in seen:
                    seen.add(child)
                    frontier.append(child)
        return seen

    def time_range(
        self, start: datetime, end: datetime
    ) -> list[Node]:
        if start > end:
            raise ValueError("start must be <= end")
        hits = [
            n for n in self._nodes.values()
            if n.ts is not None and start <= n.ts <= end
        ]
        hits.sort(key=lambda n: n.ts)  # type: ignore[arg-type]
        return hits

    # -------------------------------------------------------------- mutate

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self._nodes:
            return False
        # Drop adjacent edges on both sides.
        for edge in list(self._out.get(node_id, [])):
            bucket = self._in.get(edge.dst, [])
            bucket[:] = [e for e in bucket if e.src != node_id]
        for edge in list(self._in.get(node_id, [])):
            bucket = self._out.get(edge.src, [])
            bucket[:] = [e for e in bucket if e.dst != node_id]
        self._out.pop(node_id, None)
        self._in.pop(node_id, None)
        del self._nodes[node_id]
        return True

    # ------------------------------------------------------------ invariants

    def invariants_ok(self) -> bool:
        """Cheap sanity check used in tests and :meth:`stats`."""
        for node_id in self._nodes:
            for edge in self._out.get(node_id, []):
                if edge.dst not in self._nodes:
                    return False
                if edge not in self._in.get(edge.dst, []):
                    return False
            for edge in self._in.get(node_id, []):
                if edge.src not in self._nodes:
                    return False
        return True

    def stats(self) -> dict[str, Any]:
        edge_counts: dict[str, int] = {k: 0 for k in EDGE_KINDS}
        for edge in self.edges():
            edge_counts[edge.kind] += 1
        return {
            "n_nodes": len(self._nodes),
            "n_edges": sum(edge_counts.values()),
            "by_kind": edge_counts,
        }

    # ------------------------------------------------------------- interop

    def to_networkx(self):  # pragma: no cover - optional
        """Return a ``networkx.MultiDiGraph`` for interop when available."""
        try:
            import networkx as nx  # type: ignore
        except ImportError as exc:
            raise RuntimeError("networkx not installed") from exc
        g = nx.MultiDiGraph()
        for node in self._nodes.values():
            g.add_node(node.id, kind=node.kind, ts=node.ts, **node.attrs)
        for edge in self.edges():
            g.add_edge(
                edge.src, edge.dst, key=edge.kind, kind=edge.kind,
                **edge.attrs,
            )
        return g
