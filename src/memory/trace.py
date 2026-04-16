"""Trace neuro-symbolic graph (arxiv 2601.15311).

Typed directed multi-edge graph: nodes are episodes (raw or summary),
edges are typed (temporal, causal, topical, summary_of). Pure Python
implementation — networkx optional via ``to_networkx()``.

Backward-compat: ``Episode`` and ``CausalityEdge`` aliases preserved
for v0.2 callers (aeon.py, aeon_hook.py, tests).
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


# ── v0.2 compat dataclasses ───────────────────────────────────────────

@dataclass(frozen=True)
class Episode:
    """v0.2 compat: episode with content, domain, timestamp."""
    id: str
    content: str
    domain: str
    timestamp: datetime
    source: str = ""
    metadata: dict = field(default_factory=dict)

    def to_node(self) -> Node:
        return Node(
            id=self.id, kind="raw", ts=self.timestamp,
            attrs={"content": self.content, "domain": self.domain,
                   "source": self.source, **self.metadata},
        )


@dataclass(frozen=True)
class CausalityEdge:
    """v0.2 compat: causal edge with weight."""
    from_id: str
    to_id: str
    weight: float
    relation: str = "causes"


# ── TraceGraph ─────────────────────────────────────────────────────────

class TraceGraph:
    """Directed typed multi-graph keyed by node id."""

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._out: dict[str, list[Edge]] = defaultdict(list)
        self._in: dict[str, list[Edge]] = defaultdict(list)
        # v0.2 compat: episode store
        self._episodes: dict[str, Episode] = {}
        self._compat_edges: list[CausalityEdge] = []

    # ── v0.3 API ──────────────────────────────────────────────────────

    def add_node(self, node_id: str, *, kind: str = "raw",
                 ts: datetime | None = None, **attrs: Any) -> Node:
        if kind not in NODE_KINDS:
            raise ValueError(f"unknown node kind: {kind}")
        node = Node(id=node_id, kind=kind, ts=ts, attrs=dict(attrs))
        self._nodes[node_id] = node
        self._out.setdefault(node_id, [])
        self._in.setdefault(node_id, [])
        return node

    def add_typed_edge(self, src: str, dst: str, kind: str,
                       **attrs: Any) -> Edge:
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

    def successors(self, node_id: str, *, kind: str | None = None) -> list[str]:
        return [e.dst for e in self._out.get(node_id, [])
                if kind is None or e.kind == kind]

    def predecessors(self, node_id: str, *, kind: str | None = None) -> list[str]:
        return [e.src for e in self._in.get(node_id, [])
                if kind is None or e.kind == kind]

    def ancestors(self, node_id: str, *, kind: str | None = None) -> set[str]:
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

    def descendants(self, node_id: str, *, kind: str | None = None) -> set[str]:
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

    def time_range(self, start: datetime, end: datetime) -> list[Node]:
        return sorted(
            [n for n in self._nodes.values()
             if n.ts is not None and start <= n.ts <= end],
            key=lambda n: n.ts,  # type: ignore[arg-type]
        )

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self._nodes:
            return False
        for edge in list(self._out.get(node_id, [])):
            self._in.get(edge.dst, [])[:] = [
                e for e in self._in.get(edge.dst, []) if e.src != node_id]
        for edge in list(self._in.get(node_id, [])):
            self._out.get(edge.src, [])[:] = [
                e for e in self._out.get(edge.src, []) if e.dst != node_id]
        self._out.pop(node_id, None)
        self._in.pop(node_id, None)
        del self._nodes[node_id]
        self._episodes.pop(node_id, None)
        return True

    def stats(self) -> dict[str, Any]:
        edge_counts: dict[str, int] = {k: 0 for k in EDGE_KINDS}
        for edge in self.edges():
            edge_counts[edge.kind] += 1
        return {"n_nodes": len(self._nodes),
                "n_edges": sum(edge_counts.values()), "by_kind": edge_counts}

    # ── v0.2 compat API ──────────────────────────────────────────────

    def add_episode(self, episode: Episode) -> None:
        """v0.2 compat: add an Episode as a Node."""
        self._episodes[episode.id] = episode
        self.add_node(episode.id, kind="raw", ts=episode.timestamp,
                      content=episode.content, domain=episode.domain,
                      source=episode.source, **episode.metadata)

    def add_edge(self, edge: CausalityEdge) -> None:  # type: ignore[override]
        """v0.2 compat: add a CausalityEdge."""
        self._compat_edges.append(edge)
        if not self.has_node(edge.from_id):
            self.add_node(edge.from_id)
        if not self.has_node(edge.to_id):
            self.add_node(edge.to_id)
        self.add_typed_edge(edge.from_id, edge.to_id, "causal",
                            weight=edge.weight, relation=edge.relation)

    def get_episode(self, episode_id: str) -> Episode | None:
        return self._episodes.get(episode_id)

    def walk(self, from_id: str, max_depth: int = 3) -> list[Episode]:
        visited: set[str] = set()
        result: list[Episode] = []
        def _dfs(nid: str, depth: int) -> None:
            if depth > max_depth or nid in visited:
                return
            visited.add(nid)
            ep = self._episodes.get(nid)
            if ep:
                result.append(ep)
            for child in self.successors(nid):
                _dfs(child, depth + 1)
        _dfs(from_id, 0)
        return result

    def query_by_time(self, start: datetime, end: datetime) -> list[Episode]:
        return [ep for ep in self._episodes.values()
                if start <= ep.timestamp <= end]

    def query_by_rule(self, domain: str | None = None,
                      min_causality: float = 0.0) -> list[Episode]:
        """v0.2 compat: filter by domain and/or causality weight."""
        matching_ids: set[str] = set()
        if domain:
            for ep in self._episodes.values():
                if ep.domain == domain:
                    matching_ids.add(ep.id)
        else:
            matching_ids = set(self._episodes.keys())
        if min_causality > 0:
            causal_ids = set()
            for edge in self._compat_edges:
                if edge.weight >= min_causality and edge.to_id in matching_ids:
                    causal_ids.add(edge.to_id)
            matching_ids &= causal_ids
        return [self._episodes[eid] for eid in matching_ids
                if eid in self._episodes]

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def num_edges(self) -> int:
        return sum(len(b) for b in self._out.values())
