"""Aeon Memory Palace — unified API wrapping Atlas + Trace.

High-level interface: write, recall, walk, query_by_time, compress.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime

import numpy as np

from src.memory.atlas import AtlasIndex
from src.memory.trace import TraceGraph, Episode, CausalityEdge

logger = logging.getLogger(__name__)


def _load_st_model(model_path: str) -> tuple:
    """Load a sentence-transformers model and return (embed_fn, dim)."""
    from pathlib import Path
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Embedding model not found at {model_path}. "
            "Train one with: python3 scripts/train_embeddings.py --all"
        )
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for model_path loading. "
            "Install with: pip install 'micro-kiki[embeddings]'"
        )
    model = SentenceTransformer(str(p))
    dim = model.get_sentence_embedding_dimension()
    logger.info("Loaded embedding model from %s (dim=%d)", p, dim)
    def embed_fn(text: str) -> np.ndarray:
        return model.encode(text, normalize_embeddings=True).astype(np.float32)
    return embed_fn, dim


class AeonPalace:
    """Unified memory palace combining vector search + episodic graph."""

    def __init__(
        self,
        dim: int | None = None,
        embed_fn=None,
        model_path: str | None = None,
    ) -> None:
        if embed_fn is not None:
            self._embed_fn = embed_fn
            self._dim = dim or 384
        elif model_path is not None:
            self._embed_fn, self._dim = _load_st_model(model_path)
        else:
            raise ImportError(
                "AeonPalace requires an embed_fn or model_path. "
                "Train one with: python3 scripts/train_embeddings.py --all"
            )
        self._atlas = AtlasIndex(dim=self._dim)
        self._trace = TraceGraph()

    def write(
        self,
        content: str,
        domain: str,
        timestamp: datetime | None = None,
        links: list[str] | None = None,
        source: str = "",
        metadata: dict | None = None,
    ) -> str:
        ts = timestamp or datetime.now()
        episode_id = hashlib.sha256(f"{content}:{ts.isoformat()}".encode()).hexdigest()[:16]

        episode = Episode(
            id=episode_id, content=content, domain=domain,
            timestamp=ts, source=source, metadata=metadata or {},
        )
        self._trace.add_episode(episode)

        vec = self._embed_fn(content)
        self._atlas.insert(episode_id, vec)

        for link_id in (links or []):
            self._trace.add_edge(CausalityEdge(from_id=link_id, to_id=episode_id, weight=1.0))

        return episode_id

    def recall(self, query: str, top_k: int = 10, domain: str | None = None) -> list[Episode]:
        vec = self._embed_fn(query)
        candidates = self._atlas.recall(vec, top_k=top_k * 2)

        results = []
        for cid, score in candidates:
            ep = self._trace.get_episode(cid)
            if ep and (domain is None or ep.domain == domain):
                results.append(ep)
                if len(results) >= top_k:
                    break
        return results

    def walk(self, from_id: str, max_depth: int = 3) -> list[Episode]:
        return self._trace.walk(from_id, max_depth)

    def query_by_time(self, start: datetime, end: datetime) -> list[Episode]:
        return self._trace.query_by_time(start, end)

    def compress(self, older_than: datetime, summarize_fn=None) -> int:
        """Compress old episodes by summarizing content."""
        old_episodes = [
            ep for ep in self._trace._episodes.values()
            if ep.timestamp < older_than
        ]
        compressed = 0
        for ep in old_episodes:
            if summarize_fn:
                summary = summarize_fn(ep.content)
            else:
                summary = ep.content[:100] + "..." if len(ep.content) > 100 else ep.content

            new_ep = Episode(
                id=ep.id, content=summary, domain=ep.domain,
                timestamp=ep.timestamp, source=ep.source,
                metadata={**ep.metadata, "compressed": True},
            )
            self._trace._episodes[ep.id] = new_ep
            compressed += 1

        logger.info("Compressed %d episodes older than %s", compressed, older_than)
        return compressed

    @property
    def stats(self) -> dict:
        return {
            "vectors": self._atlas.total_vectors,
            "episodes": self._trace.num_episodes,
            "edges": self._trace.num_edges,
        }
