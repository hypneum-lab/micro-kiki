"""Atlas SIMD-friendly vector index.

Ported from the v0.2 cognitive-layer design spec. Pure-Python with
optional numpy acceleration. Stores unit-normalised vectors in a
matrix and runs cosine-similarity search via ``np.einsum`` when numpy
is available, falling back to a pure-Python dot-product loop.

Marking a TODO for a proper AVX-512 / NEON C/Rust kernel later;
today's needs are covered by numpy BLAS under the hood.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

try:  # optional numpy acceleration
    import numpy as np  # type: ignore

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - exercised on minimal envs
    np = None  # type: ignore
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class AtlasEntry:
    """Single entry stored in the index."""

    key: str
    vector: tuple[float, ...]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    """Result of a top-k search."""

    key: str
    score: float
    payload: dict[str, Any]


# ---------------------------------------------------------------------------
# Math helpers (stdlib fallback)
# ---------------------------------------------------------------------------


def _normalise_py(vec: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return tuple(vec)
    inv = 1.0 / norm
    return tuple(v * inv for v in vec)


def _cosine_py(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Atlas index
# ---------------------------------------------------------------------------


class AtlasIndex:
    """In-memory cosine-similarity top-k index.

    Parameters
    ----------
    dim : int
        Dimensionality of vectors. All inserts are validated.
    use_numpy : bool | None
        If ``None`` (default), use numpy when available. If explicitly
        ``False``, force the pure-Python path (useful for tests).
    """

    def __init__(self, dim: int, *, use_numpy: bool | None = None) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self._use_numpy = _HAS_NUMPY if use_numpy is None else (
            use_numpy and _HAS_NUMPY
        )
        self._entries: list[AtlasEntry] = []
        self._key_to_idx: dict[str, int] = {}
        self._matrix = None  # lazy; rebuilt on demand when numpy is used
        self._matrix_stale = True

    # ------------------------------------------------------------------ API

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._entries)

    def add(self, key: str, vector: Sequence[float], payload: dict[str, Any] | None = None) -> None:
        """Insert or update an entry."""
        if len(vector) != self.dim:
            raise ValueError(
                f"vector dim {len(vector)} != index dim {self.dim}"
            )
        unit = self._normalise(vector)
        entry = AtlasEntry(key=key, vector=tuple(unit), payload=payload or {})
        if key in self._key_to_idx:
            idx = self._key_to_idx[key]
            self._entries[idx] = entry
        else:
            self._key_to_idx[key] = len(self._entries)
            self._entries.append(entry)
        self._matrix_stale = True

    def remove(self, key: str) -> bool:
        """Remove an entry by key. Returns True if removed."""
        if key not in self._key_to_idx:
            return False
        idx = self._key_to_idx.pop(key)
        last = len(self._entries) - 1
        if idx != last:
            self._entries[idx] = self._entries[last]
            self._key_to_idx[self._entries[idx].key] = idx
        self._entries.pop()
        self._matrix_stale = True
        return True

    def search(self, query: Sequence[float], k: int = 10) -> list[SearchHit]:
        """Return top-``k`` entries by cosine similarity."""
        if len(query) != self.dim:
            raise ValueError(
                f"query dim {len(query)} != index dim {self.dim}"
            )
        if k <= 0 or not self._entries:
            return []
        q = self._normalise(query)
        if self._use_numpy:
            scores = self._scores_numpy(q)
        else:
            scores = [_cosine_py(q, e.vector) for e in self._entries]
        # argsort descending, take top-k
        idx_score = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [
            SearchHit(
                key=self._entries[i].key,
                score=float(scores[i]),
                payload=dict(self._entries[i].payload),
            )
            for i in idx_score
        ]

    def stats(self) -> dict[str, Any]:
        """Minimal introspection for :class:`AeonSleep`."""
        return {
            "n": len(self._entries),
            "dim": self.dim,
            "numpy": self._use_numpy,
        }

    # ------------------------------------------------------------------ impl

    def _normalise(self, vector: Sequence[float]) -> tuple[float, ...]:
        if self._use_numpy:
            arr = np.asarray(vector, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm == 0.0:
                return tuple(float(x) for x in arr.tolist())
            return tuple((arr / norm).tolist())
        return _normalise_py(vector)

    def _scores_numpy(self, q: tuple[float, ...]):
        assert _HAS_NUMPY
        if self._matrix_stale or self._matrix is None:
            self._matrix = np.asarray(
                [e.vector for e in self._entries], dtype=np.float32
            )
            self._matrix_stale = False
        q_arr = np.asarray(q, dtype=np.float32)
        # cosine = dot(q, mat.T) since both are unit-normalised
        return np.einsum("ij,j->i", self._matrix, q_arr).tolist()


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def time_search(
    index: AtlasIndex, query: Sequence[float], k: int, repeats: int = 3
) -> float:
    """Return mean wall-clock ms for ``repeats`` searches."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    total = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        index.search(query, k=k)
        total += time.perf_counter() - start
    return (total / repeats) * 1000.0
