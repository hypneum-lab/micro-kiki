"""SleepGate conflict-aware temporal tagger.

Implements the in-process portion of the conflict tagger from arxiv
2603.14517 ("Memory under sleep: conflict-tagged consolidation in
language agents"). The scorer combines:

* Cheap embedding cosine similarity against recent episodes.
* Lexical rule-based contradiction detection (negation flips, numeric
  supersession on tagged keys).
* Staleness heuristic when an episode references a key/topic that has
  not been touched for N epochs.

No LLM call is issued here — the downstream consolidation module is
free to call a teacher LLM but this tagger must be fast enough to sit
in the hot path of :meth:`AeonSleep.write`.

Reason codes:

* ``topic``         — high cosine but no explicit contradiction (drift).
* ``contradiction`` — explicit antonym/negation or numeric supersession.
* ``stale``         — long gap, same topic key, old episode deprecated.
* ``none``          — no recent episode matched.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Sequence

REASON_TOPIC = "topic"
REASON_CONTRADICTION = "contradiction"
REASON_STALE = "stale"
REASON_NONE = "none"

_NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "without",
    "n't",
    "non",
    "pas",
    "jamais",
}
_ANTONYM_PAIRS = [
    ("on", "off"),
    ("true", "false"),
    ("open", "closed"),
    ("enabled", "disabled"),
    ("up", "down"),
    ("yes", "no"),
    ("alive", "dead"),
    ("hot", "cold"),
    ("start", "stop"),
]

_NUMERIC_RE = re.compile(r"([a-zA-Z_][\w\-]*)\s*[:=]\s*(-?\d+(?:\.\d+)?)")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RecentEpisode:
    """Shape consumed by :class:`SleepTagger` — decoupled from AeonSleep."""

    id: str
    text: str
    embedding: Sequence[float]
    ts: datetime
    topic: str | None = None


@dataclass
class Tag:
    """Output of the tagger."""

    level: float
    reason: str
    ref_id: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _tokens(text: str) -> set[str]:
    return {tok.lower() for tok in re.findall(r"[\w']+", text)}


def _has_negation_flip(a: str, b: str) -> bool:
    """Return True if one text negates a key token also present in the other."""
    ta, tb = _tokens(a), _tokens(b)
    only_a = ta - tb
    only_b = tb - ta
    if _NEGATION_WORDS & only_a or _NEGATION_WORDS & only_b:
        # one adds a negator → contradiction if the shared content
        # still overlaps meaningfully
        shared = ta & tb
        shared -= _NEGATION_WORDS
        if len(shared) >= 2:
            return True
    for left, right in _ANTONYM_PAIRS:
        if (left in ta and right in tb) or (right in ta and left in tb):
            return True
    return False


def _numeric_conflict(a: str, b: str) -> bool:
    """Return True if both texts set the same key to different numbers."""
    map_a = dict(_NUMERIC_RE.findall(a))
    map_b = dict(_NUMERIC_RE.findall(b))
    for key, va in map_a.items():
        if key in map_b and map_b[key] != va:
            return True
    return False


# ---------------------------------------------------------------------------
# Tagger
# ---------------------------------------------------------------------------


class SleepTagger:
    """Conflict-aware temporal tagger.

    Parameters
    ----------
    topic_threshold : float
        Cosine similarity above which two episodes are considered to
        share a topic (default 0.6).
    contradiction_boost : float
        Extra level added when a contradiction is detected on top of a
        topical hit (default 0.4).
    stale_delta : timedelta
        Age past which a topical match is tagged ``stale`` instead of
        ``topic`` (default 7 days).
    """

    def __init__(
        self,
        *,
        topic_threshold: float = 0.6,
        contradiction_boost: float = 0.4,
        stale_delta: timedelta = timedelta(days=7),
    ) -> None:
        if not 0.0 <= topic_threshold <= 1.0:
            raise ValueError("topic_threshold must be in [0, 1]")
        self.topic_threshold = topic_threshold
        self.contradiction_boost = contradiction_boost
        self.stale_delta = stale_delta

    def tag(
        self,
        incoming: RecentEpisode,
        recent: Iterable[RecentEpisode],
    ) -> Tag:
        best: tuple[float, RecentEpisode] | None = None
        for ep in recent:
            if ep.id == incoming.id:
                continue
            sim = _cosine(incoming.embedding, ep.embedding)
            if best is None or sim > best[0]:
                best = (sim, ep)
        if best is None or best[0] < self.topic_threshold:
            return Tag(level=0.0, reason=REASON_NONE, ref_id=None)

        sim, ref = best
        # Start with the normalised similarity as the base level.
        level = max(0.0, min(1.0, sim))
        reason = REASON_TOPIC

        contradiction = _has_negation_flip(
            incoming.text, ref.text
        ) or _numeric_conflict(incoming.text, ref.text)
        if contradiction:
            level = min(1.0, level + self.contradiction_boost)
            reason = REASON_CONTRADICTION
        else:
            age = incoming.ts - ref.ts
            if age > self.stale_delta:
                reason = REASON_STALE
        return Tag(level=level, reason=reason, ref_id=ref.id)

    # ------------------------------------------------------------ batch api

    def tag_batch(
        self, episodes: Sequence[RecentEpisode]
    ) -> list[Tag]:
        """Tag ``episodes`` in arrival order, each one against earlier ones."""
        tags: list[Tag] = []
        window: list[RecentEpisode] = []
        for ep in episodes:
            tags.append(self.tag(ep, window))
            window.append(ep)
        return tags
