"""Tests for the SleepTagger conflict-aware tagger (story-8)."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

import pytest

from src.cognitive.sleep_tagger import (
    REASON_CONTRADICTION,
    REASON_NONE,
    REASON_STALE,
    REASON_TOPIC,
    RecentEpisode,
    SleepTagger,
    Tag,
)


UTC = timezone.utc


def _ep(
    eid: str,
    text: str,
    vec: list[float],
    *,
    minutes: int = 0,
    days: int = 0,
) -> RecentEpisode:
    ts = datetime(2026, 4, 16, tzinfo=UTC) + timedelta(
        minutes=minutes, days=days
    )
    return RecentEpisode(id=eid, text=text, embedding=vec, ts=ts)


def test_no_recent_returns_none() -> None:
    tagger = SleepTagger()
    incoming = _ep("a", "hello world", [1.0, 0.0])
    tag = tagger.tag(incoming, [])
    assert tag.reason == REASON_NONE
    assert tag.level == 0.0


def test_topic_hit_without_contradiction() -> None:
    tagger = SleepTagger()
    prev = _ep("p", "the battery is green", [1.0, 0.0, 0.0])
    incoming = _ep("n", "the battery still reports green", [0.95, 0.1, 0.0])
    tag = tagger.tag(incoming, [prev])
    assert tag.reason == REASON_TOPIC
    assert tag.ref_id == "p"
    assert tag.level > 0.0


def test_contradiction_negation_boost() -> None:
    tagger = SleepTagger()
    prev = _ep("p", "the door is open today", [1.0, 0.0])
    incoming = _ep("n", "the door is not open today", [0.95, 0.05])
    tag = tagger.tag(incoming, [prev])
    assert tag.reason == REASON_CONTRADICTION
    # Contradiction should push level above plain cosine.
    plain = SleepTagger(contradiction_boost=0.0).tag(incoming, [prev])
    assert tag.level >= plain.level


def test_contradiction_numeric_supersession() -> None:
    tagger = SleepTagger()
    prev = _ep("p", "voltage=12.0 state=ok", [1.0, 0.0, 0.0])
    incoming = _ep("n", "voltage=11.2 state=ok", [0.98, 0.0, 0.02])
    tag = tagger.tag(incoming, [prev])
    assert tag.reason == REASON_CONTRADICTION


def test_stale_topic_after_delta() -> None:
    tagger = SleepTagger(stale_delta=timedelta(days=3))
    prev = _ep("p", "sensor alpha online", [1.0, 0.0])
    incoming = _ep(
        "n",
        "sensor alpha online and nominal",
        [0.97, 0.05],
        days=10,
    )
    tag = tagger.tag(incoming, [prev])
    assert tag.reason == REASON_STALE


def test_below_topic_threshold_returns_none() -> None:
    tagger = SleepTagger(topic_threshold=0.9)
    prev = _ep("p", "weather report", [1.0, 0.0])
    incoming = _ep("n", "battery log", [0.0, 1.0])
    tag = tagger.tag(incoming, [prev])
    assert tag.reason == REASON_NONE


def test_precision_recall_on_synthetic_set() -> None:
    """Acceptance: precision >= 0.8, recall >= 0.7 on planted conflicts."""
    rng = random.Random(2026)
    tagger = SleepTagger(topic_threshold=0.75)

    dim = 64

    episodes: list[RecentEpisode] = []
    planted_pairs: list[tuple[str, str]] = []

    # 10 base episodes with numeric statements on topic-specific keys,
    # then 10 conflicts (same vector + numeric supersession on the same
    # key) and 10 unrelated noise with disjoint keys and fresh vectors.
    base_vectors: list[list[float]] = []
    for i in range(10):
        vec = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        base_vectors.append(vec)
        text = f"sensor-{i} reading v{i}={20 + i} status ok"
        episodes.append(_ep(f"base-{i}", text, vec, minutes=i))

    for i in range(10):
        vec = list(base_vectors[i])
        text = f"sensor-{i} reading v{i}={10 + i} status ok"
        conflict_ep = _ep(f"conflict-{i}", text, vec, minutes=100 + i)
        episodes.append(conflict_ep)
        planted_pairs.append((f"base-{i}", conflict_ep.id))

    for i in range(10):
        vec = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        text = f"unrelated-{i} random payload n{i}={500 + i}"
        episodes.append(_ep(f"noise-{i}", text, vec, minutes=200 + i))

    tags = tagger.tag_batch(episodes)

    conflict_ids = {pair[1] for pair in planted_pairs}
    flagged_contradiction = {
        ep.id for ep, tag in zip(episodes, tags)
        if tag.reason == REASON_CONTRADICTION
    }

    tp = len(conflict_ids & flagged_contradiction)
    fp = len(flagged_contradiction - conflict_ids)
    fn = len(conflict_ids - flagged_contradiction)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    assert precision >= 0.8
    assert recall >= 0.7


def test_tag_batch_matches_sequential_calls() -> None:
    tagger = SleepTagger()
    a = _ep("a", "door open", [1.0, 0.0], minutes=0)
    b = _ep("b", "door closed", [0.95, 0.1], minutes=1)
    batch = tagger.tag_batch([a, b])
    assert batch[0].reason == REASON_NONE
    assert batch[1].reason == REASON_CONTRADICTION
    assert batch[1].ref_id == "a"
