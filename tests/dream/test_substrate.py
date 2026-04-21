"""Unit tests for :class:`src.dream.substrate.MicroKikiSubstrate`.

Spike 2 scope : assert the API shape (types + round-trip) that the
framework-C substrate ABI requires. Full conformance tests (DR-0..DR-4
property tests parametrised over our substrate) run in phase 4 against
upstream ``dream-harness conformance --substrate microkiki``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.dream.substrate import (
    MICROKIKI_SUBSTRATE_NAME,
    MICROKIKI_SUBSTRATE_VERSION,
    MicroKikiSubstrate,
    microkiki_substrate_components,
)


# ---------------------------------------------------------------------------
# Minimal in-process DreamEpisode stub — avoids importing kiki_oniric so
# these tests run in the micro-kiki venv (no dream-of-kiki dep yet).
# The substrate duck-types on ``episode_id``, ``operation_set``,
# ``input_slice`` — see ``MicroKikiSubstrate.consume_episode``.
# ---------------------------------------------------------------------------


@dataclass
class _StubOp:
    value: str


@dataclass
class _StubEpisode:
    episode_id: str
    operation_set: tuple[_StubOp, ...]
    input_slice: dict[str, Any] = field(default_factory=dict)


def _make_episode(ep_id: str, ops: tuple[str, ...], records: list[dict]) -> _StubEpisode:
    return _StubEpisode(
        episode_id=ep_id,
        operation_set=tuple(_StubOp(o) for o in ops),
        input_slice={"beta_records": records},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_substrate_identity() -> None:
    """DR-3 condition-1 surface : name + version + component map."""
    assert MICROKIKI_SUBSTRATE_NAME == "microkiki"
    assert MICROKIKI_SUBSTRATE_VERSION.startswith("C-v")
    comps = microkiki_substrate_components()
    # All upstream-reference keys present.
    for key in (
        "primitives",
        "replay",
        "downscale",
        "restructure",
        "recombine",
        "finite",
        "topology",
        "runtime",
        "swap",
        "p_min",
        "p_equ",
        "p_max",
    ):
        assert key in comps, f"missing component key: {key}"


def test_awake_returns_string() -> None:
    """Mock model → :meth:`awake` returns ``str`` (type contract)."""
    sub = MicroKikiSubstrate(base_model_path=None)
    out = sub.awake("hello, kiki")
    assert isinstance(out, str)
    assert "hello, kiki" in out


def test_consolidate_consumes_episode_updates_state() -> None:
    """Feed a DE with 3 β records ; assert state updates + DR-0 log."""
    sub = MicroKikiSubstrate(seed=1)
    # Seed the adapter with a single tensor so downscale is visible.
    sub.state.adapter["layer0.lora_A"] = np.ones((4, 2), dtype=np.float32)

    records = [
        {"context": f"record-{i}", "outcome": "ok", "saillance_score": 0.5}
        for i in range(3)
    ]
    ids = sub.ingest_beta_records(records)
    assert ids == [0, 1, 2]
    assert len(sub.state.beta_buffer) == 3

    # Pull the freshly-ingested records into the episode's input slice
    # so ``mark_consumed`` can stamp them (DR-1).
    ep = _make_episode(
        "de-test-0001",
        ("replay", "downscale"),
        sub.fetch_unconsumed(limit=10),
    )
    entry = sub.consolidate(ep)

    # DR-0 : one log entry per call, completed=True path.
    assert entry["episode_id"] == "de-test-0001"
    assert entry["completed"] is True
    assert "replay" in entry["operations_executed"]
    assert "downscale" in entry["operations_executed"]
    assert len(sub.state.episode_log) == 1

    # Downscale by 0.99 must shrink the adapter tensor.
    expected = np.ones((4, 2), dtype=np.float32) * 0.99
    np.testing.assert_allclose(
        sub.state.adapter["layer0.lora_A"], expected, rtol=1e-6
    )

    # DR-1 : records now carry ``consumed_by_DE_id``.
    for rec in sub.state.beta_buffer:
        assert rec["consumed_by_DE_id"] == "de-test-0001"


def test_snapshot_roundtrip(tmp_path: Path) -> None:
    """Adapter tensors survive :meth:`snapshot` → :meth:`load_snapshot`."""
    sub = MicroKikiSubstrate()
    sub.state.adapter = {
        "layer0.lora_A": np.arange(12, dtype=np.float32).reshape(3, 4),
        "layer0.lora_B": np.linspace(0, 1, 8, dtype=np.float32).reshape(4, 2),
    }
    snap = sub.snapshot(tmp_path / "adapter.npz")
    assert snap.exists()

    # Mutate then reload ; adapter must match the original tensors.
    sub.state.adapter = {}
    sub.load_snapshot(snap)

    np.testing.assert_array_equal(
        sub.state.adapter["layer0.lora_A"],
        np.arange(12, dtype=np.float32).reshape(3, 4),
    )
    np.testing.assert_allclose(
        sub.state.adapter["layer0.lora_B"],
        np.linspace(0, 1, 8, dtype=np.float32).reshape(4, 2),
        rtol=1e-6,
    )


def test_restructure_raises_not_implemented() -> None:
    """Phase-3 stub : RESTRUCTURE must fail explicitly.

    DR-0 still satisfied — the runtime log records the failure rather
    than silently swallowing it.
    """
    sub = MicroKikiSubstrate()
    ep = _make_episode("de-test-restructure", ("restructure",), [])
    with pytest.raises(NotImplementedError, match="OPLoRA projection"):
        sub.consolidate(ep)
    # DR-0 : the abort path still appended one log entry with
    # ``completed=False`` and populated ``error``.
    assert len(sub.state.episode_log) == 1
    entry = sub.state.episode_log[0]
    assert entry["completed"] is False
    assert entry["error"] is not None


def test_handler_signatures_match_esnn_shape() -> None:
    """DR-3 condition-1 smoke : the 4 factories return callables with
    the expected arity (2, 2, 3, 3) mirroring ``EsnnSubstrate``.
    """
    sub = MicroKikiSubstrate(seed=0)
    # replay : (list[dict], int) -> NDArray
    h = sub.replay_handler_factory()
    out = h([{"context": "c", "outcome": "o", "saillance_score": 0.1}], 5)
    assert isinstance(out, np.ndarray)

    # downscale : (NDArray, float) -> NDArray
    h = sub.downscale_handler_factory()
    arr = np.ones(4, dtype=np.float32)
    out = h(arr, 0.5)
    np.testing.assert_allclose(out, np.full(4, 0.5, dtype=np.float32))

    # recombine : (NDArray, int, int) -> NDArray
    h = sub.recombine_handler_factory()
    lat = np.stack([np.ones(3, dtype=np.float32), np.zeros(3, dtype=np.float32)])
    out = h(lat, 0, 10)
    assert out.shape == (3,)
    assert np.all(out >= 0.0)
