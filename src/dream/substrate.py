"""MicroKikiSubstrate — framework-C substrate prototype (Spike 2).

This module provides a prototype implementation of the ``kiki_oniric``
framework-C substrate ABI (upstream :
``github.com/hypneum-lab/dream-of-kiki``, canonical reference
``kiki_oniric/substrates/esnn_thalamocortical.py`` at spec version
``C-v0.7.0+PARTIAL``).

The substrate composes three pieces of micro-kiki state :

1. An ``mlx_lm`` base model (loaded lazily — the prototype can run
   against a 1.5B-4bit class model, never 35B on kxkm-ai / 62 GB RAM).
2. A LoRA adapter, represented here as an in-memory ``dict[str, ndarray]``
   keyed by weight-path (matches the shape emitted by v4 training).
3. Aeon Atlas + Trace records, abstracted to a ``list[dict]`` β buffer
   compatible with the upstream ``BetaBufferProtocol`` signature. The
   full Aeon wiring lands in phase 3 ; the prototype uses a flat list
   so the contract is exercisable without the Atlas/Trace runtime.

Integration hooks (documented here, **not** implemented in Spike 2) :

* **``scripts/check_forgetting.py``** (future ``post_train_gate.py``) —
  when the S1 retained-benchmark guard trips (angle < 30° AND win-rate
  drop > 0.03), instead of executing a hard rollback, the gate can call
  ``MicroKikiSubstrate.consolidate(DreamEpisode(trigger=SATURATION,
  operation_set=(REPLAY, DOWNSCALE), ...))`` with the stack's Aeon
  records as β input_slice. The replay handler emits a corrective
  weight-delta that is promoted via ``swap_atomic`` only if S1 + S2
  guards pass — degrading the rollback to a soft recovery path.
* **v4 training loop between stacks** — at curriculum boundaries, the
  training driver dumps the Aeon Trace episodes touched during the
  just-finished stack as ``.jsonl``, feeds them to
  ``MicroKikiSubstrate.ingest_beta_records()``, then schedules a
  ``DreamEpisode(trigger=SCHEDULED, operation_set=(REPLAY, DOWNSCALE,
  RESTRUCTURE))`` before the next stack boots. This is the canonical
  "sleep between curriculum steps" pattern (upstream §4.3).

DR-axiom mapping (see
``docs/research/2026-04-21-dream-of-kiki-substrate-spec.md``) :

* **DR-0 Accountability** — ``consume_episode()`` and ``consolidate()``
  each log an episode-level record ; failure does not suppress the log.
* **DR-1 Episodic conservation** — ``ingest_beta_records()`` marks
  records with ``consumed_by_DE_id`` on consumption, mirroring the I1
  runtime guard.
* **DR-2 Compositionality (weakened)** — ``consolidate()`` executes
  operations in the canonical order (replay < downscale < restructure <
  recombine) so the precondition "no RESTRUCTURE before REPLAY" holds
  trivially.
* **DR-3 Substrate-agnosticism** — this class is the *evidence* for
  DR-3 condition 1 (typed Protocol signatures) ; conditions 2 and 3
  (axiom property tests + invariant runtime enforcement) are deferred
  to phase 4.
* **DR-4 Profile chain inclusion** — the substrate exposes the 4 op
  factories so any P_min / P_equ / P_max composition upstream-built
  can bind handlers from this substrate uniformly.

Reference (upstream) :
``docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`` §2, §4, §6.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# DualVer : the empirical axis is ``+UNSTABLE`` until phase 4 runs the
# upstream conformance harness (``dream-harness conformance --substrate
# microkiki``) on an Apple-Silicon host and promotes to ``+PARTIAL``.
MICROKIKI_SUBSTRATE_NAME = "microkiki"
MICROKIKI_SUBSTRATE_VERSION = "C-v0.7.0+UNSTABLE"


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------


@dataclass
class MicroKikiState:
    """Mutable substrate state — mlx_lm model + LoRA adapter + β buffer.

    Keeps everything as plain Python / numpy structures so the prototype
    is exercisable on kxkm-ai (Linux x86_64) without an MLX device. The
    production substrate will replace ``adapter`` with the real MLX
    parameter tree and ``base_model`` with ``mlx_lm.load(...)`` output.
    """

    base_model: Any = None  # mlx_lm Model | None — lazy-loaded
    tokenizer: Any = None   # mlx_lm Tokenizer | None
    adapter: dict[str, NDArray] = field(default_factory=dict)  # key → float32 tensor
    # β buffer : a flat list mirroring the ``BetaBufferProtocol`` API
    # surface (append / fetch_unconsumed / mark_consumed).
    beta_buffer: list[dict[str, Any]] = field(default_factory=list)
    # DR-0 audit log — every executed episode appends one entry.
    episode_log: list[dict[str, Any]] = field(default_factory=list)

    def next_beta_id(self) -> int:
        return len(self.beta_buffer)


# ---------------------------------------------------------------------------
# Substrate
# ---------------------------------------------------------------------------


@dataclass
class MicroKikiSubstrate:
    """micro-kiki framework-C substrate prototype.

    The prototype composes an ``mlx_lm`` base model + a LoRA adapter +
    a β buffer of Aeon records. All 4 op handlers are exposed as
    closure factories matching the ``EsnnSubstrate`` shape exactly.

    Parameters
    ----------
    base_model_path :
        Optional path to an ``mlx_lm`` model checkpoint. When ``None``,
        the substrate runs in **pure-stub mode** — ``awake()`` returns
        a canned string, handlers operate on adapter tensors only, and
        no model is loaded. The demo CLI uses ``None`` by default so
        the smoke test runs on any host.
    seed :
        Numpy RNG seed for recombine + any stochastic ops.
    """

    base_model_path: str | None = None
    seed: int = 0
    state: MicroKikiState = field(default_factory=MicroKikiState)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------
    # Awake-side primitive — text generation handle
    # ------------------------------------------------------------------

    def awake(self, prompt: str, max_tokens: int = 16) -> str:
        """Run the awake base+adapter forward pass — returns a string.

        When ``base_model_path`` is ``None`` the substrate returns a
        deterministic stub (``"[stub awake] <prompt>"``) so tests can
        assert on the type without loading a model. With a real
        ``mlx_lm`` checkpoint available, falls through to
        ``mlx_lm.generate(...)``.

        DR-0 — does **not** append to ``episode_log``; awake calls are
        not dream episodes, they are the process being consolidated.
        """
        if self.state.base_model is None or self.state.tokenizer is None:
            return f"[stub awake] {prompt}"

        try:  # pragma: no cover — env-gated (MLX present)
            from mlx_lm import generate  # type: ignore[import-not-found]

            return str(
                generate(
                    self.state.base_model,
                    self.state.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    verbose=False,
                )
            )
        except Exception as exc:  # pragma: no cover - env-gated
            logger.warning("awake() fell back to stub: %s", exc)
            return f"[stub awake / fallback] {prompt}"

    # ------------------------------------------------------------------
    # β-buffer ingest — maps Aeon records into framework-C β semantics
    # ------------------------------------------------------------------

    def ingest_beta_records(self, records: list[dict[str, Any]]) -> list[int]:
        """Append records to the β buffer, return their assigned ids.

        Each record is expected to carry at minimum ``context`` (str),
        ``outcome`` (str), ``saillance_score`` (float). The ``"x"`` /
        ``"y"`` fields can be pre-populated for direct consumption by
        the upstream ``replay_handler_mlx`` ; otherwise they are
        synthesised by :meth:`_record_to_xy` lazily at replay time.

        DR-1 — every record is tagged with ``consumed_by_DE_id=None``
        on ingest so the I1 sweep can discover un-consolidated
        records. ``mark_consumed`` (called by the replay handler)
        populates the field.
        """
        ids: list[int] = []
        for rec in records:
            rec_id = self.state.next_beta_id()
            self.state.beta_buffer.append(
                {
                    "id": rec_id,
                    "context": rec.get("context", ""),
                    "outcome": rec.get("outcome", ""),
                    "saillance_score": float(rec.get("saillance_score", 0.0)),
                    "consumed_by_DE_id": None,
                    "x": rec.get("x"),
                    "y": rec.get("y"),
                }
            )
            ids.append(rec_id)
        return ids

    def fetch_unconsumed(self, limit: int) -> list[dict[str, Any]]:
        """β primitive — return up to ``limit`` records pending DE consumption."""
        unconsumed = [r for r in self.state.beta_buffer if r["consumed_by_DE_id"] is None]
        return unconsumed[:limit]

    def mark_consumed(self, record_ids: list[int], de_id: str) -> None:
        """β primitive — stamp records as consolidated by the given DE."""
        ids_set = set(record_ids)
        for rec in self.state.beta_buffer:
            if rec["id"] in ids_set:
                rec["consumed_by_DE_id"] = de_id

    # ------------------------------------------------------------------
    # Handler factories — the 4 canonical operations
    # ------------------------------------------------------------------

    def replay_handler_factory(self) -> Callable[[list[dict], int], NDArray]:
        """A-Walker/Stickgold replay → LoRA gradient proxy.

        Shape mirrors ``EsnnSubstrate.replay_handler_factory`` for
        DR-3 conformance. Without MLX available, the handler emits a
        pseudo-gradient computed as the mean of the β record vectors
        — sufficient to exercise the swap protocol in phase 3.
        """

        def handler(beta_records: list[dict], n_steps: int = 20) -> NDArray:
            if not beta_records:
                return np.zeros(1, dtype=np.float32)
            vectors: list[NDArray] = []
            for rec in beta_records:
                xy = self._record_to_xy(rec)
                if xy is None:
                    continue
                vectors.append(xy)
            if not vectors:
                return np.zeros(1, dtype=np.float32)
            return np.mean(np.stack(vectors), axis=0).astype(np.float32)

        return handler

    def downscale_handler_factory(
        self,
    ) -> Callable[[NDArray, float], NDArray]:
        """B-Tononi SHY → multiplicative scaling on adapter tensors."""

        def handler(weights: NDArray, factor: float) -> NDArray:
            if not (0.0 < factor <= 1.0):
                raise ValueError(f"shrink_factor must be in (0, 1], got {factor}")
            return (weights * factor).astype(weights.dtype, copy=False)

        return handler

    def restructure_handler_factory(
        self,
    ) -> Callable[[dict, str, str], dict]:
        """D-Friston FEP restructure → adapter layer topology edit.

        phase 2 stub : supports ``op ∈ {"activate", "deactivate"}`` on
        adapter keys. Real rank-swap / layer-add lands in phase 3 once
        OPLoRA projection is wired into the MLX pipeline.
        """

        def handler(adapter: dict[str, NDArray], op: str, key: str) -> dict[str, NDArray]:
            valid_ops = {"activate", "deactivate"}
            if op not in valid_ops:
                raise ValueError(f"op must be one of {sorted(valid_ops)}, got {op!r}")
            new = dict(adapter)
            if op == "deactivate" and key in new:
                new[key] = np.zeros_like(new[key])
            elif op == "activate" and key not in new:
                # Minimal stub : a singleton zero tensor marking the
                # key as present. Real impl bootstraps from OPLoRA
                # (see docstring).
                new[key] = np.zeros(1, dtype=np.float32)
            return new

        return handler

    def recombine_handler_factory(
        self,
    ) -> Callable[[NDArray, int, int], NDArray]:
        """C-Hobson recombine → latent interpolation + noise.

        Signature matches ``EsnnSubstrate.recombine_handler_factory``
        for DR-3 uniformity.
        """

        def handler(latents: NDArray, seed: int = 0, n_steps: int = 10) -> NDArray:
            if latents.shape[0] < 2:
                raise ValueError(f"recombine needs >=2 latents, got {latents.shape[0]}")
            rng = np.random.default_rng(seed)
            alpha = float(rng.random())
            mixed = alpha * latents[0] + (1.0 - alpha) * latents[1]
            mixed = np.maximum(mixed, 0.0)
            return mixed.astype(np.float32)

        return handler

    # ------------------------------------------------------------------
    # Episode consumption — one DE, one log entry (DR-0)
    # ------------------------------------------------------------------

    def consume_episode(self, episode: Any) -> dict[str, Any]:
        """Execute one ``DreamEpisode`` against the substrate.

        Accepts the upstream ``kiki_oniric.dream.episode.DreamEpisode``
        type (duck-typed here to avoid a hard dep on the upstream
        package at import time). The method :

        1. Reads ``episode.input_slice["beta_records"]`` (list[dict]),
        2. Dispatches to the 4 handler factories in canonical order
           (replay < downscale < restructure < recombine, DR-2'),
        3. Marks the consumed records with ``episode.episode_id`` (DR-1),
        4. Appends an ``EpisodeLogEntry``-shaped dict to
           ``state.episode_log`` (DR-0 : one log entry per call, even
           when the handler raises).

        Returns the log entry as a dict so callers can inspect the
        result without touching the internal audit list.
        """
        ep_id = getattr(episode, "episode_id", "de-unknown")
        ops = tuple(getattr(episode, "operation_set", ()))
        op_names = tuple(getattr(op, "value", str(op)) for op in ops)
        input_slice = getattr(episode, "input_slice", {}) or {}
        records = list(input_slice.get("beta_records", []))

        executed: list[str] = []
        error: str | None = None
        completed = False

        try:
            # REPLAY
            if "replay" in op_names:
                self.replay_handler_factory()(records, 20)
                executed.append("replay")
            # DOWNSCALE — shrink every adapter tensor by 0.99 (SHY default)
            if "downscale" in op_names:
                for k, v in list(self.state.adapter.items()):
                    self.state.adapter[k] = self.downscale_handler_factory()(v, 0.99)
                executed.append("downscale")
            # RESTRUCTURE — no-op in Spike 2 (phase-3 feature)
            if "restructure" in op_names:
                raise NotImplementedError(
                    "restructure handler requires OPLoRA projection — phase 3"
                )
            # RECOMBINE — stub unless episode carries explicit latents
            if "recombine" in op_names:
                latents = input_slice.get("latents")
                if latents is not None:
                    self.recombine_handler_factory()(np.asarray(latents), self.seed)
                executed.append("recombine")

            # DR-1 — stamp every β record touched
            ids = [r.get("id") for r in records if "id" in r]
            if ids:
                self.mark_consumed(ids, ep_id)

            completed = True
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            entry = {
                "episode_id": ep_id,
                "operations_executed": tuple(executed),
                "completed": completed,
                "error": error,
            }
            self.state.episode_log.append(entry)
        return entry

    def consolidate(self, episode: Any) -> dict[str, Any]:
        """Alias for :meth:`consume_episode` — the hook name used by
        the future ``post_train_gate.py`` ``--consolidate-on-warning``
        flag (see module docstring).
        """
        return self.consume_episode(episode)

    # ------------------------------------------------------------------
    # γ — snapshot / checkpoint
    # ------------------------------------------------------------------

    def snapshot(self, path: str | Path) -> Path:
        """Persist the current adapter state to ``path`` (numpy npz).

        Returns the path that was written. Round-trips cleanly via
        :meth:`load_snapshot` — see ``tests/dream/test_substrate.py``.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, **self.state.adapter)
        return target if target.suffix == ".npz" else target.with_suffix(".npz")

    def load_snapshot(self, path: str | Path) -> None:
        """Restore adapter state from a ``snapshot()`` output."""
        target = Path(path)
        if not target.exists() and target.with_suffix(".npz").exists():
            target = target.with_suffix(".npz")
        data = np.load(target, allow_pickle=False)
        self.state.adapter = {k: np.asarray(data[k]) for k in data.files}

    # ------------------------------------------------------------------
    # β primitive — file loader (jsonl)
    # ------------------------------------------------------------------

    def load_beta_jsonl(self, path: str | Path) -> list[int]:
        """Read a ``.jsonl`` of Aeon records and ingest them."""
        records: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return self.ingest_beta_records(records)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _record_to_xy(self, rec: dict[str, Any]) -> NDArray | None:
        """Coerce a β record to a single vector for replay.

        Prefers the pre-populated ``x`` field. Falls back to a stable
        hash-based vector over ``context`` when ``x`` is missing so
        the prototype is exercisable on raw Aeon records without a
        pre-embedding pass.
        """
        x = rec.get("x")
        if x is not None:
            return np.asarray(x, dtype=np.float32)
        ctx = str(rec.get("context", ""))
        if not ctx:
            return None
        # Deterministic fake embedding : 8-dim vector from byte hash.
        digest = np.frombuffer(
            (ctx.encode("utf-8") * 8)[:64].ljust(64, b"\x00"), dtype=np.uint8
        )
        return (digest[:8].astype(np.float32) / 255.0).astype(np.float32)


# ---------------------------------------------------------------------------
# DR-3 condition-1 manifest
# ---------------------------------------------------------------------------


def microkiki_substrate_components() -> dict[str, str]:
    """Return the canonical component map for the micro-kiki substrate.

    Mirrors ``esnn_substrate_components()`` in shape. Values are dotted
    paths inside micro-kiki ; they point to the modules that will host
    the corresponding primitive / guard / runtime once phases 3-4 land.
    Entries annotated with ``# phase-3+`` are not yet implemented.
    """
    return {
        # 8 typed Protocols (shared contract — lives upstream)
        "primitives": "kiki_oniric.core.primitives",
        # 4 operations — factory methods on this substrate class
        "replay": "src.dream.substrate",
        "downscale": "src.dream.substrate",
        "restructure": "src.dream.substrate",  # phase-3+ for real impl
        "recombine": "src.dream.substrate",
        # 2 invariant guards (shared, will be imported from upstream)
        "finite": "kiki_oniric.dream.guards.finite",
        "topology": "kiki_oniric.dream.guards.topology",  # phase-3+
        # Runtime + swap (shared, will be imported from upstream)
        "runtime": "kiki_oniric.dream.runtime",
        "swap": "kiki_oniric.dream.swap",
        # Profiles (shared — micro-kiki targets P_min in phase 3)
        "p_min": "kiki_oniric.profiles.p_min",
        "p_equ": "kiki_oniric.profiles.p_equ",
        "p_max": "kiki_oniric.profiles.p_max",
    }
