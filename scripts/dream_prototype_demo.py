#!/usr/bin/env python3
"""Spike-2 demo : run one replay cycle of the MicroKikiSubstrate.

Prints the substrate state before and after a single
``DreamEpisode(trigger=SCHEDULED, operation_set=(REPLAY, DOWNSCALE))`` so
the integration shape is inspectable without a 35B model.

Usage
-----

.. code-block:: bash

    # Pure-stub mode — no base model, no adapter, no records.
    python scripts/dream_prototype_demo.py

    # Load a mock 1.5B base model + an adapter + records.
    python scripts/dream_prototype_demo.py \\
        --base  mlx-community/Qwen2.5-1.5B-Instruct-4bit \\
        --adapter /path/to/adapter.npz \\
        --records /path/to/aeon_records.jsonl

The script never loads a 35B model (kxkm-ai : 62 GB RAM << 67 GB BF16).
Phase-3 adds full mlx_lm + real LoRA wiring on the Mac Studio.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Resolve the repo root so ``import src.dream.substrate`` works when
# the script is invoked with its absolute path from anywhere.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.dream.substrate import (  # noqa: E402
    MICROKIKI_SUBSTRATE_NAME,
    MICROKIKI_SUBSTRATE_VERSION,
    MicroKikiSubstrate,
)


logger = logging.getLogger("dream_prototype_demo")


# ---------------------------------------------------------------------------
# Local in-process DreamEpisode stub — identical shape to the real
# ``kiki_oniric.dream.episode.DreamEpisode`` but dep-free so the demo
# runs from the micro-kiki venv alone.
# ---------------------------------------------------------------------------


@dataclass
class _Op:
    value: str


@dataclass
class _Episode:
    episode_id: str
    operation_set: tuple
    input_slice: dict = field(default_factory=dict)


def _build_episode(records: list[dict]) -> _Episode:
    return _Episode(
        episode_id="de-demo-0001",
        operation_set=(_Op("replay"), _Op("downscale")),
        input_slice={"beta_records": records},
    )


# ---------------------------------------------------------------------------
# Adapter loader — numpy npz (phase 2 shape). Phase 3 wires this to the
# real mlx_lm adapter format (``adapters.safetensors``).
# ---------------------------------------------------------------------------


def _load_adapter(path: Path | None) -> dict[str, np.ndarray]:
    if path is None:
        # Default toy adapter : one 8×4 LoRA matrix so downscale is visible.
        return {"layer0.lora_A": np.ones((8, 4), dtype=np.float32)}
    if not path.exists():
        raise FileNotFoundError(f"adapter not found: {path}")
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        return {k: np.asarray(data[k]) for k in data.files}
    raise ValueError(
        f"unsupported adapter format: {path.suffix} "
        "(phase 3 adds safetensors support)"
    )


def _load_records(path: Path | None) -> list[dict]:
    if path is None:
        # Synthesise 3 deterministic records.
        return [
            {
                "context": f"synthetic-record-{i}",
                "outcome": "ok",
                "saillance_score": 0.1 + i * 0.2,
            }
            for i in range(3)
        ]
    if not path.exists():
        raise FileNotFoundError(f"records file not found: {path}")
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base",
        default=None,
        help=(
            "Optional mlx_lm base-model path or HF repo id. When omitted, "
            "the substrate runs in stub mode (no model loaded)."
        ),
    )
    parser.add_argument(
        "--adapter",
        default=None,
        type=Path,
        help="Optional LoRA adapter .npz path (phase-2 toy format).",
    )
    parser.add_argument(
        "--records",
        default=None,
        type=Path,
        help="Optional Aeon β-records .jsonl file (defaults to 3 synthetic records).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # 1. Build the substrate and print its identity.
    sub = MicroKikiSubstrate(base_model_path=args.base, seed=args.seed)
    sub.state.adapter = _load_adapter(args.adapter)
    records = _load_records(args.records)
    sub.ingest_beta_records(records)

    print(f"# {MICROKIKI_SUBSTRATE_NAME} @ {MICROKIKI_SUBSTRATE_VERSION}")
    print(f"# base_model_path = {args.base!r} (None → stub awake)")
    print(f"# adapter keys    = {sorted(sub.state.adapter)}")
    print(f"# β buffer size   = {len(sub.state.beta_buffer)}")

    # 2. Probe awake() — must return a string (DR-3 surface smoke).
    awake_out = sub.awake("Explain one-sentence replay consolidation.")
    print(f"# awake() sample  = {awake_out[:96]!r}")

    # 3. Print adapter tensor sums before replay.
    def _sums() -> dict[str, float]:
        return {k: float(v.sum()) for k, v in sub.state.adapter.items()}

    print("# pre-DE adapter sums :", _sums())

    # 4. Build and consume a canonical DE(replay, downscale).
    ep = _build_episode(sub.fetch_unconsumed(limit=64))
    entry = sub.consolidate(ep)

    # 5. Print post-state and the DR-0 log entry.
    print("# post-DE adapter sums:", _sums())
    print(f"# DR-0 log entry      = {entry}")
    unconsumed = [r for r in sub.state.beta_buffer if r["consumed_by_DE_id"] is None]
    print(f"# unconsumed records  = {len(unconsumed)} / {len(sub.state.beta_buffer)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
