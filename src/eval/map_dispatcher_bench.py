"""Benchmark v0.2 Dispatcher vs MAP conflict-monitor module (story-2).

The v0.2 Dispatcher is specified in
``docs/specs/2026-04-15-cognitive-layer-design.md §1``: a YAML
matrix maps 32 router scores to 7 meta-intents. It has no training,
no state, and runs in microseconds.

There is no runnable Dispatcher code on main yet — this bench builds a
**conceptual model** of the v0.2 Dispatcher based on the spec, then
runs it through the MAP harness's conflict-monitor bench. The goal is
retrospective spec-level validation: we want to know whether the
Dispatcher's implied conflict signal matches MAP's entropy-based
reference.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from src.eval.map_harness import (
    MAPHarness,
    MockAgent,
    gen_conflict_prompts,
)

# v0.2 meta-intents (7) per configs/meta_intents.yaml (spec-level).
META_INTENTS: tuple[str, ...] = (
    "code",
    "electronics",
    "system",
    "knowledge",
    "creative",
    "chat",
    "security",
)


@dataclass
class V02DispatcherAgent:
    """Conceptual model of v0.2 Dispatcher as a MAP conflict monitor.

    The Dispatcher's design (per the v0.2 cognitive spec) is:

    1. Take 32 router sigmoid activations.
    2. Multiply by a 32×7 YAML weight matrix → 7 meta-intent scores.
    3. Normalise via softmax or L1 and return the top-1.

    To map that onto the MAP conflict monitor we use the **normalised
    Shannon entropy of the meta-intent distribution** as the conflict
    signal. This mirrors what the Dispatcher would report to the
    Negotiator when the top-1 and top-2 meta-intents are close in score.
    """

    chat_floor: float = 0.20
    activation_cap: int = 4  # v0.2 policy: max 4 active stacks

    def conflict_monitor(
        self, intents: Sequence[tuple[str, float]]
    ) -> float:
        if not intents:
            return 0.0
        # Clip to cap (v0.2 "max 4 active stacks" constraint).
        sorted_intents = sorted(intents, key=lambda p: -p[1])
        sorted_intents = sorted_intents[: self.activation_cap]
        scores = [s for _, s in sorted_intents]
        total = sum(scores)
        if total <= 0:
            return 0.0
        probs = [s / total for s in scores]
        n = len(probs)
        if n <= 1:
            return 0.0
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        norm = entropy / math.log(n)
        # Chat floor is the "default-to-chat" safety net: if the top
        # meta-intent is below the chat floor, the Dispatcher treats the
        # turn as ambiguous, which MAP would encode as higher conflict.
        if scores[0] < self.chat_floor:
            norm = min(1.0, norm + 0.2)
        return norm

    # --- Unused MAP protocol methods ---------------------------------
    # The Dispatcher does not implement state prediction, pairwise
    # evaluation, decomposition, or coordination; stub them with the
    # MockAgent baseline so the protocol stays satisfied and metric
    # isolation is clean.

    def state_predict(self, history: Sequence[str]) -> str:
        return "unknown"

    def evaluate_pair(self, a: str, b: str) -> str:
        return "a"

    def decompose(self, goal: str) -> list[str]:
        return []

    def coordinate(self, state: dict[str, Any]) -> str:
        return "act"


@dataclass
class BenchResult:
    """Structured result for story-2's JSON report."""

    agent: str
    n: int
    agreement_rate: float
    false_positive_rate: float
    latency_ms: float
    agreement_delta_vs_mock: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "map-dispatcher-bench/1.0",
            "agent": self.agent,
            "n": self.n,
            "agreement_rate": round(self.agreement_rate, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "latency_ms": round(self.latency_ms, 4),
            "agreement_delta_vs_mock": round(
                self.agreement_delta_vs_mock, 4
            ),
            "notes": self.notes,
        }


def run(n: int = 50, seed: int = 42, tolerance: float = 0.15) -> BenchResult:
    """Run the Dispatcher bench and return a BenchResult.

    The bench executes the MAP conflict-monitor synthetic set against
    two agents: the MockAgent baseline (entropy of raw scores) and the
    v0.2 Dispatcher conceptual model (entropy after 4-stack cap and
    chat-floor adjustment). The delta quantifies whether the v0.2
    design choices move the conflict signal closer to or further from
    the MAP reference.
    """
    harness = MAPHarness(
        seed=seed, conflict_n=n, conflict_tolerance=tolerance
    )
    dispatcher = V02DispatcherAgent()
    mock = MockAgent()

    dispatcher_metrics = harness.run_conflict(dispatcher)
    mock_metrics = harness.run_conflict(mock)

    delta = dispatcher_metrics["agreement_rate"] - mock_metrics[
        "agreement_rate"
    ]

    notes = [
        "Conceptual model of the v0.2 Dispatcher; no runnable code on main.",
        "Conflict = normalised Shannon entropy after 4-stack cap.",
        "Chat-floor adjustment raises conflict when top score < 0.20.",
        "Synthetic benchmark — not a MAP replication; descriptive only.",
    ]

    return BenchResult(
        agent="V02DispatcherAgent",
        n=dispatcher_metrics["n"],
        agreement_rate=dispatcher_metrics["agreement_rate"],
        false_positive_rate=dispatcher_metrics["false_positive_rate"],
        latency_ms=dispatcher_metrics["latency_ms"],
        agreement_delta_vs_mock=delta,
        notes=notes,
    )


def _main() -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="MAP conflict-monitor bench vs v0.2 Dispatcher"
    )
    parser.add_argument("--out", default="results/map-dispatcher.json")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=0.15)
    args = parser.parse_args()

    result = run(n=args.n, seed=args.seed, tolerance=args.tolerance)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {out}")
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
