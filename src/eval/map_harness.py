"""MAP harness — retrospective validation of v0.2 cognitive layer.

Paper: Nature Communications 2025, s41467-025-63804-5 (MAP).

This harness is deliberately lightweight (stdlib + optional numpy). It
provides:

* Synthetic benchmark generators for each of the 5 MAP modules.
* A `MAPCompatibleAgent` protocol that v0.2 adapters can implement.
* A `MAPHarness` that runs all 5 benches and emits a JSON report.

The numbers produced here are **design-validation signals**, not a
replication of the MAP paper. See `docs/specs/map-paper-spec.md`.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

SCHEMA_VERSION = "map-harness/1.0"


# ---------------------------------------------------------------------------
# Enum & protocol
# ---------------------------------------------------------------------------


class MAPModule(str, Enum):
    """The 5 MAP modules (cf. docs/specs/map-paper-spec.md §1)."""

    CONFLICT_MONITOR = "conflict_monitor"
    STATE_PREDICTOR = "state_predictor"
    EVALUATOR = "evaluator"
    DECOMPOSER = "decomposer"
    COORDINATOR = "coordinator"


class MAPCompatibleAgent(Protocol):
    """Minimal surface a v0.2 adapter must implement to be benched."""

    def conflict_monitor(
        self, intents: Sequence[tuple[str, float]]
    ) -> float: ...

    def state_predict(self, history: Sequence[str]) -> str: ...

    def evaluate_pair(self, a: str, b: str) -> str: ...

    def decompose(self, goal: str) -> list[str]: ...

    def coordinate(self, state: dict[str, Any]) -> str: ...


# ---------------------------------------------------------------------------
# Synthetic benchmark generators
# ---------------------------------------------------------------------------


@dataclass
class BenchItem:
    """Single synthetic item for any module."""

    module: MAPModule
    payload: dict[str, Any]
    reference: Any


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def gen_conflict_prompts(n: int = 50, seed: int = 42) -> list[BenchItem]:
    """Generate synthetic conflict-monitor prompts.

    Each item has a list of (intent, score) pairs and a reference
    conflict level derived from the Shannon entropy of the scores,
    normalised to [0, 1].
    """
    rng = _rng(seed)
    items: list[BenchItem] = []
    intents_pool = [
        "code",
        "electronics",
        "system",
        "knowledge",
        "creative",
        "chat",
        "security",
    ]
    for i in range(n):
        k = rng.randint(2, 5)
        picks = rng.sample(intents_pool, k)
        # Some items are deliberately conflicted (flat distribution),
        # others clear (one dominant score).
        if i % 3 == 0:
            scores = [rng.uniform(0.4, 0.6) for _ in picks]  # conflicted
        else:
            scores = [0.1] * k
            scores[0] = rng.uniform(0.7, 0.95)  # clear
        total = sum(scores)
        probs = [s / total for s in scores]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        norm_entropy = entropy / math.log(k)  # in [0, 1]
        items.append(
            BenchItem(
                module=MAPModule.CONFLICT_MONITOR,
                payload={"intents": list(zip(picks, scores))},
                reference=round(norm_entropy, 4),
            )
        )
    return items


def gen_trajectory_prompts(n: int = 50, seed: int = 43) -> list[BenchItem]:
    """Generate state-predictor prompts.

    Each item has a short history and a reference next token.
    """
    rng = _rng(seed)
    verbs = ["opens", "closes", "reads", "writes", "queries"]
    objects = ["file", "socket", "database", "memory", "log"]
    items: list[BenchItem] = []
    for _ in range(n):
        verb = rng.choice(verbs)
        obj = rng.choice(objects)
        history = [f"user {verb} {obj}", f"system acknowledges {obj}"]
        reference = f"user reads {obj}"  # deterministic continuation
        items.append(
            BenchItem(
                module=MAPModule.STATE_PREDICTOR,
                payload={"history": history},
                reference=reference,
            )
        )
    return items


def gen_judge_prompts(n: int = 50, seed: int = 44) -> list[BenchItem]:
    """Generate evaluator pairwise-judgement prompts.

    Each item has two candidate responses and a reference winner.
    """
    rng = _rng(seed)
    items: list[BenchItem] = []
    for i in range(n):
        if i % 2 == 0:
            a = "short helpful answer with evidence."
            b = "irrelevant"
            winner = "a"
        else:
            a = "irrelevant"
            b = "long helpful answer with evidence and citations."
            winner = "b"
        # Randomise order sometimes
        if rng.random() < 0.25:
            a, b = b, a
            winner = "b" if winner == "a" else "a"
        items.append(
            BenchItem(
                module=MAPModule.EVALUATOR,
                payload={"pair": (a, b)},
                reference=winner,
            )
        )
    return items


def gen_plan_prompts(n: int = 20, seed: int = 45) -> list[BenchItem]:
    """Generate decomposer prompts with reference plans."""
    templates = [
        ("cook pasta", ["boil water", "add pasta", "drain", "serve"]),
        ("send email", ["open client", "compose", "review", "send"]),
        ("debug crash", ["reproduce", "collect logs", "bisect", "fix"]),
        ("deploy build", ["run tests", "tag release", "push", "verify"]),
    ]
    rng = _rng(seed)
    items: list[BenchItem] = []
    for _ in range(n):
        goal, plan = rng.choice(templates)
        items.append(
            BenchItem(
                module=MAPModule.DECOMPOSER,
                payload={"goal": goal},
                reference=plan,
            )
        )
    return items


def gen_meta_prompts(n: int = 50, seed: int = 46) -> list[BenchItem]:
    """Generate coordinator meta-decision prompts."""
    rng = _rng(seed)
    items: list[BenchItem] = []
    for i in range(n):
        uncertainty = rng.uniform(0, 1)
        if uncertainty > 0.7:
            decision = "reflect"
        elif uncertainty > 0.4:
            decision = "plan"
        else:
            decision = "act"
        items.append(
            BenchItem(
                module=MAPModule.COORDINATOR,
                payload={"state": {"uncertainty": round(uncertainty, 3)}},
                reference=decision,
            )
        )
    return items


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _pearson_or_fallback(xs: list[float], ys: list[float]) -> float:
    """Light-weight correlation (Pearson). Returns 0.0 if degenerate."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _rank(xs: list[float]) -> list[float]:
    sorted_pairs = sorted(enumerate(xs), key=lambda p: p[1])
    ranks = [0.0] * len(xs)
    for rank, (orig, _) in enumerate(sorted_pairs, start=1):
        ranks[orig] = float(rank)
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation via Pearson on ranks."""
    return _pearson_or_fallback(_rank(xs), _rank(ys))


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Levenshtein on sequences of tokens / plan steps."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def token_cosine(a: str, b: str) -> float:
    """Bag-of-words cosine as a cheap semantic proxy."""
    va: dict[str, int] = {}
    vb: dict[str, int] = {}
    for tok in a.split():
        va[tok] = va.get(tok, 0) + 1
    for tok in b.split():
        vb[tok] = vb.get(tok, 0) + 1
    num = sum(va.get(k, 0) * vb.get(k, 0) for k in set(va) | set(vb))
    den_a = math.sqrt(sum(v * v for v in va.values()))
    den_b = math.sqrt(sum(v * v for v in vb.values()))
    if den_a == 0 or den_b == 0:
        return 0.0
    return num / (den_a * den_b)


# ---------------------------------------------------------------------------
# Mock agent
# ---------------------------------------------------------------------------


@dataclass
class MockAgent:
    """Reasonable baseline implementing the MAPCompatibleAgent protocol."""

    seed: int = 0

    def conflict_monitor(
        self, intents: Sequence[tuple[str, float]]
    ) -> float:
        if not intents:
            return 0.0
        scores = [s for _, s in intents]
        total = sum(scores) or 1.0
        probs = [s / total for s in scores]
        n = len(probs)
        if n <= 1:
            return 0.0
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        return entropy / math.log(n)

    def state_predict(self, history: Sequence[str]) -> str:
        if not history:
            return "unknown"
        last = history[-1]
        # Extract noun heuristically (last word).
        obj = last.split()[-1]
        return f"user reads {obj}"

    def evaluate_pair(self, a: str, b: str) -> str:
        return "a" if len(a) >= len(b) else "b"

    def decompose(self, goal: str) -> list[str]:
        # Minimal 4-step plan template — deliberately generic.
        return ["inspect", "plan", "execute", "verify"]

    def coordinate(self, state: dict[str, Any]) -> str:
        u = float(state.get("uncertainty", 0.0))
        if u > 0.7:
            return "reflect"
        if u > 0.4:
            return "plan"
        return "act"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@dataclass
class MAPHarness:
    """Run all 5 MAP module benches against a MAPCompatibleAgent."""

    seed: int = 42
    conflict_n: int = 50
    trajectory_n: int = 50
    judge_n: int = 50
    plan_n: int = 20
    meta_n: int = 50
    conflict_tolerance: float = 0.15

    def run_conflict(self, agent: MAPCompatibleAgent) -> dict[str, Any]:
        items = gen_conflict_prompts(self.conflict_n, seed=self.seed)
        agree = 0
        fp = 0
        start = time.perf_counter()
        for it in items:
            pred = agent.conflict_monitor(it.payload["intents"])
            ref = float(it.reference)
            if abs(pred - ref) <= self.conflict_tolerance:
                agree += 1
            elif pred > ref + self.conflict_tolerance:
                fp += 1
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        n = len(items)
        return {
            "agreement_rate": round(agree / n, 4),
            "false_positive_rate": round(fp / n, 4),
            "latency_ms": round(elapsed_ms / max(n, 1), 4),
            "n": n,
        }

    def run_state_predictor(self, agent: MAPCompatibleAgent) -> dict[str, Any]:
        items = gen_trajectory_prompts(self.trajectory_n, seed=self.seed + 1)
        sims: list[float] = []
        for it in items:
            pred = agent.state_predict(it.payload["history"])
            sims.append(token_cosine(pred, it.reference))
        n = len(items)
        mean_sim = sum(sims) / max(n, 1)
        # prediction_uncertainty = 1 - mean_sim (coarse proxy).
        return {
            "cosine_to_reference": round(mean_sim, 4),
            "prediction_uncertainty": round(1.0 - mean_sim, 4),
            "n": n,
        }

    def run_evaluator(self, agent: MAPCompatibleAgent) -> dict[str, Any]:
        items = gen_judge_prompts(self.judge_n, seed=self.seed + 2)
        # For Spearman we need scalar predicted scores; map a->1.0, b->0.0.
        preds: list[float] = []
        refs: list[float] = []
        correct = 0
        for it in items:
            a, b = it.payload["pair"]
            pred_winner = agent.evaluate_pair(a, b)
            preds.append(1.0 if pred_winner == "a" else 0.0)
            refs.append(1.0 if it.reference == "a" else 0.0)
            if pred_winner == it.reference:
                correct += 1
        n = len(items)
        rho = spearman(preds, refs)
        # Escalation proxy: fraction of ambiguous judgements (|a-b| similar).
        escalate = sum(
            1
            for it in items
            if abs(len(it.payload["pair"][0]) - len(it.payload["pair"][1]))
            < 5
        )
        return {
            "spearman_rho": round(rho, 4),
            "accuracy": round(correct / n, 4),
            "escalation_rate": round(escalate / n, 4),
            "n": n,
        }

    def run_decomposer(self, agent: MAPCompatibleAgent) -> dict[str, Any]:
        items = gen_plan_prompts(self.plan_n, seed=self.seed + 3)
        dists: list[int] = []
        exact = 0
        for it in items:
            plan = agent.decompose(it.payload["goal"])
            d = edit_distance(plan, it.reference)
            dists.append(d)
            if d == 0:
                exact += 1
        n = len(items)
        return {
            "plan_edit_distance": round(sum(dists) / max(n, 1), 4),
            "exact_match_rate": round(exact / n, 4),
            "n": n,
        }

    def run_coordinator(self, agent: MAPCompatibleAgent) -> dict[str, Any]:
        items = gen_meta_prompts(self.meta_n, seed=self.seed + 4)
        correct = 0
        start = time.perf_counter()
        for it in items:
            pred = agent.coordinate(it.payload["state"])
            if pred == it.reference:
                correct += 1
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        n = len(items)
        return {
            "meta_accuracy": round(correct / n, 4),
            "latency_ms": round(elapsed_ms / max(n, 1), 4),
            "n": n,
        }

    def run_all(self, agent: MAPCompatibleAgent) -> dict[str, Any]:
        """Run all 5 module benches and return a single JSON-ready dict."""
        return {
            "schema_version": SCHEMA_VERSION,
            "seed": self.seed,
            "agent": type(agent).__name__,
            "modules": {
                MAPModule.CONFLICT_MONITOR.value: self.run_conflict(agent),
                MAPModule.STATE_PREDICTOR.value: self.run_state_predictor(
                    agent
                ),
                MAPModule.EVALUATOR.value: self.run_evaluator(agent),
                MAPModule.DECOMPOSER.value: self.run_decomposer(agent),
                MAPModule.COORDINATOR.value: self.run_coordinator(agent),
            },
        }

    @staticmethod
    def write_json(report: dict[str, Any], path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> int:  # pragma: no cover - thin CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description="Run MAP harness on MockAgent")
    parser.add_argument("--out", default="results/map-mock.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    harness = MAPHarness(seed=args.seed)
    report = harness.run_all(MockAgent())
    path = harness.write_json(report, args.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
