# MAP Paper — Architectural Spec for Retrospective Validation

**Status**: Phase N-I, story 1 (v0.3 neuroscience branch)
**Paper**: Nature Communications 2025, s41467-025-63804-5 — *Meta-cognitive
Architecture for Prefrontal cognition (MAP)*.
**Purpose**: distil the MAP paper into a spec concrete enough to benchmark
micro-kiki v0.2 cognitive components against, module by module.

> This document is **retrospective**: v0.2 shipped before MAP was read.
> The goal is to check whether v0.2's cognitive layer accidentally
> implements the MAP pattern (expected) and where gaps lie (for v0.4).

---

## 1. The 5 MAP modules

The MAP paper partitions prefrontal meta-cognition into five functional
modules. Each has a defined input, output, and canonical benchmark.

### 1.1 Conflict Monitor

- **Role**: detect disagreement between candidate intents for the next
  action. Biological analogue: anterior cingulate cortex (ACC).
- **Input**: set of candidate intent activations
  `A = {(intent_i, score_i)}`.
- **Output**: scalar conflict signal `c ∈ [0, 1]` + optional top-k
  conflicting pairs.
- **Formal signal**: paper uses Shannon entropy of normalised activations
  as the conflict proxy; high entropy = high conflict = route to
  deliberation. Secondary signal: gap between top-1 and top-2.
- **Canonical benchmark**: *Conflict-Match-to-Sample* — synthetic prompts
  where ground-truth has a known conflict label.
  - Metrics: agreement rate with reference labels, false-positive rate,
    latency per decision.

### 1.2 State Predictor

- **Role**: forward-model the expected outcome of each candidate action
  given current context. Biological analogue: dorsolateral PFC + HC.
- **Input**: current state `s_t`, candidate action `a_i`.
- **Output**: predicted next state `ŝ_{t+1}` (typically an embedding
  or a short natural-language description).
- **Formal signal**: prediction uncertainty `σ(ŝ_{t+1})` used to drive
  sleep / replay / consolidation of high-uncertainty trajectories.
- **Canonical benchmark**: *Trajectory-Forecast* — given partial
  episodic traces, predict the next step; score via cosine similarity
  against held-out continuations.

### 1.3 Evaluator

- **Role**: score a completed response against a criterion (quality,
  coherence, safety, relevance). Biological analogue: OFC + vmPFC.
- **Input**: response `r`, criterion `k`, context `c`.
- **Output**: scalar score `v ∈ [0, 1]` with explanation trace.
- **Formal signal**: the paper uses a pairwise-preference model so the
  evaluator also emits a Bradley-Terry-ranked list if given multiple
  responses.
- **Canonical benchmark**: *Pairwise-Judge* — MT-Bench-style pairs;
  metrics: Spearman rank correlation vs reference scores, escalation
  rate to "hard" evaluator, per-decision cost in FLOPs / judge calls.

### 1.4 Decomposer

- **Role**: break a high-level goal into ordered sub-goals. Biological
  analogue: rostral PFC + basal ganglia (hierarchical RL).
- **Input**: natural-language goal `g`.
- **Output**: sequence of sub-goals `[g_1, …, g_n]` + dependency graph.
- **Formal signal**: the paper uses HRL-style option macros with a
  termination probability per step.
- **Canonical benchmark**: *Recipe-Step-Plan* — decompose multi-step
  instructions (e.g., ALFWorld recipes) and score against reference
  plans.

### 1.5 Coordinator

- **Role**: route the current turn to the appropriate module (act vs
  plan vs reflect) and maintain meta-state.
- **Biological analogue**: fronto-parietal control network + ACC.
- **Input**: outputs of the four other modules + working-memory state.
- **Output**: a single scalar "what-to-do-next" decision
  `d ∈ {act, plan, reflect, wait}` + a confidence.
- **Canonical benchmark**: *Meta-Decision* — synthetic episodes with
  hand-labelled ideal meta-decisions; scored by classification accuracy
  and latency.

---

## 2. Harness design

A minimal metrics harness lives at `src/eval/map_harness.py`. It is
intentionally lightweight:

- **No heavy deps** — pure-Python stdlib + numpy (optional).
- **No LLM calls in the hot path** — mock agents are injected; benches
  for story-2 and story-3 supply concrete agents.
- **Deterministic** — fixed RNG seed, reproducible JSON output.

### 2.1 Public API

```python
from src.eval.map_harness import MAPHarness, MAPModule, MockAgent

harness = MAPHarness(seed=42)
agent = MockAgent()            # stand-in for v0.2 Dispatcher/Negotiator
report = harness.run_all(agent) # dict with 5 sub-reports
harness.write_json(report, "results/map-mock.json")
```

### 2.2 Per-module metrics

| Module | Metric (primary) | Metric (secondary) |
| ------ | ---------------- | ------------------ |
| ConflictMonitor | agreement_rate | false_positive_rate, latency_ms |
| StatePredictor  | cosine_to_reference | prediction_uncertainty |
| Evaluator       | spearman_rho    | escalation_rate, judge_cost |
| Decomposer      | plan_edit_distance | sub_goal_accuracy |
| Coordinator     | meta_accuracy   | latency_ms |

### 2.3 Synthetic benchmarks

Because we cannot redistribute the paper's datasets, the harness ships
with **synthetic benchmark generators** per module. They are seeded and
produce stable ground-truth labels. The generators are transparent about
being synthetic — this is declarative validation, not competitive
benchmarking.

- `gen_conflict_prompts(n=50)` → list of
  `{"intents": [...], "reference_conflict": float}` items.
- `gen_trajectory_prompts(n=50)` → list of
  `{"history": [...], "next": str}` items.
- `gen_judge_prompts(n=50)` → list of
  `{"pair": (a, b), "reference_winner": "a"|"b"}` items.
- `gen_plan_prompts(n=20)` → list of
  `{"goal": str, "reference_plan": [str, …]}` items.
- `gen_meta_prompts(n=50)` → list of
  `{"state": {...}, "reference_decision": str}` items.

### 2.4 Mock-agent interface

The `MAPModule` enum is also the set of protocol methods a concrete
agent must implement:

```python
class MAPCompatibleAgent(Protocol):
    def conflict_monitor(self, intents: list[tuple[str, float]]) -> float: ...
    def state_predict(self, history: list[str]) -> str: ...
    def evaluate_pair(self, a: str, b: str) -> str: ...
    def decompose(self, goal: str) -> list[str]: ...
    def coordinate(self, state: dict) -> str: ...
```

A `MockAgent` implementing random-but-plausible answers is included so
the harness can be tested end-to-end without any v0.2 code.

### 2.5 JSON report schema

```jsonc
{
  "schema_version": "map-harness/1.0",
  "seed": 42,
  "generated_at": "2026-04-15T...",
  "agent": "MockAgent",
  "modules": {
    "conflict_monitor": {"agreement_rate": 0.72, "false_positive_rate": 0.11,
                           "latency_ms": 0.04, "n": 50},
    "state_predictor":  {"cosine_to_reference": 0.41, "n": 50},
    "evaluator":        {"spearman_rho": 0.33, "escalation_rate": 0.14,
                           "n": 50},
    "decomposer":       {"plan_edit_distance": 2.8, "n": 20},
    "coordinator":      {"meta_accuracy": 0.48, "latency_ms": 0.02, "n": 50}
  }
}
```

---

## 3. Non-goals

- This harness is **not** a LLM benchmark. It cannot decide whether
  v0.2 or MAP "wins" — it can only say whether v0.2 covers the same
  architectural surface.
- Numbers produced on synthetic data are **not** publishable as MAP
  replication; they are internal design-validation signals.

## 4. References

- MAP paper (Nature Communications 2025, s41467-025-63804-5).
- v0.2 cognitive-layer design: `docs/specs/2026-04-15-cognitive-layer-design.md`.
- CAMP arbitration: arxiv 2604.00085.
- Catfish dissent: arxiv 2505.21503.
- Aeon memory: arxiv 2601.15311.
