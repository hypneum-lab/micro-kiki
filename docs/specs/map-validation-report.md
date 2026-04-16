# MAP Validation Report — v0.2 as a MAP Implementation

**Status**: Phase N-I, story 4 (v0.3 neuroscience branch)
**Scope**: retrospective, spec-level mapping between micro-kiki v0.2's
cognitive layer and the 5 modules described in the MAP paper
(Nature Communications 2025, s41467-025-63804-5).
**Input artefacts**:

- `docs/specs/map-paper-spec.md` (MAP paper distilled).
- `docs/specs/2026-04-15-cognitive-layer-design.md` (v0.2 cognitive
  layer spec).
- `src/eval/map_harness.py` (story-1 harness).
- `results/map-dispatcher.json` (story-2 numeric output).
- `results/map-negotiator.json` (story-3 numeric output).

> **Methodology caveat — read this first.** None of the v0.2 cognitive
> modules are implemented in code on `main` yet. The stories 2 and 3
> benches run a *conceptual model* of each module derived verbatim from
> the v0.2 spec (CAMP as "longer candidate", Catfish as deterministic
> 10% flip, Dispatcher as entropy after 4-stack cap). The numeric
> outputs below are therefore **design-validation signals**, not
> empirical benchmarks. Treat them as consistency checks on the v0.2
> spec's intent, not as evidence about a running system.

---

## 1. Executive summary

The v0.2 cognitive layer covers **all 5 MAP modules** at the
architectural level. Three of the five (conflict monitor, evaluator,
state predictor) have direct counterparts; the remaining two
(decomposer, coordinator) are implemented *implicitly* through
combinations of Dispatcher + Negotiator + Aeon, rather than as
standalone components. This matches the "explicit specialists, implicit
generalists" pattern the MAP authors describe for compact systems.

**Conclusion.** v0.2 is a MAP-compatible architecture. There is one
notable gap — no explicit decomposer — which is logged as a v0.4
candidate. All other gaps are metadata (e.g., MAP names a surface the
spec already implements but under a different name).

---

## 2. Module-by-module mapping

### 2.1 MAP Conflict Monitor ↔ v0.2 Dispatcher

- **MAP surface**: scalar conflict `c ∈ [0, 1]` over candidate intents,
  biologically anchored in anterior cingulate cortex.
- **v0.2 component**: Dispatcher (cognitive spec §1).
- **Mechanism**: the Dispatcher multiplies 32-dim router activations by
  a 32×7 YAML matrix and returns the top meta-intent. The
  **normalised Shannon entropy** of the resulting 7-dim distribution is
  a natural conflict signal; when two meta-intents are both active
  above the "chat floor" (0.20), the Dispatcher surfaces a flag used
  by the Negotiator to pick a heavier judge.
- **Additional policies baked in**:
  - `activation_cap = 4` (max 4 active stacks). Applied before entropy
    so the conflict signal describes the set the model will actually
    run, not the full router output.
  - `chat_floor = 0.20`. If the top score is below the floor, the
    Dispatcher bumps the reported conflict by +0.20 (clipped to 1.0).
    This implements the "default to chat when nothing is confident"
    behaviour specified in §1 of the cognitive spec.
- **Story-2 numeric snapshot** (`results/map-dispatcher.json`):

  ```json
  {
    "agent": "V02DispatcherAgent",
    "agreement_rate": 1.0,
    "false_positive_rate": 0.0,
    "latency_ms": 0.0008,
    "agreement_delta_vs_mock": 0.0,
    "n": 50
  }
  ```

  On the synthetic bench, the Dispatcher's entropy signal matches the
  MockAgent baseline exactly (delta 0.0). This is expected because the
  synthetic generator uses the same entropy formula as the reference;
  the bench is therefore a **consistency test**, not a discrimination
  test. Confirms: the Dispatcher spec is internally consistent and
  aligns with MAP's conflict-monitor definition.

- **Gap**: none at this level. The Dispatcher covers conflict
  monitoring without training, which is desirable per the v0.2 "zero
  moving parts" design philosophy.

### 2.2 MAP Evaluator ↔ v0.2 Negotiator

- **MAP surface**: pairwise evaluator returning a winner + confidence,
  with ranking correlation vs reference, an escalation rate, and a
  judge-cost footprint.
- **v0.2 component**: Negotiator (cognitive spec §3). Runs CAMP
  arbitration + Catfish dissent + adaptive judge.
- **Mechanism**:
  - **CAMP** (arxiv 2604.00085). Evidence-based winner pick; in the
    bench modelled as "longer candidate wins" because length is a
    reasonable evidence proxy for generic prompts.
  - **Catfish** (arxiv 2505.21503). Structured dissent pass that flips
    the winner with probability 0.10 to break groupthink. Modelled
    deterministically so the bench is reproducible.
  - **Adaptive judge**. Fast Qwen3.5-35B-A3B by default; escalate to
    Mistral-Large-Opus when judge confidence < 0.5 (length-gap proxy
    clamped to [0, 1]) or when dispatcher says `reasoning` / `security`.
- **Story-3 numeric snapshot** (`results/map-negotiator.json`):

  ```json
  {
    "agent": "V02NegotiatorAgent",
    "spearman_rho": 0.8587,
    "accuracy": 0.92,
    "escalation_rate": 0.0,
    "judge_cost_mean": 1.0,
    "dissent_rate": 0.08,
    "spearman_delta_vs_mock": -0.1413,
    "n": 50
  }
  ```

  Interpretation:

  - **spearman_rho 0.86** vs a MockAgent that scores 1.0 on the
    synthetic set: the mock outperforms because the synthetic
    generator hard-codes "longer = correct" (no noise), whereas the
    Negotiator adds Catfish dissent, which intentionally adds 10%
    noise. The delta (−0.14) is therefore a **feature**, not a
    regression: it quantifies the cost of structured dissent against
    a noiseless ranker.
  - **escalation_rate 0.0** on this bench — the synthetic pairs always
    have a clear length gap, so the judge confidence threshold is
    never triggered. The escalation machinery is exercised in unit
    tests (`test_negotiator_escalation_cost_increases_on_close_pairs`).
  - **judge_cost_mean 1.0** in Qwen35B-equivalent units: the fast judge
    alone suffices on clear prompts. Deep-judge cost is 7.0 units per
    escalation; an escalation rate of 0.05 in production would raise
    mean cost to ~1.35, matching the v0.2 "< 1.5x fast-only cost"
    design envelope.
  - **dissent_rate 0.08** — empirical Catfish flips on n=50. Matches
    the 0.10 design target within sample variance.

- **Gap**: Catfish dissent is modelled but its *effectiveness* is not
  evaluated here. The synthetic bench rewards the deterministic ranker
  (no groupthink to break). A future bench with adversarially crafted
  agreement-on-wrong-answer pairs would measure Catfish's actual lift.

### 2.3 MAP State Predictor ↔ v0.2 Aeon (memory palace)

- **MAP surface**: forward model `ŝ_{t+1} = f(s_t, a)` plus prediction
  uncertainty.
- **v0.2 component**: Aeon (cognitive spec §2) — Atlas SIMD index +
  Trace neuro-symbolic graph.
- **Mechanism**: the Trace graph stores typed edges (temporal, causal,
  topical) between episodes. "Predict the next state" in this system
  is the graph query *"walk from `s_t` along causal/temporal edges and
  return the nearest neighbour"*. Uncertainty = distance between the
  query embedding and the top-k retrieved episodes in Atlas.
- **Gap — real**: the forward-model behaviour is *recall-based*, not
  *generative*. Aeon does not synthesise a novel `ŝ_{t+1}` when none is
  in memory; it returns the nearest known continuation. MAP's state
  predictor in the paper is a generative model. For v0.3 this gap is
  closed by AeonSleep's consolidation module (story-10), which
  summarises clusters into abstract episodes that can serve as priors
  when recall misses.

### 2.4 MAP Decomposer ↔ v0.2 (gap — implicit only)

- **MAP surface**: goal → ordered sub-goals with dependency graph.
- **v0.2 component**: **none explicit**. The closest analogue is the
  32-stack activation pattern itself: when the router activates
  multiple domain stacks (e.g., `embedded` + `debug_systems`), the
  implicit plan is "run each stack, aggregate answers via Negotiator".
  This is decomposition via specialisation, not via sub-goal
  sequencing.
- **Gap — logged for v0.4**: no explicit plan generator. Candidate
  designs:
  1. Add a lightweight planner adapter (a 33rd LoRA stack trained on
     recipe/TODO data) gated on dispatcher = `system` or `reasoning`.
  2. Use the teacher LLM (Qwen3.5-35B-A3B) for one-shot plan emission
     when the user prompt matches a plan pattern, cached in Aeon.
  3. Reuse Aeon's Trace graph with a new edge type `step_of` to
     represent plans as traversable sub-goal chains.
- **Severity**: medium. v0.2 ships without a planner and still
  produces correct output on single-step prompts; complex multi-step
  prompts degrade to "answer from the most-activated stack" rather
  than a structured plan.

### 2.5 MAP Coordinator ↔ v0.2 (implicit via Dispatcher → Negotiator)

- **MAP surface**: meta-decision {act, plan, reflect, wait} per turn.
- **v0.2 component**: **no single module**. Implicitly emergent from
  Dispatcher (decides which stacks and judge) + Negotiator (decides
  how much deliberation) + Aeon (decides how much recall). The
  meta-state lives in the dispatcher's conflict signal and the
  negotiator's escalation decision.
- **Gap — logged for v0.4**: the meta-decision surface is not
  introspectable. A user cannot query "was this turn act or reflect?"
  — only the downstream effects (stacks fired, judge called,
  memory hit) are observable. Adding a tiny coordinator that consumes
  Dispatcher conflict + Negotiator confidence + Aeon recall quality
  and emits a single meta-decision label would make the system
  auditable without changing behaviour. This would be a one-afternoon
  task and is strongly recommended for v0.4.

---

## 3. Aggregate scorecard

| MAP module       | v0.2 implementation      | Coverage | Notes                              |
| ---------------- | ------------------------ | -------- | ---------------------------------- |
| Conflict monitor | Dispatcher (explicit)    | Full     | Entropy + chat floor               |
| State predictor  | Aeon (explicit, recall)  | Partial  | No generative forward model        |
| Evaluator        | Negotiator (explicit)    | Full     | CAMP + Catfish + adaptive judge    |
| Decomposer       | Stack activation pattern | Implicit | No explicit planner                |
| Coordinator      | Dispatcher + Negotiator  | Implicit | Not introspectable                 |

**Score 3/5 explicit + 2/5 implicit = MAP-compatible architecture.**
The 2 implicit modules are acceptable in a compact system but are the
right ones to log for v0.4.

---

## 4. Outstanding gaps and v0.4 candidates

1. **Generative state predictor** — close via AeonSleep consolidation
   (story-10 on v0.3). Status: planned.
2. **Explicit decomposer** — new v0.4 proposal. Minimal viable = one
   LoRA stack + Aeon `step_of` edge type.
3. **Explicit coordinator** — new v0.4 proposal. Minimal viable =
   10-line aggregator returning `{act, plan, reflect}` label.
4. **Catfish effectiveness bench** — adversarial pairs to measure
   the lift from structured dissent against agreement-on-wrong.

## 5. Conclusion

The cognitive layer specified in v0.2 is **architecturally MAP-
compatible**: 3 of 5 MAP modules have direct v0.2 counterparts and the
remaining 2 emerge implicitly. The story-2/3 numeric outputs confirm
internal spec consistency (Dispatcher entropy behaves as MAP conflict
monitor; Negotiator Spearman tracks reference ranking with the
expected dissent penalty). No v0.2 design decision needs to be
reversed based on MAP.

**Outcome**: option (a) from the plan — "v0.2 is a MAP-compatible
architecture". Gaps itemised above feed into v0.4 planning; none
block v0.3's focus on AeonSleep, SpikingBrain, and neuromorphic edge.

## 6. Appendix A — Synthetic bench design rationale

The story-2 and story-3 benches use synthetic prompts rather than
borrowing the MAP paper's datasets. Rationale:

1. **Licensing.** The MAP paper's dataset licensing is unclear for
   redistribution; embedding it in a fork would be a compliance
   hazard. Synthetic generators produce the same measurement surface
   without that risk.
2. **Reproducibility.** Generators are seeded and have no external
   dependencies (stdlib only). Anyone running
   `python -m src.eval.map_harness` on a fresh clone gets
   bit-identical JSON output for a given seed.
3. **Transparency.** Because the generators are 30-line Python
   functions, readers can inspect exactly what "conflict reference"
   or "pair reference winner" means in this report. That visibility
   is impossible with an opaque downloaded benchmark.
4. **Scope-appropriate.** Story-2 and story-3 are descriptive
   checks, not competitive benchmarks; synthetic data is sufficient
   to surface design inconsistencies (e.g., if the Dispatcher's
   entropy formula disagreed with MAP's, we would see it).

The trade-off is that the numeric outputs cannot be compared to the
MAP paper's headline numbers. That is acceptable because the goal of
the phase is spec validation, not replication.

## 7. Appendix B — Story-2 bench walkthrough

Step-by-step trace for one synthetic conflict prompt:

1. Generator emits `intents = [("code", 0.95), ("chat", 0.02),
   ("knowledge", 0.02), ("creative", 0.01)]`, `reference = 0.11`
   (low conflict, one-hot).
2. Harness calls `V02DispatcherAgent.conflict_monitor(intents)`.
3. Agent sorts by score, truncates to top 4 (here already 4 entries).
4. Computes total = 1.0, probabilities = the same list, Shannon
   entropy ≈ 0.26 nats, normalised by ln(4) ≈ 1.39 → `c ≈ 0.19`.
5. Top score 0.95 > chat_floor 0.20, so no bump.
6. Harness compares `|0.19 - 0.11| = 0.08 ≤ tolerance 0.15` → agree.

Step-by-step trace for one conflicted prompt:

1. Generator emits near-uniform `intents = [("code", 0.48),
   ("chat", 0.52), ("system", 0.50)]`, `reference ≈ 1.00` (flat).
2. Agent truncation leaves the same 3 entries; probabilities ≈
   [0.32, 0.35, 0.33]; entropy ≈ 1.09 nats; normalised by ln(3) ≈
   0.99.
3. Top score 0.52 > 0.20, no bump. Pred ≈ 0.99, ref 1.00, agree.

The bench therefore succeeds whenever the Dispatcher's entropy
follows the synthetic reference. This is the expected "internal
consistency" behaviour; divergence would indicate a specification
bug, not a quality deficit.

## 8. Appendix C — Story-3 bench walkthrough

Step-by-step trace for one clear judgement pair:

1. Generator emits
   `a = "short helpful answer with evidence."` (37 chars)
   `b = "irrelevant"` (10 chars)
   `reference_winner = "a"`.
2. Harness calls `V02NegotiatorAgent.evaluate_pair(a, b)`.
3. Agent: `la=37, lb=10` → baseline winner "a". Dissent draw
   increments counter; deterministic 0.1 probability ⇒ no flip on
   this item. Winner emitted: "a".
4. Harness records correct pick. Judge confidence
   `|37-10|/37 ≈ 0.73` > threshold 0.50 → no escalation, cost = 1.0.

Step-by-step trace for a flipped pair (dissent trigger on item #5):

1. Same `a` / `b` as above, winner = "a" by baseline.
2. Dissent counter hits a frac < 0.10 sample → flip. Winner
   emitted: "b".
3. Harness records incorrect pick; the Catfish flip lost a point on
   this item. Spearman rho drops accordingly.

This is the mechanism behind the −0.14 `spearman_delta_vs_mock`: the
Negotiator intentionally accepts ranking noise as the price of
groupthink resistance.

## 9. Appendix D — How to reproduce

```bash
# Clone the neuroscience branch.
git clone -b neuroscience git@github.com:electron-rare/micro-kiki.git
cd micro-kiki

# Run story-1 harness on the mock agent (sanity).
uv run python -m src.eval.map_harness --out results/map-mock.json

# Run story-2 dispatcher bench.
uv run python -m src.eval.map_dispatcher_bench \
    --out results/map-dispatcher.json --n 50 --seed 42

# Run story-3 negotiator bench.
uv run python -m src.eval.map_negotiator_bench \
    --out results/map-negotiator.json --n 50 --seed 42

# Run the test suite to verify no drift.
uv run pytest -q
```

All three JSON files are deterministic for a given seed; diffing a
fresh run against the committed artefacts is a valid regression check
for any future refactor of the harness.

## 10. Appendix E — Cross-branch invariants

The v0.2 cognitive-layer spec referenced throughout this report lives
on `main`, not on `neuroscience`. This branch deliberately does **not**
modify `main` or any v0.2 source (per the cousin-fork rule in
`.ralph-v0.3/guardrails.md`). The mapping established here is
therefore one-directional: future changes on `main` could in principle
invalidate the mapping; if that happens, this report should be
re-run against the updated spec rather than the mapping being copied
back to `main`.

Concretely, the invariants this report depends on are:

- 32 domain stacks + sigmoid meta-router (base architecture).
- 7 meta-intents in `configs/meta_intents.yaml` (Dispatcher).
- CAMP + Catfish + adaptive judge as the Negotiator's three pillars.
- Aeon = Atlas SIMD + Trace graph with typed edges.

If any of these four changes on `main`, stories 2 and 3 of v0.3 need
to be re-run and section 2 of this report updated.

## 11. References

- MAP paper (Nature Communications 2025, s41467-025-63804-5).
- MAP spec: `docs/specs/map-paper-spec.md`.
- Cognitive-layer spec: `docs/specs/2026-04-15-cognitive-layer-design.md`.
- CAMP: arxiv 2604.00085.
- Catfish Agent: arxiv 2505.21503.
- Aeon memory: arxiv 2601.15311.
- Story-2 results: `results/map-dispatcher.json`.
- Story-3 results: `results/map-negotiator.json`.
