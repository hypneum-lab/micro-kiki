# src/cognitive/ — negotiator (CAMP+Catfish), anti-bias, forgetting gate, sleep tagger

Read-only cognitive layer wrapped around the model + adapters. Reads model outputs,
embeddings, episodic state from `src/memory/`; **never** mutates LoRA weights, base
weights, or the adapter registry. Aeon (long-term episodic memory) lives in
`src/memory/`, not here — this tree only consumes it.

## Modules (3 axes)

| Axis | Modules | Role |
|---|---|---|
| Negotiation (CAMP) | `negotiator.py`, `argument_extractor.py`, `judge.py`, `catfish.py` | K candidates → arguments → adaptive judge → optional Catfish dissent (arxiv 2505.21503) → winner. |
| Anti-bias (KnowBias / RBD / DeFrame) | `rbd.py`, `antibias.py`, `bias_probe.py` | Per-response RBD detector (arxiv 2505.17100) → DeFrame rewrite when biased. KnowBias neuron probe is a measurement tool, not in the hot path. |
| Sleep / consolidation | `sleep_tagger.py`, `consolidation.py`, `forgetting_gate.py` | SleepGate temporal conflict tagger (arxiv 2603.14517) → consolidation → small numpy MLP P(keep) per episode. |

## Coupling

| Direction | Contract |
|---|---|
| ← `src/serving/` | `full_pipeline_server.py` calls negotiator after K-sample inference, anti-bias after winner pick, Aeon write last. |
| ↔ `src/memory/` | `sleep_tagger` consumes `Aeon.recall`/`AeonSleep.write` snapshots; `consolidation` reads episode store; `forgetting_gate` reads `(age, access_count, conflict, emb_norm)` features. |
| ← `src/eval/forgetting.py` | Stack-level forgetting gate (angle math, win-rate). Distinct from per-episode `forgetting_gate.py` here — do not conflate. |
| External LLM | `judge.AdaptiveJudge` fast-passes with Qwen3.5-35B local, escalates to Mistral-Large-Opus only if confidence < 0.5 (see `src/eval/CLAUDE.md`). |

## Negotiator wiring

- `Negotiator` takes (`extractor`, `judge`, `catfish`) — pure DI, no globals.
- Catfish triggers iff `agreement > 0.95 AND avg_argument_quality < 0.6`. Both
  thresholds belong to `CatfishModule.__init__`, never hardcoded in callers.
- `judge_result.confidence < 0.5` → escalate. Don't add a third tier.

## Anti-bias wiring

- RBD runs on **every** response (per root CLAUDE.md and `src/eval/CLAUDE.md`); it is
  not opt-in.
- DeFrame uses `DEFRAME_PROMPT` template — don't fork it per call site, parametrise the
  `bias_type` field instead.
- `AntiBiasPipeline` logs `PipelineDecision` per call — the log is the audit trail; do
  not silence it under a flag.

## Sleep / forgetting

- `forgetting_gate.py` is a **numpy** MLP (no torch, no sklearn) — keep it that way so
  it runs on CI and the GrosMac R&D box. Features are fixed: `("age", "access_count",
  "conflict", "emb_norm")` — adding a feature invalidates trained weights.
- `sleep_tagger.py` must stay in the hot path of `AeonSleep.write`, so no LLM call —
  cosine + lexical rules + staleness only. Reason codes are `topic | contradiction |
  stale | none`; downstream `consolidation` may then call a teacher LLM.

## Anti-patterns

- Mutating `model`, `tokenizer`, or any LoRA weight from this tree — cognitive layer is
  read-only on the model surface (per root invariant: adapters are runtime-swappable,
  never merged).
- Folding `Aeon` / `CAMP` / `KnowBias` into a single `CognitiveManager` god-class —
  three independent axes, three injection points. They compose via the negotiator/
  anti-bias/full-pipeline glue, not via inheritance.
- Module-level caches (e.g. memoising `JudgeResult` by prompt hash) — sleep/consolidation
  state is per-instance, and shared mutable state breaks `tests/cognitive/` isolation.
- Calling a teacher LLM from `sleep_tagger.py` to "improve" tagging — this gates the
  hot-path latency budget; downstream `consolidation` is the place for LLM calls.
- Adding a new bias detector that bypasses RBD — it must compose with `AntiBiasPipeline`
  so the per-response audit log is preserved.
