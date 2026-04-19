<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# cognitive

## Purpose
Cognitive layer that sits around inference: multi-candidate arbitration (CAMP negotiator + Catfish dissent), bias hygiene (KnowBias neuron probing + RBD post-inference detector + DeFrame rewrite), and the AeonSleep memory maintenance primitives (conflict tagger, MLP forgetting gate, clustering consolidator). These modules are invoked by `src.routing.hybrid_pipeline`, by `src.memory.aeonsleep.AeonSleep`, and by `src.serving.aeon_hook` to keep answers grounded, diverse, and free of drift.

## Key Files
| File | Description |
|------|-------------|
| `negotiator.py` | `Negotiator.negotiate(prompt, candidates)` — runs argument extraction, computes agreement, calls `AdaptiveJudge`, then `CatfishModule.maybe_dissent`. Returns `NegotiationResult` with winner idx, judge metadata, catfish metadata. |
| `argument_extractor.py` | `ArgumentExtractor` — async structured extraction into `Argument(claim, evidence, reasoning, quality_score)` via JSON prompt. Quality score = 0.4·evidence + 0.4·reasoning + 0.2·claim (bounded to 1.0). |
| `judge.py` | `AdaptiveJudge` — CAMP arbitration (arxiv 2604.00085). Routes to `skip` (>0.9 agreement), `fast` (Qwen35B), or `deep` (Mistral-Large) based on agreement score. Returns `JudgeResult`. |
| `catfish.py` | `CatfishModule` — devil's advocate injection (arxiv 2505.21503). Triggers when `agreement_score > 0.95` AND `avg_argument_quality < 0.6`. Produces `CatfishResult`. |
| `rbd.py` | `ReasoningBiasDetector` — post-inference bias pass (arxiv 2505.17100). Flags confirmation / stereotyping / framing / authority / anchoring biases. Returns `BiasDetection`. |
| `antibias.py` | `AntiBiasOrchestrator` (single-shot RBD + DeFrame rewrite) and `AntiBiasPipeline` (orchestrator + structured decision log, flush to `results/antibias-decisions.json`). |
| `bias_probe.py` | KnowBias (arxiv 2601.21864) — `probe_bias_neurons(model, tokenizer, bias_pairs, top_n)` runs forward hooks on all layers, diffs activations on biased vs fair prompts, returns ranked `BiasNeuron` list. Requires torch. |
| `sleep_tagger.py` | `SleepTagger.tag(incoming, recent)` — embedding-cosine + negation-flip + antonym-pair + numeric-conflict rules. Returns `Tag(level, reason ∈ {topic, contradiction, stale, none}, ref_id)`. Zero-LLM hot path for `AeonSleep.write`. |
| `forgetting_gate.py` | `ForgettingGate` — tiny numpy MLP (4→16→16→1) predicting P(keep). Features: log1p(age_h)/7, log1p(access)/3, conflict_level, emb_norm. Mini-batch SGD + BCE. `generate_synthetic_pairs` and `read_jsonl` / `write_jsonl` helpers. |
| `consolidation.py` | `Consolidator.consolidate(episodes)` — single-link cosine clustering with temporal window gate + topic-strict matching. Default `heuristic_summary` (offline, deterministic extractive). Pluggable `summarize_fn` — Qwen3.5-35B backend documented as TODO. `recall_via_summary` utility for the PI-depth probe. |

## For AI Agents

### Working In This Directory
- `negotiator.py`, `judge.py`, `catfish.py`, `rbd.py`, `antibias.py`, `argument_extractor.py` all take an injectable async `generate_fn` or client — never hardcode a model endpoint. Keep them network-free at import time.
- `sleep_tagger.py`, `forgetting_gate.py`, `consolidation.py` form the sleep-cycle trio consumed by `src.memory.aeonsleep.AeonSleep`. Do **not** introduce LLM calls in `SleepTagger` — it sits on the hot write path; LLM summarisation goes through the `Consolidator.summarize_fn` injection.
- `ForgettingGate` must stay pure numpy — no torch, no scikit. The `MLPParams.save/load` format (.npz with keys w1/b1/w2/b2/w3/b3) is the on-disk contract.
- `CatfishModule` thresholds (0.95 agreement, 0.6 quality) are load-bearing — bumping them changes acceptance-test behaviour.
- `ReasoningBiasDetector` requires `confidence >= 0.5` before `AntiBiasOrchestrator` triggers DeFrame rewrite (see `antibias.py:55`).
- Project rule: LoRA touches q/k/v/o attention only — `bias_probe.py` hooks every `model.model.layers[i]`, which is fine because it is read-only; do not turn it into a training helper.

### Testing Requirements
- `tests/cognitive/test_negotiator.py` covers the full Negotiator flow with async stub clients.
- `tests/test_antibias_pipeline.py` — RBD+DeFrame pipeline, decision logging, stats.
- `tests/test_sleep_tagger.py` — tagger rules on planted contradictions.
- `tests/test_consolidation.py` — clustering + PI-depth `recall_via_summary` probe.
- `tests/test_forgetting_gate.py` — synthetic BCE training reaches F1 > threshold.

### Common Patterns
- `@dataclass(frozen=True)` for all result types (`JudgeResult`, `CatfishResult`, `BiasDetection`, `AntiBiasResult`, `PipelineDecision`, `NegotiationResult`).
- Async-first: every external-facing method (`extract`, `detect`, `maybe_dissent`, `judge`, `negotiate`, `check_and_fix`, `process`) is a coroutine.
- `from __future__ import annotations` in every module (matches `src/CLAUDE.md`).
- Graceful degradation: `generate_fn=None` always returns a safe default (`biased=False`, `quality_score=0.1`, etc.) instead of raising.
- `logging.getLogger(__name__)` — no `print()`.

## Dependencies

### Internal
- `src.cognitive.argument_extractor`, `src.cognitive.judge`, `src.cognitive.catfish` composed by `src.cognitive.negotiator`.
- `src.cognitive.rbd` composed by `src.cognitive.antibias`.
- `src.cognitive.sleep_tagger`, `src.cognitive.forgetting_gate`, `src.cognitive.consolidation` consumed by `src.memory.aeonsleep`.
- Consumed upstream by `src.routing.hybrid_pipeline.HybridPipeline` (optional negotiator wiring).

### External
- `numpy` (forgetting_gate MLP, consolidation math).
- `torch` (bias_probe only — optional, required for model probing).
- stdlib: `json`, `logging`, `dataclasses`, `datetime`, `pathlib`, `re`, `math`, `collections.Counter`.

<!-- MANUAL: -->
