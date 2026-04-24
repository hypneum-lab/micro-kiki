# RLM vs Negotiator (CAMP + Catfish): Comparative Analysis

**Date**: 2026-04-25
**Status**: Proposal
**Author**: electron-rare
**arXiv refs**: CAMP (2604.00085), Catfish (2505.21503), RLM (2512.24601)

---

## 1. Current Negotiator (CAMP + Catfish)

### How it works

Stage 5 of the micro-kiki 7-stage cognitive pipeline:

1. **K candidates generated** (default K=3, configurable via `negotiator_k` in `FullPipelineConfig`). All K candidates use the same prompt, same adapter, same system context.
2. **Argument extraction** (`ArgumentExtractor`): each candidate is parsed into structured claims, evidence, and reasoning chains. Quality scored 0–1.
3. **Agreement scoring**: proxy metric based on textual prefix overlap across candidates (simple heuristic, not embedding-based).
4. **Adaptive judge** (`AdaptiveJudge`): routes to `skip` (agreement > 0.9), `fast` (Qwen35B, agreement >= 0.5), or `deep` (Mistral-Large, agreement < 0.5). Returns winner index + confidence + rationale.
5. **Catfish dissent** (`CatfishModule`): triggers devil's advocate when agreement > 0.95 AND average argument quality < 0.6 (suspiciously uniform but weak consensus). If triggered, the dissent response is added as candidate K+1 and the judge re-runs with halved agreement.

### Strengths

- **Explicit quality comparison**: structured argument extraction forces the pipeline to articulate *why* a response is good, not just pattern-match.
- **Catches groupthink**: Catfish specifically targets the high-agreement-low-quality failure mode where all candidates parrot the same mediocre answer.
- **Adaptive cost**: the `skip` path at agreement > 0.9 avoids unnecessary judging for trivial queries.
- **Modular**: judge, extractor, and catfish are independently testable and replaceable.

### Weaknesses

- **3x inference cost baseline**: generating K=3 candidates from the same adapter is the dominant cost. All three see the same prompt with the same LoRA stack — diversity comes only from sampling temperature, not from different expertise.
- **No domain decomposition**: a multi-domain prompt (e.g., "design a battery-powered STM32 system with CAN bus and EMC compliance") gets the same single adapter applied to all K candidates. The router picks *one* best adapter; the negotiator never routes sub-problems to specialized stacks.
- **Agreement heuristic is shallow**: prefix-overlap is a poor proxy for semantic agreement. Two candidates can have radically different technical content but similar opening sentences.
- **Catfish is rare in practice**: the dual threshold (>0.95 agreement AND <0.6 quality) fires infrequently. Most real queries either have low agreement (triggering the judge) or high quality (no Catfish needed).
- **Sequential bottleneck**: argument extraction → judge → catfish → optional re-judge is 3–4 serial LLM calls on top of the K inference calls.

---

## 2. RLM Recursive Approach

### How it works (alexzhang13/rlm, arXiv 2512.24601)

The Recursive Language Model treats the LLM as a recursive function:

1. **Root call**: LLM receives the full prompt + system instructions that enable recursive self-calls.
2. **Decomposition**: if the problem is complex, the LLM emits code blocks that call `rlm_query(sub_prompt)` — spawning child RLM instances.
3. **Sub-calls**: each child runs in its own context, potentially decomposing further (up to `max_depth`).
4. **Composition**: parent LLM receives sub-call results and synthesizes a final answer.
5. **Iteration**: the root can iterate (up to `max_iterations`) if the composed answer is unsatisfactory.

Key parameters:
- `max_depth` (default 1): how many recursion levels are allowed.
- `max_iterations` (default 30): how many root-level iterations before returning best partial answer.
- `custom_system_prompt`: controls how the LLM decides to decompose vs. answer directly.
- `other_backends`: sub-calls can target a different model/endpoint than the root.

### Applied to negotiation (replacing Stage 5)

Instead of K parallel candidates from the same adapter, RLM would:

1. **Analyze the prompt**: identify distinct sub-problems (e.g., "STM32 power supply design" + "CAN bus protocol selection" + "EMC pre-compliance layout rules").
2. **Route each sub-problem**: use `/v1/route` to classify each sub-query, selecting the best adapter per sub-problem.
3. **Solve each sub-problem**: call micro-kiki's own `/v1/chat/completions` with the domain-specific model alias for each sub-query.
4. **Compose**: synthesize sub-answers into a coherent, multi-domain response.

This is fundamentally different from the Negotiator: instead of redundancy (K answers to the same question), it's **specialization** (1 answer per sub-problem, each from the best expert).

### Applied to routing (recursive domain classification)

The current MetaRouter (sigmoid, 35 outputs) classifies a prompt into domains in a single pass. For complex multi-topic prompts, a recursive approach would:

1. Root classifier identifies the dominant domain.
2. If confidence is split across 3+ domains above threshold, RLM decomposes the prompt into domain-specific sub-queries.
3. Each sub-query is independently classified (simpler classification, higher confidence).
4. The decomposition tree becomes the execution plan for inference.

---

## 3. Comparative Analysis

| Criterion | Negotiator (CAMP + Catfish) | RLM Recursive |
|---|---|---|
| **Inference cost** | 3x base (K=3 candidates) + 3–4 judge/extract calls | Variable: 1x for simple, 2–5x for complex (depth-dependent) |
| **Multi-domain** | No — all candidates use same adapter | Yes — sub-calls can use different adapters via `/v1/route` |
| **Explainability** | Moderate (winner idx + quality scores + judge rationale) | High (full decomposition tree, sub-problem → sub-answer mapping) |
| **Latency** | ~3x single inference (parallel candidates) + serial judge | ~2–5x single inference (sequential decomposition, parallelizable sub-calls) |
| **Quality on simple prompts** | Overkill — K=3 from same adapter rarely disagree | Equivalent to single pass (RLM answers directly, no decomposition) |
| **Quality on complex prompts** | Limited — same-adapter candidates cannot specialize | Better — sub-problem specialization matches adapter to sub-domain |
| **Diversity of candidates** | Temperature-only (same model, same adapter, same prompt) | Structural (different sub-problems, potentially different adapters) |
| **Groupthink detection** | Explicit (Catfish module) | Implicit (decomposition avoids same-prompt echo) |
| **Error recovery** | Catfish re-judge on dissent | RLM iteration loop (up to `max_iterations`) |
| **Implementation complexity** | Moderate (CAMP+Catfish code exists, ~160 LOC across 4 modules) | Low (rlm library handles recursion; integration is a system prompt + custom tools) |
| **Adapter utilization** | 1 adapter per query (router picks best single) | N adapters per query (1 per sub-problem) |

### Cost model (concrete)

For a multi-domain prompt like "Design a battery-powered STM32 system with CAN bus, EMC compliance, and a custom KiCad PCB layout":

**Negotiator (current)**:
- 3× full inference (~2048 tokens each) = 6144 output tokens
- 3× argument extraction = ~600 tokens
- 1× judge call = ~200 tokens
- Total: ~6944 tokens, 4 serial stages

**RLM recursive** (depth=1):
- 1× decomposition call (~300 tokens output)
- 4× sub-problem inference (~512 tokens each, from `stm32`, `power`, `emc`, `kicad-pcb` adapters) = 2048 tokens
- 1× composition call (~1024 tokens output)
- Total: ~3372 tokens, 3 serial stages (decompose → parallel sub-calls → compose)

**Result**: RLM uses ~49% of the tokens while providing domain-specialized sub-answers.

---

## 4. Hybrid Proposal

Not all queries benefit from recursive decomposition. The optimal approach is **conditional routing**:

### Decision tree

```
Query arrives
  │
  ├─ Router detects 0-1 active adapters (single domain / base)
  │   → Pass-through: direct inference, no negotiation, no decomposition
  │
  ├─ Router detects 2+ active adapters (multi-domain)
  │   → RLM recursive decomposition:
  │     1. Decompose into domain-specific sub-queries
  │     2. Route each sub-query to its best adapter
  │     3. Solve each sub-query independently
  │     4. Compose final answer
  │
  └─ Safety/controversial flag (future: content classifier)
      → Keep Catfish dissent as post-processing check on composed answer
```

### Why keep Catfish

Catfish solves a different problem than RLM. RLM improves answer quality through specialization; Catfish catches *adversarial consensus* — when a model confidently produces a harmful or misleading answer. These are orthogonal:

- RLM decomposition handles **complexity** (multi-domain, multi-step).
- Catfish handles **reliability** (detect when the model is confidently wrong).

Catfish should migrate from Stage 5 (Negotiator) to Stage 6 (post-composition check), running only when the composed answer triggers safety heuristics.

### What to deprecate

- **K-candidate generation**: eliminated entirely. No more 3× inference of the same prompt.
- **Argument extraction**: eliminated. RLM's decomposition tree provides structural explainability.
- **Adaptive judge**: eliminated. Composition replaces arbitration — there is no "winner" to pick.

---

## 5. Integration Plan

### Stage remapping

| Stage | Current | Proposed |
|---|---|---|
| 1 | MetaRouter (domain classification) | MetaRouter (unchanged) |
| 2 | Adapter selection + swap | Adapter selection + swap |
| 3 | Aeon recall (memory injection) | Aeon recall (unchanged) |
| 4 | MLX inference (single or K candidates) | **Conditional**: single inference OR RLM recursive decomposition |
| 5 | Negotiator (CAMP + Catfish) | **Removed** (merged into Stage 4 conditional logic) |
| 6 | AntiBias (KnowBias, RBD, DeFrame) | AntiBias + **Catfish safety check** (migrated from Stage 5) |
| 7 | Aeon write (memory persistence) | Aeon write (unchanged) |

### Implementation steps

1. **Add RLM custom tools** (`scripts/poc_rlm_negotiator.py` — this PoC):
   - `route_query(query)`: calls `/v1/route` to classify a sub-query.
   - `domain_inference(query, domain)`: calls `/v1/chat/completions` with the domain-specific model alias.

2. **Create `src/cognitive/recursive_negotiator.py`**:
   - Wraps `rlm.RLM` with micro-kiki's domain routing as custom tools.
   - System prompt instructs decomposition when 2+ domains detected.
   - Falls back to direct inference for single-domain queries.

3. **Modify `full_pipeline_server.py` Stage 4–5 logic**:
   - If `len(active_adapters) == 1`: direct inference (current path, no change).
   - If `len(active_adapters) >= 2`: invoke `RecursiveNegotiator` instead of K-candidate + CAMP.
   - Catfish dissent check moves to Stage 6 (configurable, default off for non-safety queries).

4. **Config changes** (`FullPipelineConfig`):
   - `negotiator_k` deprecated (default 1, effectively disabling K-candidate path).
   - New: `rlm_max_depth: int = 1`, `rlm_max_iterations: int = 5`.
   - New: `catfish_stage: Literal["negotiator", "antibias"] = "antibias"` (migration flag).

5. **Eval**: A/B test on the 5 test prompts (single-domain vs. multi-domain), measuring:
   - Token count (cost proxy).
   - Latency.
   - Domain coverage (does the response address all sub-domains?).
   - Quality (human eval on 1–5 scale per sub-domain).

### Risks

- **RLM library maturity**: `alexzhang13/rlm` is research code. The `completion()` call is synchronous and spawns threads for sub-calls. May need async wrapper for FastAPI integration.
- **Decomposition quality**: the LLM's ability to decompose correctly depends on the system prompt. Bad decomposition = worse than single-pass.
- **Latency**: recursive calls are sequential by nature (decompose → sub-calls → compose). Parallel sub-calls help, but the serial decompose + compose overhead is new.
- **Adapter swap cost**: each sub-call may require a different adapter. Current swap is ~10ms (unpatch) but 4 consecutive swaps in a single request is untested.
