# C2 Diagnostic — Design Spec

**Status:** Design approved 2026-04-20. Next step: writing-plans.
**Created:** 2026-04-20
**Parent context:** Phase C completed 2026-04-19 with negative C2 result (kill triggered at oracle−random=0.29 < 0.30). This spec is a post-hoc diagnostic on the existing 100 per-query records.

## Goal

Explain the C2 negative result via three targeted analyses on `results/c2-downstream.json`, producing paper-facing artefacts that Paper A §5 can cite. Scope strictly limited to **post-hoc analysis of existing data** — zero new compute, zero new LLM calls.

## Sibling sub-project (deferred)

A second sub-project — **Real LoRA adapter experiment** — was scoped in the same brainstorm but deferred. It requires 10 LoRA trainings + adapter-swap C2 rerun, 10-20h compute. Its spec is written AFTER this diagnostic, informed by the findings here.

## Purpose and data flow

Single Python script reads the existing JSON and produces three analyses plus one narrative file. No state mutation beyond writing new artefacts.

```
results/c2-downstream.json            ─► scripts/c2_diagnostic.py
                                            │
                                            ├─► results/c2-diagnostic.json         (machine-readable)
                                            ├─► docs/paper-a/c2-diagnostic-per-domain.pdf
                                            ├─► docs/paper-a/c2-diagnostic-stratified.pdf
                                            ├─► docs/paper-a/c2-diagnostic-top10.md
                                            └─► docs/paper-a/c2-diagnostic.md       (paper-facing narrative)
```

The three analyses are produced in one invocation. Each analysis is implemented as a function in the script (per-responsibility boundaries).

## Analyses

### Analysis B — Per-domain gap breakdown

**Input:** `results/c2-downstream.json` — 100 per-query records across 3 routers (vqc, random, oracle).

**Computation:** for each domain d ∈ {dsp, electronics, emc, embedded, freecad, kicad-dsl, platformio, power, spice, stm32}:

- `mean_oracle[d]` = mean of `score` over queries where `expected_domain == d` for the oracle router.
- `mean_vqc[d]` = same for vqc router.
- `mean_random[d]` = same for random router.
- `gap_oracle_vs_vqc[d]` = `mean_oracle[d] - mean_vqc[d]`
- `gap_vqc_vs_random[d]` = `mean_vqc[d] - mean_random[d]` (negative ⇒ confidently-wrong dominates in domain d)

**Output:**
- Sorted bar chart (decreasing `gap_oracle_vs_vqc`) with two sub-bars per domain (oracle-vqc, vqc-random). Saved as `c2-diagnostic-per-domain.pdf`.
- Table in `c2-diagnostic.json` under `per_domain` key.

**Interpretation guidance:**
- Gap uniform across all 10 domains → confidently-wrong pathology is systemic.
- Gap concentrated in 2-3 domains → domain-specific fragility (e.g., domains where expert-persona-prompt strongly misleads off-topic).
- If any `gap_vqc_vs_random[d] > 0`, those domains are where routing DOES help, despite aggregate negative result.

### Analysis C — Correctness stratification

**Definitions:** for each query q, compute two booleans:
- `vqc_correct(q)` = `results.vqc.per_query[q].correct_route`
- `oracle_correct(q)` = always True (oracle is always correct by construction — 100/100).

So the stratification is effectively a binary split on VQC correctness:
- Bucket A: `vqc_correct == True` (VQC routed correctly on this query)
- Bucket B: `vqc_correct == False` (VQC routed incorrectly)

For each bucket, compute:
- `vqc_mean_score_bucket`
- `oracle_mean_score_bucket` (on the same queries)
- `random_mean_score_bucket` (same queries, but random's route differed — this is the key comparison)
- Bucket size (n)

**The confidently-wrong pathology test:** in Bucket B (VQC wrong), compare `vqc_mean_score_bucket` to `random_mean_score_bucket` over the same queries. If VQC < Random in bucket B, the pathology is confirmed at stratified level (not just aggregate). If VQC ≈ Random, the pathology was an aggregation artefact.

**Output:**
- Bar chart: x-axis = {bucket A, bucket B}, bars = {vqc, random, oracle} scores per bucket. Saved as `c2-diagnostic-stratified.pdf`.
- Table in `c2-diagnostic.json` under `stratified` key with bucket sizes + per-router means.

### Analysis E — Qualitative top-10 review

**Computation:** compute `gap[q] = oracle.per_query[q].score - vqc.per_query[q].score` for each q ∈ 100. Sort descending. Take top 10 (ties broken by original order for determinism).

**Output:** `c2-diagnostic-top10.md` with one section per query:

```markdown
## Query k (gap = Δ)

**Question:** {q.question}
**Expected domain:** {q.expected_domain}

**VQC routed to:** {q.vqc.routed_domain}  (score: {q.vqc.score})

> {q.vqc.answer truncated to 500 chars}

**Oracle routed to:** {q.oracle.routed_domain}  (score: {q.oracle.score})

> {q.oracle.answer truncated to 500 chars}
```

This is **data-only** — no automated pattern detection. Human reads the 10 pairs and writes a "Patterns observed" bullet list at the end of the file, committing that with the generated content.

## Outputs (final deliverables)

| File | Format | Lines/size | Auto-generated | Hand-edited? |
|---|---|---|---|---|
| `scripts/c2_diagnostic.py` | Python CLI | ~150 lines | yes | no |
| `tests/scripts/test_c2_diagnostic.py` | pytest | ~80 lines | yes | no |
| `results/c2-diagnostic.json` | JSON | ~3 KB | yes | no |
| `docs/paper-a/c2-diagnostic-per-domain.pdf` | matplotlib | ~15 KB | yes | no |
| `docs/paper-a/c2-diagnostic-stratified.pdf` | matplotlib | ~15 KB | yes | no |
| `docs/paper-a/c2-diagnostic-top10.md` | Markdown | ~400 lines | yes + hand-edit | **yes** (Patterns section) |
| `docs/paper-a/c2-diagnostic.md` | Markdown | ~100 lines | yes + hand-edit | **yes** (Interpretation + Implications) |

## Testing

Unit tests on synthetic fixture JSONs (3-query mock data):

1. **Per-domain test:** given a mock JSON with exactly 1 query per domain, verify `mean_oracle[d]` equals the single score. Verify gap arithmetic.
2. **Stratified buckets test:** mock 4 queries: {vqc correct × 2, vqc wrong × 2}. Assert bucket A has 2 queries, bucket B has 2, and scores match the mock input.
3. **Top-10 sorting test:** mock 15 queries with known gap values. Assert top-10 returned in correct order, ties stable.
4. **Output JSON schema test:** run full pipeline on mock data, assert output has keys `per_domain`, `stratified`, `top_gaps`, `config`.

Runtime constraint: entire diagnostic must run in <10s on real data (100 queries, no LLM calls). Tested via `pytest -x tests/scripts/test_c2_diagnostic.py` which runs the unit tests only (~1s).

## Non-goals (out of scope)

- **No new LLM calls** — all data from existing `c2-downstream.json`.
- **No re-judging with different seed** — that is Analysis F from the brainstorm, explicitly skipped.
- **No self-judging artefact detection beyond per-domain/stratified aggregates** — Analysis D was skipped.
- **No automated pattern detection on top-10 answers** — Analysis E is data dump + human narrative, deliberately.
- **No update to Paper A v2 main document in the same commit** — a separate `docs(paper-a):` commit wraps the diagnostic findings into §5 after human review of the qualitative data.

## Kill criterion

If the diagnostic reveals `max(gap_oracle_vs_vqc[d]) < 0.5` AND `all(gap_vqc_vs_random[d]) ≈ 0 (within ±0.3)`, the confidently-wrong pathology is statistical noise rather than a real effect. In that case:
- Document as "negative diagnostic" in `c2-diagnostic.md`.
- Do NOT write the sibling LoRA experiment spec — the premise (pseudo-adapters weak) is not supported by the data, so real adapters are unlikely to produce dramatic change.
- Paper A §5 is updated to mark C2 as "underpowered, negative at noise floor".

## Next step

Upon user approval of this spec:
1. Invoke `superpowers:writing-plans` skill to produce the TDD implementation plan.
2. Execute the plan (sub-agent-driven or inline, user choice).
3. Write the sibling spec (Real LoRA experiment) informed by diagnostic findings.
