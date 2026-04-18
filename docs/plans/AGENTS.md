<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# docs/plans

## Purpose
Multi-phase execution plans — the "how we actually ship this" documents that translate the design specs in `../specs/` into ordered work packages with deliverables, checkpoints, and rollback criteria. Plans are long-form (often 80–100 KB), authored once per major workstream, and updated in place as phases complete (checkboxes flip, not headings rewritten). The four `2026-04-15-micro-kiki-plan{1..4}-*.md` documents partition v0.1 into data / training / router / ANE tracks; `v0.2-roadmap.md` is a lightweight tracker for the next milestone.

## Key Files

Naming convention: `YYYY-MM-DD-<project>-plan<N>-<slug>.md` for phased plans, `<version>-roadmap.md` for version trackers.

| File | Description |
|------|-------------|
| `2026-04-15-micro-kiki-plan1-data-pipeline.md` | Phase 1 — 32-domain data pipeline (classify, dedupe, synthetic gen, split). ~98 KB. |
| `2026-04-15-micro-kiki-plan2-brainstacks-training.md` | Phase 2 — per-stack LoRA training, curriculum, forgetting gate. ~79 KB. |
| `2026-04-15-micro-kiki-plan3-meta-router.md` | Phase 3 — router, dispatcher, negotiator, cognitive layer, serving. ~96 KB. |
| `2026-04-15-micro-kiki-plan4-ane-pipeline.md` | Phase 4 — Apple Neural Engine hybrid pipeline (superseded by `research/ane-hybrid/` verdict: MLX pure wins). ~12 KB. |
| `v0.2-roadmap.md` | Placeholder tracker for v0.2 (temporal context + future-reasoner). Status checkboxes only. |

## For AI Agents

### Working In This Directory

- Plans are **living documents**: flip checkboxes (`- [ ]` → `- [x]`), update "Status" sections, add "Lessons learned" appendices. Don't rewrite completed phases — if the plan changes direction, add a new plan file.
- When a plan section is invalidated by an architecture pivot (e.g. plan4-ane-pipeline post-2026-04-16 MLX-wins verdict), prepend a "Superseded / status" note at the top linking the spec that caused the change.
- Plans cite specs, not the other way around: `../specs/<file>.md` is authoritative for "what", plans are authoritative for "in what order and by when".
- Keep the phase-N filename pattern when adding new plans to the same project; new projects get their own date + slug.

### Testing Requirements

No executable tests. Sanity checks:
- Each plan SHOULD have a top-level status section (checkbox list) so progress is skimmable.
- Links to specs and other plans should resolve.

### Common Patterns

- Plans use H2 for phases, H3 for work packages inside a phase.
- Each work package: goal, inputs, outputs, acceptance criteria, rollback criteria.
- Estimates use wall-clock hours, not ideal-engineer-days.
- ASCII diagrams are fine; no mermaid-only diagrams (plain text must suffice for grep).

## Dependencies

### Internal
- Plans cite `../specs/*.md` for design, and `scripts/` / `src/` paths for implementation.
- `v0.2-roadmap.md` is referenced from `/home/kxkm/micro-kiki/CLAUDE.md` implicitly (post-v0.1 milestone).

### External
- None. Plans are plain Markdown.

<!-- MANUAL: -->
