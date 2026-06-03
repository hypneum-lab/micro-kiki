# research/ — micro-kiki R&D pre-spec

Exploratory code that is **not** part of the shipping `src/` package. Lives here until promoted to `src/` (with spec + tests) or frozen as a `docs/specs/` decision. **Companion file**: `AGENTS.md` (file-by-file index, dependency notes) — do not duplicate it; this file scopes Claude-specific edit + promotion discipline.

> Distinct from `docs/research/`: that dir holds **frozen narrative notes** (landscape scans, audits). This dir holds **runnable R&D scripts**.

## Subdirs

| Subdir | Track | Status | Owner CLAUDE.md |
|---|---|---|---|
| `ane-hybrid/` | ANE+Metal hybrid for Qwen DeltaNet | conclusion recorded (MLX wins) | `ane-hybrid/CLAUDE.md` |

Currently single-track. New investigation = new sibling subdir + its own `CLAUDE.md`.

## Promotion path

R&D code lifecycle (do not skip steps):

1. **Land here** — exploratory script, no API contract, no tests required, ad-hoc numerical diff vs reference is enough.
2. **Stabilize** — once a track produces a usable artefact, write a dated `docs/specs/YYYY-MM-DD-<slug>.md` (Status: Draft, RFC2119 tone) describing the API and invariants.
3. **Promote to `src/`** — only after the spec is Accepted. Add `tests/` coverage, register in the relevant `src/<area>/CLAUDE.md`. Either delete the `research/<track>/` or rename to `research/_frozen/<track>/` as ablation reference.

A track that produces a **negative result** (e.g. `ane-hybrid` verdict: MLX > ANE) does not promote — it freezes in place with the verdict logged in its `CLAUDE.md`.

## Discipline (Claude-specific)

- New track → new subdir + its own `CLAUDE.md` (recipes, anti-patterns) + an entry in `AGENTS.md`.
- `research/` may import from `src/` read-only (load weights, reuse a tokenizer); **never write back** into `src/` state.
- `src/` must **never import from** `research/` — that is scope creep; promote first.
- No CI gating on `research/` scripts. `pytest` does not collect this tree.
- Touching a frozen track (e.g. `ane-hybrid/` post-verdict) → bump the verdict header with a new dated rationale, do not silently rewrite results.

## Anti-patterns

- Importing `research/<anything>` from `src/` or `scripts/` (= scope creep without spec).
- Editing `ane-hybrid/CLAUDE.md` recipes without re-running the phase scripts and updating the throughput table.
- Promoting a track to `src/` without first writing `docs/specs/<dated>.md`.
- Mixing two investigation tracks in one subdir — fork a new subdir instead.
- Pasting MLX **training** logic here — training execution lives in `~/KIKI-Mac_tunner/`; `research/` is for inference / architecture exploration only.
- Deleting a frozen track to "clean up" — negative results are reproducibility artefacts; keep them.
