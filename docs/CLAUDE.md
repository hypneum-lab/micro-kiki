# docs/ — micro-kiki documentation hub

Paper-A drafts, dated specs/plans, training methodology, hardware setup. Read-only narrative artefacts; runtime code lives in `src/`. **Companion file**: `AGENTS.md` (file-by-file index + RFC2119 conventions) — do not duplicate it here; this file scopes Claude-specific edit discipline.

## Subdirs

| Subdir | Role |
|---|---|
| `specs/` | Dated architecture decisions (ISO date prefix, RFC2119 tone, Draft / Accepted / Superseded) |
| `papers/` | Paper-B drafts: spikingkiki v3, stack-conditioning case study, related-work notes |
| `paper-a/` | Paper-A LaTeX sources (triple-hybrid: MoE + DiffAttention + Spiking) — `.tex` + figures + bib |
| `plans/` | Dated implementation plans `YYYY-MM-DD-<slug>.md` |
| `training/` | Operational training docs (3-phase MLX curriculum, forgetting-gate, e2e-smoke runbook) |
| `research/` | Frozen research notes — landscape scans, MoE-LoRA audits, sota surveys |
| `superpowers/` | Agentic-capabilities specs + plans (separate axis from runtime) |
| `blog/` | Long-form posts (non-RFC tone) |

## Hard rules (Claude-specific, additive to root + AGENTS.md)

- **Dated files are immutable in slug + date**. To revise, write a new dated spec/plan and mark the old one `Superseded by <new file>` in its header. Renaming breaks plan-from-plan citations and commit-SHA stability.
- **Pin paper outlines to a tag** (`paper-A-v0.3`, `spikingkiki-v3-final`), never `HEAD`. Cross-paper citations (e.g. paper-a → spikingkiki-v3) use the tagged filename.
- **Specs cite ADRs of dependent repos by tag**: `dream-of-kiki@C-v0.4.1+STABLE`, `nerve-wml@v0.9-draft`. Never `main`.
- **Hardware setup files** (`setup-studio-neuro.md`) belong here, not `hardware/`. `hardware/` is for board files; `docs/` is for the narrative around them.
- **Triple-hybrid scope (Paper-A)** = MoE router + Differential Attention + Spiking sub-layer; the SNN/VQC layers from `paper-outline-triple-hybrid.md` are research artefacts, **not** active runtime — see the scope-note banner at the top of that file.
- **Training pipeline split**: this repo's `training/` documents runtime + eval + curriculum config; the actual MLX training execution lives in `~/KIKI-Mac_tunner/`. Do not paste tuner internals here.
- **Research notes are frozen snapshots**: edits require a "freeze date" header bump and a rationale line. Prefer writing a new dated note that supersedes.

## Edit discipline before writing

- New spec/plan → ISO-date filename, status header (`Status: Draft|Accepted|Superseded`), one-line purpose, RFC2119 verbs.
- Touching a spec referenced by root `CLAUDE.md` or by a plan → update both citers in the same commit (or supersede + cross-link).
- Adding a paper draft → register the tag in `papers/` README or in the paper outline; CI does not catch dangling drafts.
- Cross-paper figure or table → render via `paper-a/build.sh` or `papers/build-pdf.sh`; never hand-edit a generated `.pdf`.
- Code fences are language-tagged (`bash`, `python`, `yaml`); tables over prose for configs / metrics.

## Reading order for new agents

1. Root `CLAUDE.md` (invariants: 35B-A3B base, all-17-modules LoRA, sigmoid router).
2. `AGENTS.md` (this dir) — file-by-file index, freeze dates, RFC2119 tone.
3. Most recent dated spec under `specs/` — last-write-wins for architectural decisions.
4. `STATUS` proxies: `MODEL_CARD-v*.md` (root) + `CHANGELOG.md` + `paper-a/c2-diagnostic.md` for empirical state.
5. `training/README.md` only when touching the curriculum or forgetting-gate.

## Cross-repo citations

| Target | Pin via |
|---|---|
| `dream-of-kiki` framework C | `C-vX.Y.Z+{STABLE,UNSTABLE}` tag |
| `nerve-wml` Nerve Protocol | repo tag (e.g. `v0.9`) |
| `KIKI-Mac_tunner` recipes | commit SHA in `mlx-per-domain/*.yaml` reference |
| arXiv paper | full id (`2510.13003`, not "OPLoRA paper") |

## Anti-patterns

- Renaming or deleting a dated file after the commit that introduced it (breaks `commit_sha + filename` citation).
- Citing a paper or spec by `HEAD` in another doc — use the tag or the dated filename.
- Mixing Paper-A (triple-hybrid) and Paper-B (spikingkiki) scope in one document — they are separate venue tracks.
- Editing `research/2026-04-19-*-audit.md` post-pivot — research notes are frozen; write a follow-up note instead.
- Adding training execution details (MLX trainer flags, dataset tokenization) here — those belong in `~/KIKI-Mac_tunner/`.
- Duplicating the file-by-file index from `AGENTS.md` — defer to it; this file is for invariants and discipline.
