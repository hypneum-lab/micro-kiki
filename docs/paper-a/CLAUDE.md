# docs/paper-a/ — Paper-A LaTeX sources (triple-hybrid, VQC routing benchmark)

LaTeX sources for **Paper-A v2**: VQC-based domain routing benchmark with
information-theoretic upper bound and downstream-quality negative result.
Apache-2.0 release; companion code at `github.com/electron-rare/torch-vqc`.
Triple-hybrid scope per parent `docs/CLAUDE.md` = MoE router + DiffAttention +
Spiking sub-layer; SNN/VQC artefacts here are research, not active runtime.

## Layout

| File | Role |
|---|---|
| `paper-a-v2.tex` | Active draft (replaces earlier `paper-a-v1.*`). |
| `paper-preamble.tex` | Shared preamble — `\input` it, do not duplicate macros across drafts. |
| `c1-baseline-results.md`, `c2-diagnostic*.md`, `c2-downstream-results.md` | Source-of-truth result tables; numbers in `.tex` are claims under R1 — do not hand-edit. |
| `c3-corpus-validation.md` | PII-audit + cluster-taxonomy overlap (0.410 vs chance 0.100) — cited from §2 corpora. |
| `c5-info-bound.tex`, `c5-bound-figure.pdf`, `c5-references.bib` | Holevo + Fano upper bound — the 0.911 figure is **withdrawn** (erratum 2026-10-25, `docs/ERRATA.md` E-2026-10-25-02: sum of marginal MIs); figure and JSON await recomputation with the joint estimator. The bib is c5-only; merge into a global bib only at submission. |
| `*.pdf` (`c1-comparison.pdf`, `c2-*.pdf`, `c5-bound-figure.pdf`) | Generated figures — regenerate via the source notebooks; never hand-edit. |
| `build.sh` | Two-pass `pdflatex` + `bibtex` + final pass; `set -euo pipefail`. |

## Hard rules

- **Build via `build.sh`** — three pdflatex passes around bibtex are required for
  cross-refs and citation numbers. Single-pass output has stale `??` markers.
- **Numbers are R1 claims**, not prose. If `c1-baseline-results.md` says
  `0.410 ± 0.083`, the `.tex` quotes that exact value; updating one without the
  other is a bug. Re-run the source script and update both.
- **Citations to other repos** pin to a tag (`dream-of-kiki@C-v0.4.1+STABLE`,
  `nerve-wml@v0.9-draft`) per `docs/CLAUDE.md`. Never `main` / `HEAD`.
- **`paper-preamble.tex` is shared** — do not redefine `\torchvqc`, `\R`, `\E`
  macros inline in `paper-a-v2.tex`. Add them to the preamble.
- **arXiv IDs in full** in `c5-references.bib` (`2510.13003`, not "OPLoRA paper")
  per parent rule.
- **Do not commit `*.aux`, `*.log`, `*.bbl`, `*.blg`, `*.out`** — they regenerate.
  `paper-a-v2.pdf` itself is committed (release artefact).

## Scope boundary

Paper-B (`docs/papers/`, spikingkiki v3) and Paper-A (this dir) are separate venue
tracks. Do not mix scope in one document. Cross-paper citations use the tagged
filename, not `HEAD` (e.g. `paper-A-v0.3` tag). Confidently-wrong VQC pathology
(§3.3) is the headline negative result — keep it in the abstract.

## Anti-patterns

- Hand-editing a number in `paper-a-v2.tex` to match a re-run — update the source
  `.md` first, regenerate, re-quote. The R1 chain is `script → .md → .tex`.
- Renaming `paper-a-v2.tex` to `paper-a-v3.tex` for "minor edits" — version bumps
  belong to the tag (`paper-A-v0.3`), not the filename. The filename is stable.
- Forking the preamble per chapter file — `\input{paper-preamble}` once, at top.
- Adding a new bib file alongside `c5-references.bib` without merging at
  submission time — bibtex picks one `\bibliography{...}` argument.
- Editing a generated PDF directly (`c2-downstream-figure.pdf` etc.) — regenerate
  from the source notebook in `scripts/` or the analysis script that produced it.
- Mixing Paper-B (spikingkiki) results into Paper-A — separate venue, separate
  scope; cross-cite by tag instead.
