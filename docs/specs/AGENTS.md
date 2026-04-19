<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# docs/specs

## Purpose
Technical design specs and architectural decision records for micro_kiki. Specs are dated, single-topic, and immutable once merged — if the design changes, write a new spec with a newer date and cross-link the previous one (the 2026-04-16 architecture-pivot spec supersedes the earlier 35B choices in `2026-04-15-micro-kiki-design.md`, for example). Non-dated files (`aeonsleep-architecture.md`, `diffattn-integration.md`, etc.) are reference specs for subsystems whose design pre-dates the dating convention or is considered stable.

## Key Files

Naming convention: `YYYY-MM-DD-<topic>.md` for dated specs, `<topic>.md` for stable reference specs.

| File | Description |
|------|-------------|
| `2026-04-16-architecture-pivot-35b.md` | **Active** — pivot from Qwen3.5-4B + custom MoE-LoRA to Qwen3.5-35B-A3B + standard LoRA. Referenced from root `CLAUDE.md`. |
| `2026-04-16-reorientation-rationale.md` | **Active** — longer narrative of why the 35B pivot happened. |
| `2026-04-17-stm32h743-usb-bootloader-design.md` | STM32H743 USB bootloader design (hardware integration). |
| `2026-04-17-ethernet-bootloader-update-protocol.md` | Ethernet bootloader update protocol spec. |
| `2026-04-15-micro-kiki-design.md` | Original 35B system design (partly superseded by 04-16 pivot). |
| `2026-04-15-cognitive-layer-design.md` | Negotiator + anti-bias + memory palace design. |
| `2026-04-15-micro-kiki-v0.2-quantum-inspired.md` | v0.2 router-as-VQC proposal. |
| `2026-04-15-micro-kiki-v0.3-neuroscience.md` | v0.3 neuro-inspired design (SpikingBrain, DeltaNet). |
| `aeonsleep-architecture.md` | Aeon memory compression / replay architecture. |
| `diffattn-integration.md` | DiffAttention integration notes (Qwen fork). |
| `energy-methodology.md` | Energy-per-token benchmarking methodology. |
| `las-conversion-framework.md` | Linear-Attention Substitution conversion framework. |
| `map-paper-spec.md`, `map-validation-report.md` | MAP (memory access pattern) paper spec + validation run. |
| `mlx-lm-fork-reference.md` | Notes on the mlx-lm fork used for 3x Metal limit. |
| `spikingbrain-7b.md`, `spikingbrain-acquisition.md`, `spikingbrain-quant.md` | SpikingBrain-7B integration, weight acquisition, quantization. |

## For AI Agents

### Working In This Directory

- **Never edit a spec in place** once the ink is dry (merged). Write a new dated spec that supersedes it, and add a `Supersedes:` note in the new spec plus a forward-pointing `Superseded by:` note at the top of the old one.
- Use RFC2119 language (MUST / SHOULD / MAY) for normative requirements. Keep decision rationale ("why X, not Y") in the spec — that's the whole point.
- Cross-link: if your spec depends on an upstream paper, include the arXiv ID (e.g. `2510.13003` for OPLoRA). If it depends on another spec, link by filename.
- When the pivot-level architecture changes, update root `/home/kxkm/micro-kiki/CLAUDE.md` to point at the new spec.
- Keep individual specs focused. One topic per file. If you find yourself writing "and also...", split it.

### Testing Requirements

No executable tests. Sanity checks:
- Each dated spec MUST have a top-level `#` heading that matches the filename topic.
- Internal cross-links (`docs/specs/<other>.md`) should resolve (no dangling links).

### Common Patterns

- Front matter is plain Markdown — no YAML front-matter convention in this project.
- Filenames use lowercase-kebab-case for the topic slug.
- Tables preferred over prose for comparison matrices (model A vs B, config X vs Y).
- Code blocks should declare language (```yaml, ```python, ```bash).

## Dependencies

### Internal
- Specs are referenced from `/home/kxkm/micro-kiki/CLAUDE.md`, `docs/plans/*.md`, and code comments.
- Implementation code in `src/` and `scripts/` should cite the relevant spec in its module docstring.

### External
- arXiv papers are cited by ID only (not stored locally). Canonical papers: OPLoRA 2510.13003, Aeon 2601.15311, CAMP 2604.00085, Catfish 2505.21503, KnowBias 2601.21864, RBD 2505.17100.

<!-- MANUAL: -->
