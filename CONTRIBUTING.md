# Contributing to micro-kiki

micro-kiki is a LoRA/MLX deployment artefact (35 domain adapters on a
Qwen3.6-35B-A3B base, a router and a serving pipeline), developed by
Hypneum Lab. Claims about it are held to a research standard: a number
exists only if a committed artefact under `results/` reproduces it.
Read `CLAUDE.md` (hard invariants) and `AGENTS.md` before a PR.

## Set up

```bash
git clone https://github.com/hypneum-lab/micro-kiki.git
cd micro-kiki
uv sync --all-extras          # Python >= 3.11
```

Training and large-model serving need Apple Silicon (MLX, Mac Studio
class) or a CUDA host; you do not need either to contribute to
configs, validators, tests, docs or the router code.

## Run the checks

CI (`.github/workflows/validators.yml`) runs these; run them locally:

```bash
python scripts/validate_domains.py
python scripts/validate_rank_schema.py        # alpha = 2 x rank
python scripts/validate_curriculum_order.py
python scripts/validate_no_pre_pivot.py
python scripts/validate_adapter_health.py <adapter.safetensors>
uv run python -m pytest tests/test_validate_*.py -q
```

The wider suite under `tests/` (router, memory, negotiator, reward) is
run with `uv run python -m pytest tests/<subdir>`. It loads no 35B model;
tests marked `integration` need real models or SSH and are skipped by
default. CI only gates the validators above, so say in your PR which
other tests you ran.

## Commits

- `feat(<area>): ...`, `fix(<area>): ...`, `docs(<area>): ...`
- Subject <= 50 characters, body lines <= 72, no underscore in the
  scope. English only.
- No `Co-Authored-By` trailer or AI attribution (the pre-commit hook
  rejects it). Never bypass git hooks.
- Branches: `feat/<name>`, `fix/<name>`, `docs/<name>`.

## Research rules

- Pre-registered experiments (`docs/prereg/`, e.g. `nh81-gb300/`) are
  frozen by a git tag recorded in the study's `freeze-record.md`. Only
  the PI creates or moves freeze tags.
- A PR must not modify a frozen run, its outputs, or a pre-registration
  or freeze file. A change after a freeze is a new dated amendment file
  (see `docs/prereg/nh81-gb300/amendment-2026-10-31.md`) signed by the
  PI.
- Null and negative results are published as they are. Do not tune,
  re-seed or re-threshold to turn a null into a positive. Wrong claims
  are corrected by a dated entry in `docs/ERRATA.md`, never deleted.
- Analytical estimates are labelled as estimates, never as
  measurements; no "saves energy" wording without a measurement.
- Keep the alpha = 2 x rank rule and the other hard invariants in
  `CLAUDE.md`; `validate_rank_schema.py` enforces the first.

## Where to start

Issues labelled
[`good first issue`](https://github.com/hypneum-lab/micro-kiki/issues?q=is%3Aopen+label%3A%22good+first+issue%22)
are scoped for newcomers (there may be none open at a given time;
issues tagged `documentation` or `hote:aucun` need no special
hardware). Questions and ideas: GitHub Discussions.

By contributing you agree your work is released under the repository
license (see `LICENSE`, `LICENSE-docs`).
