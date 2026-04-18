# src/ — Python runtime

Python 3.11+, uv-managed. This tree is the **runtime** (routing, cognitive layer, serving, eval glue) — training execution lives in `~/KIKI-Mac_tunner/`.

## Layout (what lives where)

| Dir | Responsibility |
|---|---|
| `base/` | Base-model loading helpers, tokenizer wrappers |
| `stacks/` | LoRA stack runtime: `moe_lora.py` (legacy), `oplora.py` (orthogonal init), `qtha.py`, `trainer.py`. See `stacks/CLAUDE.md` |
| `routing/` | Meta-router (sigmoid) + dispatcher + VQC/TN experimental routers. See `routing/CLAUDE.md` |
| `memory/` | Aeon cognitive memory: `atlas.py` (SIMD), `trace.py` (graph), `aeonsleep.py`, `aeon_predictor.py`, `backends/` |
| `cognitive/` | Negotiator (CAMP + Catfish), anti-bias (KnowBias, RBD, DeFrame), forgetting gate, sleep tagger, consolidation |
| `distill/` | `teacher_client.py` (Qwen3-Coder-480B), `generator.py`, `dedup.py` |
| `eval/` | Forgetting math, benchmarks, MAP harness. See `eval/CLAUDE.md` |
| `serving/` | MLX server, vLLM server, aeon_hook, switchable runtime, ANE draft/scorer, `moe_lora_runtime.py` |
| `search/`, `critique/`, `compress/`, `orchestrator/`, `ralph/`, `spiking/` | Supporting subsystems |

## Module style

- `from __future__ import annotations` in every module.
- Type hints on all public functions; Google-style docstrings for non-trivial ones.
- Import order: stdlib → third-party (torch, transformers, peft, mlx) → `src.*`.
- Immutable configs: `@dataclass(frozen=True)` or Pydantic `BaseModel`.
- Explicit device placement (`device_map`, `.to(device)`); no implicit global device.
- Assert tensor shapes before reshape — name dimensions in comments when non-obvious.

## Anti-patterns (src-specific)

- No global model/tokenizer state — pass them explicitly.
- No bare `except:` — catch specific exceptions.
- No `print()` — use `logging`.
- No hardcoded paths — load from configs/env.
- Don't import training/tuner code from this tree — cross into `~/KIKI-Mac_tunner/` via subprocess or HTTP, not Python import.
- The legacy `moe_lora.py` is kept for reference only; new stacks use PEFT `LoraConfig` directly (see `stacks/CLAUDE.md`).
