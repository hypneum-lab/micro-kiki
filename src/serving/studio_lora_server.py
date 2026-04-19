"""Studio-side MLX inference server with per-request adapter-swap.

Wraps Qwen3.6-35B-A3B + any of 10 trained LoRA adapters from
~/micro-kiki/outputs/stacks/stack-<domain>/. Base model loaded once with
LoRALinear shells; per-request swap reloads adapter weights in place (no
base reload, no double-conversion of Linear → LoRALinear).

Pattern derived from commit b102115 ('E2E smoke on Studio + real results').

Exposes two HTTP endpoints (OpenAI-compatible):
- GET  /v1/models               — health / currently-loaded adapter
- POST /v1/chat/completions     — body {messages, max_tokens, adapter?}
                                  if adapter is set and differs from current, swap first.

Run:
    ~/micro-kiki/.venv/bin/python -m src.serving.studio_lora_server \\
        --base ~/models/Qwen3.6-35B-A3B \\
        --stacks-root ~/micro-kiki/outputs/stacks \\
        --host 0.0.0.0 --port 19000
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# MLX imports are deferred; tests mock them.
try:  # pragma: no cover — only available on Studio / systems with MLX
    import mlx.core as mx  # type: ignore
except Exception:  # pragma: no cover
    mx = None  # type: ignore


@dataclass
class AdapterState:
    model: Any
    tokenizer: Any
    stacks_root: Path
    currently_loaded: str | None = None


def adapter_name_to_path(name: str, stacks_root: Path) -> Path:
    """Resolve 'dsp' → <stacks_root>/stack-dsp/adapters.safetensors."""
    p = stacks_root / f"stack-{name}" / "adapters.safetensors"
    if not p.parent.exists():
        raise FileNotFoundError(f"no adapter for domain {name!r} under {stacks_root}")
    return p


def build_chat_prompt(tokenizer: Any, user_message: str) -> str:
    """Build the prompt string using the tokenizer's chat template.

    IMPORTANT: NO system-prompt persona is injected. The weight-level LoRA
    adapter provides specialisation — we do NOT want the persona-refusal
    pathology from C2 baseline (pattern #1).
    """
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def swap_adapter(
    state: AdapterState,
    adapter_name: str,
    *,
    adapter_path_override: Path | None = None,
) -> None:
    """Load the named adapter's weights onto the pre-converted LoRALinear layers.

    Passing adapter_name='base' resets currently_loaded to None without weight IO
    (LoRALinear's LoRA.B branch contributes 0 when freshly zeroed; in practice
    when swapping across non-base adapters we always overwrite, so 'base' is
    only used at startup).
    """
    if adapter_name == "base":
        state.currently_loaded = None
        return

    if state.currently_loaded == adapter_name:
        return  # already loaded, no-op

    adapter_path = adapter_path_override or adapter_name_to_path(
        adapter_name, state.stacks_root
    )
    if mx is None:  # pragma: no cover — only hit in MLX-less envs
        raise RuntimeError("mlx.core unavailable; run this on Studio")
    weights = mx.load(str(adapter_path))
    state.model.load_weights(list(weights.items()), strict=False)
    state.currently_loaded = adapter_name


def _bootstrap(base_path: Path, stacks_root: Path) -> AdapterState:  # pragma: no cover
    """Load base + convert to LoRALinear shells once at server startup."""
    from mlx_lm import load  # type: ignore
    from mlx_lm.tuner.utils import linear_to_lora_layers  # type: ignore

    logger.info("loading base model from %s", base_path)
    model, tokenizer = load(str(base_path))

    # Shell conversion: use rank 16, alpha 32 (matches PRD story-4 config).
    lora_config = [
        {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.0,
            "scale": 2.0,
            "keys": [
                "self_attn.q_proj", "self_attn.k_proj",
                "self_attn.v_proj", "self_attn.o_proj",
            ],
        },
    ]
    num_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if num_layers is None:
        num_layers = 40  # Qwen3.6-35B-A3B default; fallback
    linear_to_lora_layers(model, num_layers, lora_config[0])
    logger.info("converted %d layers to LoRALinear shells", num_layers)

    return AdapterState(
        model=model, tokenizer=tokenizer, stacks_root=stacks_root, currently_loaded=None
    )


def _build_app(state: AdapterState):  # pragma: no cover — tested manually via curl
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="Studio LoRA Swap Server")

    class Message(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        model: str | None = None
        messages: list[Message]
        max_tokens: int = 512
        temperature: float = 0.0
        adapter: str | None = None  # our extension
        chat_template_kwargs: dict | None = None  # ignored; MLX handles its own template

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {"id": "qwen3.6-35b-a3b",
                 "currently_loaded_adapter": state.currently_loaded},
            ],
        }

    @app.post("/v1/chat/completions")
    def completions(req: ChatCompletionRequest):
        from mlx_lm import generate  # type: ignore

        if req.adapter is not None and req.adapter != state.currently_loaded:
            swap_adapter(state, req.adapter)

        # Flatten messages into the user-turn content expected by build_chat_prompt.
        # We only take the LAST user message (one-shot); keep the flow simple.
        user_content = next(
            (m.content for m in reversed(req.messages) if m.role == "user"),
            req.messages[-1].content if req.messages else "",
        )
        prompt = build_chat_prompt(state.tokenizer, user_content)
        text = generate(
            state.model, state.tokenizer, prompt, max_tokens=req.max_tokens,
            verbose=False,
        )
        return {
            "id": "cmpl-c2lora",
            "object": "chat.completion",
            "model": "qwen3.6-35b-a3b",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "c2_lora_meta": {"adapter": state.currently_loaded},
        }

    return app


def main() -> int:  # pragma: no cover
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, type=Path)
    p.add_argument("--stacks-root", required=True, type=Path)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=19000)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    state = _bootstrap(args.base, args.stacks_root)
    app = _build_app(state)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
