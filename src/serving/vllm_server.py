"""vLLM server wrapper with dynamic LoRA and router sidecar."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

VLLM_ARGS = [
    "--model", "models/qwen3.5-4b-diffattn/",
    "--enable-lora",
    "--max-loras", "4",
    "--max-lora-rank", "16",
    "--port", "8100",
    "--trust-remote-code",
]

VLLM_ENV = {
    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
}


def get_vllm_command() -> list[str]:
    return ["python", "-m", "vllm.entrypoints.openai.api_server"] + VLLM_ARGS


def get_router_sidecar_command() -> list[str]:
    return ["uvicorn", "src.serving.router_sidecar:app", "--port", "8101"]
