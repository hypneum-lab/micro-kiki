from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceResponse:
    text: str
    log_prob: float


class HttpBridge:
    """HTTP bridge from Mac Studio orchestrator to kxkm-ai vLLM server."""

    def __init__(
        self,
        vllm_url: str = "http://kxkm-ai:8000",
        timeout: float = 120.0,
    ) -> None:
        self._vllm_url = vllm_url
        self._timeout = timeout

    async def generate(self, prompt: str, **kwargs) -> tuple[str, float]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._vllm_url}/v1/completions",
                json={
                    "model": "micro-kiki",
                    "prompt": prompt,
                    "max_tokens": kwargs.get("max_tokens", 2048),
                    "temperature": kwargs.get("temperature", 0.7),
                    "logprobs": 1,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()

        data = response.json()
        choice = data["choices"][0]
        text = choice["text"]
        logprobs = choice.get("logprobs", {})
        avg_log_prob = 0.0
        if logprobs and logprobs.get("token_logprobs"):
            probs = [lp for lp in logprobs["token_logprobs"] if lp is not None]
            avg_log_prob = sum(probs) / len(probs) if probs else 0.0
        return text, avg_log_prob

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self._vllm_url}/health", timeout=5.0)
                return response.status_code == 200
        except httpx.RequestError:
            return False
