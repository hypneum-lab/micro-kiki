from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


class BaseModelLoader:
    """Loads Qwen3.5-4B base model with LoRA adapter hot-swapping."""

    def __init__(self, model_path: str = "models/qwen3.5-4b/bf16", stacks_dir: str = "outputs/stacks") -> None:
        self.model_path = model_path
        self.stacks_dir = stacks_dir
        self.lora_enabled = False
        self._model = None
        self._tokenizer = None
        self._active_stack: str | None = None

    def load_bf16(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info("Loading BF16 from %s", self.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True,
        )
        return self._model, self._tokenizer

    def enable_lora_switching(self) -> None:
        self.lora_enabled = True
        logger.info("LoRA switching enabled")

    @contextmanager
    def with_stack(self, adapter_name: str) -> Generator[None, None, None]:
        if not self.lora_enabled:
            raise RuntimeError("LoRA switching not enabled. Call enable_lora_switching() first.")
        adapter_path = Path(self.stacks_dir) / adapter_name
        if not adapter_path.exists():
            raise FileNotFoundError(f"Stack adapter not found: {adapter_path}")
        logger.info("Loading adapter: %s", adapter_name)
        prev_stack = self._active_stack
        self._active_stack = adapter_name
        try:
            yield
        finally:
            self._active_stack = prev_stack

    def list_stacks(self) -> list[str]:
        stacks_path = Path(self.stacks_dir)
        if not stacks_path.exists():
            return []
        return sorted(d.name for d in stacks_path.iterdir() if d.is_dir())

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer
