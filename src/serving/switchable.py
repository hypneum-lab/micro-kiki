"""Runtime adapter switcher: hot-swap LoRA stacks with caching."""
from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_ACTIVE_STACKS = 4


class SwitchableModel:
    """Holds base model + registry of LoRA adapters with hot-swap support."""

    def __init__(self, base_model=None, tokenizer=None, stacks_dir: str = "outputs/stacks") -> None:
        self._base = base_model
        self._tokenizer = tokenizer
        self._stacks_dir = Path(stacks_dir)
        self._loaded_adapters: dict[str, object] = {}
        self._active_stacks: list[str] = []
        self._cache_key: tuple[str, ...] = ()

    def list_available(self) -> list[str]:
        if not self._stacks_dir.exists():
            return []
        return sorted(d.name for d in self._stacks_dir.iterdir() if d.is_dir())

    def apply_stacks(self, names: list[str]) -> None:
        """Apply list of LoRA stacks (max 4). Caches merged weights."""
        if len(names) > MAX_ACTIVE_STACKS:
            raise ValueError(f"Cannot activate {len(names)} stacks (max {MAX_ACTIVE_STACKS})")

        new_key = tuple(sorted(names))
        if new_key == self._cache_key:
            logger.debug("Stack set unchanged, using cache")
            return

        start = time.monotonic()
        self._active_stacks = list(names)
        self._cache_key = new_key

        if self._base is not None:
            self._apply_adapters_to_model(names)

        elapsed = time.monotonic() - start
        logger.info("Applied stacks %s in %.0fms", names, elapsed * 1000)

    def _apply_adapters_to_model(self, names: list[str]) -> None:
        """Apply PEFT adapters to the model. Requires torch + peft."""
        from peft import PeftModel

        for name in names:
            adapter_path = self._stacks_dir / name
            if not adapter_path.exists():
                raise FileNotFoundError(f"Adapter not found: {adapter_path}")

            if name not in self._loaded_adapters:
                if isinstance(self._base, PeftModel):
                    self._base.load_adapter(str(adapter_path), adapter_name=name)
                else:
                    self._base = PeftModel.from_pretrained(self._base, str(adapter_path))
                self._loaded_adapters[name] = True

        if names and isinstance(self._base, PeftModel):
            self._base.set_adapter(names[0])

    def clear_stacks(self) -> None:
        self._active_stacks = []
        self._cache_key = ()
        logger.info("Cleared all active stacks")

    @property
    def active_stacks(self) -> list[str]:
        return list(self._active_stacks)

    @property
    def model(self):
        return self._base

    @property
    def tokenizer(self):
        return self._tokenizer
