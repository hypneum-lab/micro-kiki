"""ANE speculative draft model for Mac (CoreML)."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ANEDraftModel:
    """Speculative decoder using Qwen3.5-0.8B on Apple Neural Engine."""

    def __init__(self, model_path: str = "models/qwen3.5-0.8b.mlpackage") -> None:
        self._model_path = model_path
        self._model = None

    def load(self) -> None:
        import coremltools as ct
        self._model = ct.models.MLModel(self._model_path)
        logger.info("ANE draft model loaded from %s", self._model_path)

    def predict_next_tokens(self, input_ids: list[int], n_tokens: int = 5) -> list[int]:
        """Draft n_tokens speculatively for verification by main model."""
        if self._model is None:
            raise RuntimeError("Call load() first")
        # CoreML inference — actual implementation depends on model export format
        logger.info("Drafting %d tokens via ANE", n_tokens)
        return []  # Placeholder — wired during CoreML export step
