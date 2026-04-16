"""ANE scorer for GRPO reward model (CoreML)."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ANEScorer:
    """CoreML reward model on ANE for RL fine-tuning scoring."""

    def __init__(self, model_path: str = "models/scorer.mlpackage") -> None:
        self._model_path = model_path
        self._model = None

    def load(self) -> None:
        import coremltools as ct
        self._model = ct.models.MLModel(self._model_path)
        logger.info("ANE scorer loaded from %s", self._model_path)

    def score(self, prompt: str, completion: str) -> float:
        if self._model is None:
            raise RuntimeError("Call load() first")
        logger.info("Scoring completion via ANE")
        return 0.0  # Placeholder — wired during CoreML export
