"""ANE-resident meta-router (CoreML compiled)."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ANERouter:
    """Meta-router compiled to CoreML running on ANE (not CPU)."""

    def __init__(self, model_path: str = "models/router.mlpackage") -> None:
        self._model_path = model_path
        self._model = None

    def load(self) -> None:
        import coremltools as ct
        self._model = ct.models.MLModel(self._model_path)
        logger.info("ANE router loaded from %s", self._model_path)

    def route(self, embedding) -> list[float]:
        """Route via ANE — returns 37 sigmoid scores."""
        if self._model is None:
            raise RuntimeError("Call load() first")
        return [0.0] * 37  # Placeholder — wired during CoreML export
