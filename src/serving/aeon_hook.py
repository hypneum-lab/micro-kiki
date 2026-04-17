"""Aeon integration hook for the serving pipeline.

Prepends recalled memories to prompts and writes new memories post-inference.
Uses dynamic memory budget and structured format matching POC v2.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.memory.aeon import AeonPalace

logger = logging.getLogger(__name__)

MEMORY_BUDGET = 3000

_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "niche-embeddings"


def _hash_embed(text: str) -> np.ndarray:
    """Deterministic hash-based embedding fallback (dim=384)."""
    h = hashlib.sha256(text.encode()).digest()
    rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
    vec = rng.randn(384).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


def create_aeon_palace(model_path: Path | None = None) -> AeonPalace:
    """Factory: create AeonPalace with trained embeddings or hash fallback.

    Args:
        model_path: Override path to embedding model directory.
            Defaults to ``models/niche-embeddings`` relative to repo root.

    Returns:
        AeonPalace configured with the best available embedding source.
    """
    from src.memory.aeon import AeonPalace

    path = model_path or _MODEL_PATH
    if path.exists() and (path / "config.json").exists():
        try:
            palace = AeonPalace(model_path=str(path))
            logger.info("AeonPalace: trained embeddings from %s", path)
            return palace
        except ImportError:
            logger.warning(
                "sentence-transformers not installed, falling back to hash embed"
            )

    # hash fallback for environments without sentence-transformers
    palace = AeonPalace(dim=384, embed_fn=_hash_embed)
    logger.info("AeonPalace: hash embedding fallback (dim=384)")
    return palace


class AeonServingHook:
    """Wraps AeonPalace for pre/post inference memory injection."""

    def __init__(self, palace: AeonPalace) -> None:
        self._palace = palace

    def pre_inference(self, prompt: str, top_k: int = 8) -> str:
        """Recall memories and prepend them with structured format.

        Uses dynamic budget: MEMORY_BUDGET chars split evenly across
        recalled episodes.
        """
        try:
            episodes = self._palace.recall(prompt, top_k=top_k)
        except Exception:
            logger.warning("Aeon recall failed, returning original prompt", exc_info=True)
            return prompt

        if not episodes:
            return prompt

        per_ep = max(200, MEMORY_BUDGET // len(episodes))
        lines = [ep.content[:per_ep] for ep in episodes]
        memory_block = (
            "### Previous conversation context:\n"
            + "\n---\n".join(lines)
            + "\n\n### Current question:\n"
        )
        return memory_block + prompt

    def post_inference(
        self,
        prompt: str,
        response: str,
        domain: str,
        turn_id: str,
    ) -> None:
        """Write the full interaction to Aeon memory."""
        content = f"User: {prompt}\nAssistant: {response}"
        try:
            self._palace.write(
                content=content,
                domain=domain,
                source=turn_id,
            )
        except Exception:
            logger.warning("Aeon write failed for turn %s", turn_id, exc_info=True)
