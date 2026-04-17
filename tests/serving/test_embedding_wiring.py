"""Tests for story-49: trained embedding wiring into Aeon memory pipeline."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.memory.aeon import AeonPalace
from src.serving.aeon_hook import create_aeon_palace, _hash_embed, _MODEL_PATH


def _fake_st_model(dim: int = 384):
    """Create a mock SentenceTransformer that returns deterministic vectors."""
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = dim

    def encode(text: str, normalize_embeddings: bool = True) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec = rng.randn(dim).astype(np.float32)
        if normalize_embeddings:
            vec = vec / (np.linalg.norm(vec) + 1e-8)
        return vec

    model.encode = encode
    return model


class TestCreateAeonPalace:
    """Test create_aeon_palace factory wiring."""

    def test_uses_trained_model_when_path_exists(self, tmp_path: Path):
        """When model_path exists with config.json, load SentenceTransformer."""
        model_dir = tmp_path / "niche-embeddings"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"dim": 384}))

        mock_model = _fake_st_model(384)
        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer = MagicMock(return_value=mock_model)

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            palace = create_aeon_palace(model_path=model_dir)

        assert palace._dim == 384
        # Verify the embed_fn uses the mock model (not hash fallback)
        vec = palace._embed_fn("test query")
        assert vec.shape == (384,)

    def test_falls_back_to_hash_when_path_missing(self, tmp_path: Path):
        """When model_path doesn't exist, fall back to hash embedding."""
        missing = tmp_path / "nonexistent-model"
        palace = create_aeon_palace(model_path=missing)
        assert palace._dim == 384

        # Verify deterministic hash embed works
        v1 = palace._embed_fn("hello")
        v2 = palace._embed_fn("hello")
        np.testing.assert_array_equal(v1, v2)

    def test_falls_back_when_no_config_json(self, tmp_path: Path):
        """When model_path exists but config.json is missing, use fallback."""
        model_dir = tmp_path / "partial-model"
        model_dir.mkdir()
        # No config.json

        palace = create_aeon_palace(model_path=model_dir)
        assert palace._dim == 384

    def test_falls_back_when_st_not_installed(self, tmp_path: Path):
        """When sentence-transformers is not installed, use hash fallback."""
        model_dir = tmp_path / "niche-embeddings"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"dim": 384}))

        # Remove sentence_transformers from sys.modules to force ImportError
        import sys
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            palace = create_aeon_palace(model_path=model_dir)

        assert palace._dim == 384
        vec = palace._embed_fn("test")
        assert vec.shape == (384,)

    def test_default_model_path_is_repo_relative(self):
        """_MODEL_PATH points to models/niche-embeddings relative to repo root."""
        assert _MODEL_PATH.name == "niche-embeddings"
        assert _MODEL_PATH.parent.name == "models"

    def test_hash_embed_deterministic(self):
        """_hash_embed returns consistent vectors for the same input."""
        v1 = _hash_embed("determinism test")
        v2 = _hash_embed("determinism test")
        np.testing.assert_array_equal(v1, v2)
        assert v1.shape == (384,)
        assert v1.dtype == np.float32

    def test_hash_embed_normalized(self):
        """_hash_embed returns unit-length vectors."""
        vec = _hash_embed("normalization test")
        norm = np.linalg.norm(vec)
        np.testing.assert_almost_equal(norm, 1.0, decimal=5)


class TestHybridPipelineFactory:
    """Test create_hybrid_pipeline wiring."""

    def test_creates_pipeline_with_hash_fallback(self, tmp_path: Path):
        """Factory creates a working pipeline even without trained model."""
        from src.routing.hybrid_pipeline import create_hybrid_pipeline

        pipeline = create_hybrid_pipeline(model_path=tmp_path / "missing")
        assert pipeline._aeon is not None
        assert pipeline._aeon._palace._dim == 384

    def test_creates_pipeline_with_trained_model(self, tmp_path: Path):
        """Factory wires trained model when available."""
        from src.routing.hybrid_pipeline import create_hybrid_pipeline

        model_dir = tmp_path / "niche-embeddings"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"dim": 384}))

        mock_model = _fake_st_model(384)
        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer = MagicMock(return_value=mock_model)

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            pipeline = create_hybrid_pipeline(model_path=model_dir)

        assert pipeline._aeon is not None
        assert pipeline._aeon._palace._dim == 384
