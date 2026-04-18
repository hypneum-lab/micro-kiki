from __future__ import annotations

import json

import pytest

pytest.importorskip("huggingface_hub")
from scripts.download_base import verify_safetensors_index, verify_sizes


class TestVerifySizes:
    def test_empty_dir_returns_zero(self, tmp_path):
        total = verify_sizes(tmp_path)
        assert total == 0

    def test_sums_file_bytes(self, tmp_path):
        (tmp_path / "model.safetensors").write_bytes(b"\x00" * 1024)
        (tmp_path / "tokenizer.json").write_bytes(b"\x00" * 256)
        total = verify_sizes(tmp_path)
        assert total == 1024 + 256


class TestVerifySafetensorsIndex:
    def test_missing_index_is_noop(self, tmp_path):
        # No index.json => function returns without raising.
        verify_safetensors_index(tmp_path)

    def test_empty_weight_map_is_noop(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {}})
        )
        verify_safetensors_index(tmp_path)

    def test_missing_shard_raises(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"layer.0.weight": "model-00001-of-00002.safetensors"}})
        )
        with pytest.raises(RuntimeError, match="missing shard"):
            verify_safetensors_index(tmp_path)

    def test_all_shards_present_ok(self, tmp_path):
        (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"\x00" * 16)
        (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"\x00" * 16)
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors",
                    }
                }
            )
        )
        verify_safetensors_index(tmp_path)
