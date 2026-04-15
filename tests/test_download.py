from __future__ import annotations

import pytest

pytest.importorskip("huggingface_hub")
from scripts.download_base import verify_download


class TestDownloadVerify:
    def test_verify_missing_files(self, tmp_path):
        result = verify_download(str(tmp_path / "nope"), str(tmp_path / "nope.gguf"))
        assert result["bf16_exists"] is False
        assert result["q4_exists"] is False

    def test_verify_bf16_size_check(self, tmp_path):
        bf16_dir = tmp_path / "bf16"
        bf16_dir.mkdir()
        (bf16_dir / "model.safetensors").write_bytes(b"\x00" * 1024)
        result = verify_download(str(bf16_dir), str(tmp_path / "q.gguf"))
        assert result["bf16_exists"] is True
        assert result["bf16_ok"] is False
