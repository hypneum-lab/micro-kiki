from __future__ import annotations

import pytest
from src.base.loader import BaseModelLoader


class TestBaseModelLoader:
    def test_init_with_path(self, tmp_path):
        loader = BaseModelLoader(model_path=str(tmp_path))
        assert loader.model_path == str(tmp_path)

    def test_enable_lora_switching(self):
        loader = BaseModelLoader()
        assert loader.lora_enabled is False
        loader.enable_lora_switching()
        assert loader.lora_enabled is True

    def test_with_stack_requires_lora(self):
        loader = BaseModelLoader()
        with pytest.raises(RuntimeError, match="LoRA switching not enabled"):
            with loader.with_stack("stack-01"):
                pass

    def test_list_available_stacks(self, tmp_path):
        stacks = tmp_path / "stacks"
        stacks.mkdir()
        (stacks / "stack-01-chat-fr").mkdir()
        (stacks / "stack-02-reasoning").mkdir()
        loader = BaseModelLoader(stacks_dir=str(stacks))
        assert "stack-01-chat-fr" in loader.list_stacks()
        assert "stack-02-reasoning" in loader.list_stacks()
