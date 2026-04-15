from __future__ import annotations

import pytest
from src.serving.switchable import SwitchableModel, MAX_ACTIVE_STACKS


class TestSwitchableModel:
    def test_list_available(self, tmp_path):
        stacks = tmp_path / "stacks"
        stacks.mkdir()
        (stacks / "stack-01-chat-fr").mkdir()
        (stacks / "stack-02-reasoning").mkdir()
        model = SwitchableModel(stacks_dir=str(stacks))
        available = model.list_available()
        assert "stack-01-chat-fr" in available
        assert "stack-02-reasoning" in available

    def test_apply_stacks_caches(self, tmp_path):
        stacks = tmp_path / "stacks"
        stacks.mkdir()
        model = SwitchableModel(stacks_dir=str(stacks))
        model.apply_stacks([])
        assert model.active_stacks == []

    def test_max_stacks_enforced(self):
        model = SwitchableModel()
        with pytest.raises(ValueError, match="max"):
            model.apply_stacks(["a", "b", "c", "d", "e"])

    def test_clear_stacks(self):
        model = SwitchableModel()
        model.apply_stacks([])
        model.clear_stacks()
        assert model.active_stacks == []

    def test_cache_key_prevents_reload(self, tmp_path):
        model = SwitchableModel(stacks_dir=str(tmp_path))
        model.apply_stacks([])
        key1 = model._cache_key
        model.apply_stacks([])
        assert model._cache_key == key1
