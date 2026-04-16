from __future__ import annotations

import pytest
from pathlib import Path

from src.stacks.trainer import load_training_config, lora_config_from_dict, StackTrainer, LoRAConfig


class TestLoadConfig:
    def test_loads_yaml(self, tmp_path):
        cfg = tmp_path / "s.yaml"
        cfg.write_text(
            "base_model: Qwen/Qwen3.5-35B-A3B\nlora_rank: 16\nlora_alpha: 32\n"
            "learning_rate: 0.0002\nbatch_size: 4\ngrad_accum: 8\nepochs: 3\n"
            "seq_len: 4096\ndataset: d.jsonl\n"
        )
        config = load_training_config(cfg)
        assert config["lora_rank"] == 16

    def test_missing_key_raises(self, tmp_path):
        (tmp_path / "bad.yaml").write_text("lora_rank: 16\n")
        with pytest.raises(KeyError):
            load_training_config(tmp_path / "bad.yaml")


class TestLoRAConfig:
    def test_defaults(self):
        c = LoRAConfig()
        assert c.rank == 16 and c.alpha == 32
        assert "q_proj" in c.target_modules

    def test_from_dict(self):
        c = lora_config_from_dict({"lora_rank": 32, "lora_alpha": 64})
        assert c.rank == 32 and c.alpha == 64


class TestStackTrainer:
    def test_init(self):
        t = StackTrainer(output_dir="outputs/test")
        assert t.output_dir == "outputs/test"
        assert "35B-A3B" in t.base_model_path

    def test_from_config(self, tmp_path):
        cfg = tmp_path / "stack.yaml"
        cfg.write_text(
            "base_model: Qwen/Qwen3.5-35B-A3B\nlora_rank: 16\nlora_alpha: 32\n"
            "learning_rate: 0.0002\nbatch_size: 4\ngrad_accum: 8\nepochs: 3\n"
            "seq_len: 4096\ndataset: data/distilled/chat-fr.jsonl\n"
        )
        t = StackTrainer.from_config(cfg)
        assert t.lora_config.rank == 16
