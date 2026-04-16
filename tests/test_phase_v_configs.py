"""Validate all Phase V stack configs and eval files."""
from __future__ import annotations

import json
import pytest
from pathlib import Path

import yaml

PHASE_V_STACKS = [
    (4, "typescript"), (5, "cpp"), (6, "rust"), (7, "html-css"),
    (8, "shell"), (9, "sql"), (10, "yaml-json"), (11, "docker"),
    (12, "kicad-dsl"), (13, "spice"), (14, "lua-upy"),
]

REQUIRED_CONFIG_KEYS = [
    "base_model", "lora_rank", "lora_alpha",
    "learning_rate", "batch_size", "grad_accum", "epochs", "seq_len", "dataset",
    "domain", "curriculum_order",
]


class TestPhaseVConfigs:
    @pytest.mark.parametrize("num,domain", PHASE_V_STACKS)
    def test_config_exists_and_valid(self, num, domain):
        path = Path(f"configs/stack-{num:02d}-{domain}.yaml")
        assert path.exists(), f"Missing config: {path}"
        with open(path) as f:
            config = yaml.safe_load(f)
        for key in REQUIRED_CONFIG_KEYS:
            assert key in config, f"Missing key '{key}' in {path}"
        assert config["curriculum_order"] == num
        assert config["domain"] == domain

    @pytest.mark.parametrize("num,domain", PHASE_V_STACKS)
    def test_base_model_is_35b(self, num, domain):
        path = Path(f"configs/stack-{num:02d}-{domain}.yaml")
        with open(path) as f:
            config = yaml.safe_load(f)
        assert "35B-A3B" in config["base_model"]

    @pytest.mark.parametrize("num,domain", PHASE_V_STACKS)
    def test_eval_file_exists(self, num, domain):
        path = Path(f"data/eval/{domain}.jsonl")
        assert path.exists(), f"Missing eval: {path}"
        lines = [l for l in path.read_text().strip().split("\n") if l]
        assert len(lines) >= 5
