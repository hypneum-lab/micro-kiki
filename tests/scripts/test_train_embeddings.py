"""Tests for train_embeddings data pipeline (no model needed)."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import scripts.train_embeddings as te


class TestLoadTextsFromFile:
    def test_loads_messages_format(self, tmp_path):
        f = tmp_path / "train.jsonl"
        f.write_text(json.dumps({"messages": [
            {"role": "user", "content": "Design a buck converter for 12V to 5V"},
            {"role": "assistant", "content": "Here is the SPICE netlist for your converter..."},
        ]}) + "\n")
        texts = te.load_texts_from_file(f)
        assert len(texts) == 2
        assert "buck converter" in texts[0]

    def test_loads_prompt_format(self, tmp_path):
        f = tmp_path / "train.jsonl"
        f.write_text(json.dumps({"prompt": "Design an EMI filter for USB 3.0"}) + "\n")
        texts = te.load_texts_from_file(f)
        assert len(texts) == 1

    def test_skips_short_texts(self, tmp_path):
        f = tmp_path / "train.jsonl"
        f.write_text(json.dumps({"prompt": "hi"}) + "\n")
        texts = te.load_texts_from_file(f, min_len=20)
        assert len(texts) == 0


class TestBuildMnrlPairs:
    def test_creates_same_domain_pairs(self):
        domain_texts = {
            "spice": ["text A about SPICE circuits", "text B about SPICE simulation"],
            "emc": ["text C about EMC compliance", "text D about EMI filtering"],
        }
        pairs = te.build_mnrl_pairs(domain_texts)
        assert len(pairs) == 2  # one pair per domain
        for anchor, positive in pairs:
            assert isinstance(anchor, str)
            assert isinstance(positive, str)

    def test_skips_domain_with_one_text(self):
        domain_texts = {"spice": ["only one text here"]}
        pairs = te.build_mnrl_pairs(domain_texts)
        assert len(pairs) == 0


class TestBuildHardNegativeTriplets:
    def test_creates_triplets_from_confusing_pairs(self):
        domain_texts = {
            "embedded": ["firmware code for ARM Cortex", "interrupt handler for UART"],
            "stm32": ["HAL driver for STM32F4 timer", "CubeMX configuration guide"],
        }
        triplets = te.build_hard_negative_triplets(domain_texts)
        assert len(triplets) >= 2
        for anchor, positive, negative in triplets:
            assert isinstance(anchor, str)
            assert isinstance(negative, str)

    def test_skips_when_insufficient_data(self):
        domain_texts = {"embedded": ["only one"], "stm32": []}
        triplets = te.build_hard_negative_triplets(domain_texts)
        assert len(triplets) == 0


class TestOversample:
    def test_oversamples_to_target(self):
        texts = ["one", "two", "three"]
        result = te.oversample(texts, target=10, seed=42)
        assert len(result) == 10
        assert "one" in result

    def test_no_change_when_enough(self):
        texts = ["a", "b", "c", "d", "e"]
        result = te.oversample(texts, target=3, seed=42)
        assert len(result) == 5  # already exceeds target
