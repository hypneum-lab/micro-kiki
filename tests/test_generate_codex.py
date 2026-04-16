"""Tests for scripts/generate_dataset_codex.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Make the scripts directory importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import generate_dataset_codex as gdc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**kwargs) -> SimpleNamespace:
    defaults = {
        "domain": "kicad-dsl",
        "all": False,
        "max": 10,
        "delay": 0.0,
        "dry_run": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Test: dry_run does not create output files
# ---------------------------------------------------------------------------


def test_dry_run_does_not_create_files(tmp_path, monkeypatch):
    """dry_run mode must not write any output files."""
    # Patch output root to tmp_path so we don't touch real data
    monkeypatch.setattr(gdc, "OUTPUT_ROOT", tmp_path / "codex-generated")
    monkeypatch.setattr(gdc, "EXPANDED_ROOT", tmp_path / "prompts-expanded")

    # Create a minimal prompts file
    prompts_dir = tmp_path / "prompts-expanded"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "kicad-dsl.jsonl").write_text(
        json.dumps({"prompt": "Describe a KiCad symbol"}) + "\n"
    )

    args = _make_args(dry_run=True)
    gdc.process_domain("kicad-dsl", args)

    out_file = tmp_path / "codex-generated" / "kicad-dsl" / "train.jsonl"
    assert not out_file.exists(), "dry_run must not create output files"


# ---------------------------------------------------------------------------
# Test: load_prompts returns correct count
# ---------------------------------------------------------------------------


def test_load_prompts_returns_correct_count(tmp_path, monkeypatch):
    """load_prompts returns at most max_n unique prompts."""
    prompts_dir = tmp_path / "prompts-expanded"
    prompts_dir.mkdir(parents=True)
    monkeypatch.setattr(gdc, "EXPANDED_ROOT", prompts_dir)

    n = 7
    lines = [json.dumps({"prompt": f"Prompt #{i}"}) for i in range(n)]
    (prompts_dir / "stm32.jsonl").write_text("\n".join(lines) + "\n")

    result = gdc.load_prompts("stm32", max_n=5)
    assert len(result) == 5

    result_all = gdc.load_prompts("stm32", max_n=100)
    assert len(result_all) == n


def test_load_prompts_deduplicates(tmp_path, monkeypatch):
    """load_prompts skips duplicate prompts."""
    prompts_dir = tmp_path / "prompts-expanded"
    prompts_dir.mkdir(parents=True)
    monkeypatch.setattr(gdc, "EXPANDED_ROOT", prompts_dir)

    lines = [json.dumps({"prompt": "Same prompt"}) for _ in range(5)]
    (prompts_dir / "embedded.jsonl").write_text("\n".join(lines) + "\n")

    result = gdc.load_prompts("embedded", max_n=100)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Test: output format has required keys
# ---------------------------------------------------------------------------


def test_output_format_has_required_keys(tmp_path, monkeypatch):
    """Generated examples must contain messages, domain, and source keys."""
    monkeypatch.setattr(gdc, "OUTPUT_ROOT", tmp_path / "codex-generated")
    monkeypatch.setattr(gdc, "EXPANDED_ROOT", tmp_path / "prompts-expanded")

    prompts_dir = tmp_path / "prompts-expanded"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "spice.jsonl").write_text(
        json.dumps({"prompt": "Write a SPICE netlist for a RC circuit"}) + "\n"
    )

    # Patch generate_one to return a fake response without calling codex
    monkeypatch.setattr(
        gdc, "generate_one", lambda prompt, retries=3: "Here is the netlist: .tran 1n 10u"
    )

    args = _make_args(domain="spice")
    gdc.process_domain("spice", args)

    out_file = tmp_path / "codex-generated" / "spice" / "train.jsonl"
    assert out_file.exists()

    with open(out_file) as f:
        example = json.loads(f.readline())

    assert "messages" in example, "output must have 'messages' key"
    assert "domain" in example, "output must have 'domain' key"
    assert "source" in example, "output must have 'source' key"
    assert example["domain"] == "spice"
    assert example["source"] == "codex-generated"

    messages = example["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# Test: FileNotFoundError for missing codex CLI handled gracefully
# ---------------------------------------------------------------------------


def test_generate_one_handles_missing_codex(monkeypatch):
    """generate_one must return None (not crash) when codex is not installed."""
    with patch("subprocess.run", side_effect=FileNotFoundError("codex not found")):
        result = gdc.generate_one("some prompt", retries=1)
    assert result is None
