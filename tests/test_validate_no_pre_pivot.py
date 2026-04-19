"""Smoke test for scripts/validate_no_pre_pivot.py against the live src tree."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from validate_no_pre_pivot import FORBIDDEN_STRINGS, scan_file, validate  # noqa: E402


def test_no_pre_pivot_identifiers():
    ok, msg = validate()
    assert ok, msg


def test_forbidden_strings_nonempty():
    assert FORBIDDEN_STRINGS, "forbidden list must not be empty"
    assert "Qwen3.5-4B" in FORBIDDEN_STRINGS
    assert "[0.0] * 32" in FORBIDDEN_STRINGS


def test_scan_file_detects_synthetic_hit(tmp_path):
    fake = tmp_path / "synthetic.py"
    fake.write_text("base = 'Qwen3.5-4B'\n", encoding="utf-8")
    hits = scan_file(fake)
    assert any(needle == "Qwen3.5-4B" for _, needle, _ in hits)
