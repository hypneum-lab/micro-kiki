"""Tests for scripts/validate_adapter_health.py.

Build synthetic safetensors fixtures (no real adapter weights) and
invoke the validator via its ``main()`` entry point to assert on the
exit code contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from validate_adapter_health import main  # noqa: E402


def _write_adapter(
    path: Path,
    *,
    a_norm: float = 1.0,
    b_norm: float = 0.0,
    n_layers: int = 2,
) -> Path:
    """Write a synthetic adapter with configurable lora_a / lora_b norms."""
    tensors: dict[str, np.ndarray] = {}
    for layer in range(n_layers):
        base = f"language_model.model.layers.{layer}.self_attn.q_proj"
        a = np.random.rand(128, 8).astype(np.float32)
        if a_norm == 0.0:
            a = np.zeros_like(a)
        else:
            a = a * (a_norm / float(np.linalg.norm(a) or 1.0))
        b = np.random.rand(8, 128).astype(np.float32)
        if b_norm == 0.0:
            b = np.zeros_like(b)
        else:
            b = b * (b_norm / float(np.linalg.norm(b) or 1.0))
        tensors[f"{base}.lora_a"] = a
        tensors[f"{base}.lora_b"] = b
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    return path


def test_all_zero_lora_b_fails(tmp_path: Path) -> None:
    adapter = _write_adapter(tmp_path / "bad.safetensors", a_norm=1.0, b_norm=0.0)
    assert main([str(adapter)]) == 1


def test_one_nonzero_lora_b_passes(tmp_path: Path) -> None:
    adapter = _write_adapter(tmp_path / "good.safetensors", a_norm=1.0, b_norm=1.0)
    assert main([str(adapter)]) == 0


def test_dir_mode_fails_if_any_bad(tmp_path: Path) -> None:
    _write_adapter(tmp_path / "good" / "adapter.safetensors", a_norm=1.0, b_norm=1.0)
    _write_adapter(tmp_path / "bad" / "adapter.safetensors", a_norm=1.0, b_norm=0.0)
    assert main(["--adapters-dir", str(tmp_path)]) == 1


def test_epsilon_override(tmp_path: Path) -> None:
    # max B norm = 5e-5: passes default epsilon (1e-6), fails --epsilon 1e-4
    adapter = _write_adapter(tmp_path / "tiny.safetensors", a_norm=1.0, b_norm=5e-5)
    assert main([str(adapter)]) == 0
    assert main(["--epsilon", "1e-4", str(adapter)]) == 1


def test_help_parses() -> None:
    """argparse --help should exit(0) cleanly (sanity for CLI wiring)."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
