"""Tests for scripts/measure_forgetting.py (phase 1a OPLoRA)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "measure_forgetting.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("measure_forgetting", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    spec.loader.exec_module(mod)
    return mod


def _adapter_tensors(b: torch.Tensor, a: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build a single-layer, single-projection LoRA adapter dict."""
    base = "base_model.model.model.layers.0.self_attn.q_proj"
    return {
        f"{base}.lora_A.weight": a.contiguous(),
        f"{base}.lora_B.weight": b.contiguous(),
    }


def test_orthogonal_weights_high_angle(tmp_path: Path) -> None:
    mod = _load_script_module()

    # Prior: delta = B1 @ A1 = [[1,0],[0,0]]
    b1 = torch.tensor([[1.0], [0.0]])
    a1 = torch.tensor([[1.0, 0.0]])
    # New: delta = B2 @ A2 = [[0,0],[0,1]] — orthogonal direction
    b2 = torch.tensor([[0.0], [1.0]])
    a2 = torch.tensor([[0.0, 1.0]])

    prior_path = tmp_path / "prior.safetensors"
    new_path = tmp_path / "new.safetensors"
    save_file(_adapter_tensors(b1, a1), str(prior_path))
    save_file(_adapter_tensors(b2, a2), str(new_path))

    out_path = tmp_path / "report.json"
    rc = mod.main(
        [
            "--prior-adapter",
            str(prior_path),
            "--new-adapter",
            str(new_path),
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    report = json.loads(out_path.read_text())
    assert report["gate_status"] == "angle_only_partial"
    assert report["angle_degrees_mean"] > 85.0, report


def test_parallel_weights_low_angle(tmp_path: Path) -> None:
    mod = _load_script_module()

    # Prior: delta = [[1,0],[0,0]]
    b1 = torch.tensor([[1.0], [0.0]])
    a1 = torch.tensor([[1.0, 0.0]])
    # New: same direction, tiny perturbation only in A/B scale
    b2 = torch.tensor([[1.001], [0.0005]])
    a2 = torch.tensor([[1.0005, 0.0002]])

    prior_path = tmp_path / "prior.safetensors"
    new_path = tmp_path / "new.safetensors"
    save_file(_adapter_tensors(b1, a1), str(prior_path))
    save_file(_adapter_tensors(b2, a2), str(new_path))

    out_path = tmp_path / "report.json"
    rc = mod.main(
        [
            "--prior-adapter",
            str(prior_path),
            "--new-adapter",
            str(new_path),
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    report = json.loads(out_path.read_text())
    assert report["angle_degrees_mean"] < 10.0, report
    assert report["warning"] == "angle below threshold"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
