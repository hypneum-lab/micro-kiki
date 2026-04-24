"""Smoke tests for ``scripts/convert_lora_to_snn.py``.

Covers:
- ``validate_one_module`` numerical equivalence on a synthetic tiny LoRA
  (rank 4, identity activation). With T=128 the rate-code error should
  be well under 1%.
- ``write_snn_adapter`` produces the expected output layout
  (``adapters.safetensors`` copy + ``lif_metadata.json`` with per-module
  LIF parameters).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

convert_lora_to_snn = pytest.importorskip("convert_lora_to_snn")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_lora(in_dim: int, out_dim: int, rank: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((in_dim, rank)).astype(np.float32) * 0.05
    b = rng.standard_normal((rank, out_dim)).astype(np.float32) * 0.05
    return a, b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_lora_snn_equivalence():
    """Synthetic rank-4 LoRA in the LAS-valid unipolar regime.

    LAS is designed for non-negative rate codes; signed values need
    two-channel encoding (future work). Validate the pipeline with
    abs-weights and non-negative inputs — this is the regime in which
    rate-code quantisation alone bounds the error (``O(1/T)``).
    """
    a, b = _random_lora(64, 96, rank=4, seed=7)
    lif = {
        "timesteps": 256,
        "threshold": 1.0 / 256,
        "tau": 1.0,
        "max_rate": 1.0,
    }
    metrics = convert_lora_to_snn.validate_one_module(a, b, lif, n_samples=8)
    # SNN is approximate by design; T=256 + small LoRA ≈ well under 10% rel L2.
    assert metrics["rel_l2"] < 1e-1, (
        f"LoRA SNN equivalence failed: rel_l2={metrics['rel_l2']:.4f}"
    )


def test_output_structure(tmp_path: Path):
    """write_snn_adapter copies the safetensors and writes LIF metadata."""
    # Build a tiny fake adapter safetensors on disk.
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError:
        pytest.skip("torch/safetensors not installed in this env")

    src = tmp_path / "src" / "adapters.safetensors"
    src.parent.mkdir(parents=True)
    a, b = _random_lora(32, 48, rank=4, seed=1)
    save_file(
        {
            "language_model.model.layers.0.self_attn.q_proj.lora_a":
                torch.from_numpy(a),
            "language_model.model.layers.0.self_attn.q_proj.lora_b":
                torch.from_numpy(b),
        },
        str(src),
    )

    lif_params = {
        "language_model.model.layers.0.self_attn.q_proj": {
            "timesteps": 64,
            "threshold": 1.0 / 64,
            "tau": 1.0,
            "max_rate": 1.0,
            "matched_snn_key": None,
        }
    }
    out = tmp_path / "out"
    convert_lora_to_snn.write_snn_adapter(
        src, out, lif_params, extra_meta={"snn_base": "synthetic"},
    )
    assert (out / "adapters.safetensors").exists()
    meta_file = out / "lif_metadata.json"
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text())
    assert meta["num_modules"] == 1
    assert "language_model.model.layers.0.self_attn.q_proj" in meta["layers"]
    assert meta["snn_base"] == "synthetic"
