"""Tests for scripts/run_forgetting_sweep.py.

Synthetic 3-adapter directory → 6 ordered pairs; verifies JSON matrix
shape, flag fields, and the below-30° detection path.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "run_forgetting_sweep.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_forgetting_sweep", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    spec.loader.exec_module(mod)
    return mod


def _write_adapter(
    dir_: Path,
    b: torch.Tensor,
    a: torch.Tensor,
    *,
    filename: str = "adapters.safetensors",
) -> Path:
    """Write a single-layer, q_proj-only PEFT-style LoRA adapter."""
    dir_.mkdir(parents=True, exist_ok=True)
    base = "base_model.model.model.layers.0.self_attn.q_proj"
    tensors = {
        f"{base}.lora_A.weight": a.detach().clone().contiguous(),
        f"{base}.lora_B.weight": b.detach().clone().contiguous(),
    }
    path = dir_ / filename
    save_file(tensors, str(path))
    return path


def _orthogonal_axes() -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Three adapters with deltas on mutually-orthogonal directions in R^4.

    Each produces a rank-1 delta aligned with a distinct canonical axis so
    pairwise angles are ~90°.
    """
    # delta_0 = B0 A0 = e_0 * e_0^T (2x2 block [[1,0],[0,0]])
    # delta_1 = e_1 * e_1^T                    ([[0,0],[0,1]])
    # delta_2 = e_0 * e_1^T                    ([[0,1],[0,0]]) — orthogonal to 0 and 1
    return [
        (torch.tensor([[1.0], [0.0]]), torch.tensor([[1.0, 0.0]])),
        (torch.tensor([[0.0], [1.0]]), torch.tensor([[0.0, 1.0]])),
        (torch.tensor([[1.0], [0.0]]), torch.tensor([[0.0, 1.0]])),
    ]


def test_three_adapter_matrix_shape(tmp_path: Path) -> None:
    """3 adapters → 6 ordered pairs, matrix well-formed, all-pass returns 0."""
    mod = _load_script_module()

    adapters_root = tmp_path / "adapters"
    names = ["alpha", "beta", "gamma"]
    for name, (b, a) in zip(names, _orthogonal_axes()):
        _write_adapter(adapters_root / name, b, a)

    out_path = tmp_path / "matrix.json"
    rc = mod.main(
        [
            "--adapters-dir",
            str(adapters_root),
            "--output",
            str(out_path),
        ]
    )

    assert rc == 0, "orthogonal adapters should all pass the 30° gate"
    payload = json.loads(out_path.read_text())
    assert sorted(payload["adapters"]) == sorted(names)
    assert len(payload["pairs"]) == 6  # 3 * 2 ordered pairs
    seen: set[tuple[str, str]] = set()
    for pair in payload["pairs"]:
        assert pair["prior"] != pair["new"]
        seen.add((pair["prior"], pair["new"]))
        assert pair["gate_status"] == "angle_only_partial"
        assert pair["angle_degrees_mean"] > 30.0
        assert isinstance(pair["angle_degrees_per_module"], dict)
        assert pair["angle_degrees_per_module"]
    assert len(seen) == 6, "every ordered pair must appear exactly once"

    flags = payload["flags"]
    assert flags["any_pair_below_30"] is False
    assert flags["min_mean_angle"] > 30.0
    assert flags["angle_threshold_degrees"] == 30.0
    assert flags["worst_pair"] is not None


def test_parallel_pair_flags_below_30(tmp_path: Path) -> None:
    """Two near-parallel adapters + one orthogonal → sweep must flag the low pair."""
    mod = _load_script_module()

    adapters_root = tmp_path / "adapters"
    # Near-parallel pair: adapter_a and adapter_b share the same delta direction.
    _write_adapter(
        adapters_root / "adapter_a",
        torch.tensor([[1.0], [0.0]]),
        torch.tensor([[1.0, 0.0]]),
    )
    _write_adapter(
        adapters_root / "adapter_b",
        torch.tensor([[1.001], [0.0005]]),
        torch.tensor([[1.0005, 0.0002]]),
    )
    # adapter_c orthogonal to both.
    _write_adapter(
        adapters_root / "adapter_c",
        torch.tensor([[0.0], [1.0]]),
        torch.tensor([[0.0, 1.0]]),
    )

    out_path = tmp_path / "matrix.json"
    rc = mod.main(
        [
            "--adapters-dir",
            str(adapters_root),
            "--output",
            str(out_path),
        ]
    )
    assert rc == 1, "parallel pair must trigger exit 1"
    payload = json.loads(out_path.read_text())
    flags = payload["flags"]
    assert flags["any_pair_below_30"] is True
    assert flags["min_mean_angle"] < 30.0
    assert set(flags["worst_pair"]) == {"adapter_a", "adapter_b"}


def test_missing_directory_errors(tmp_path: Path) -> None:
    mod = _load_script_module()
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        mod.main(["--adapters-dir", str(missing)])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
