"""Tests for scripts/measure_forgetting.py.

Phase 1a covered the angle-only path (status ``angle_only_partial``).
Phase 1b adds the full gate: ``angle<30° AND winrate_drop>0.03``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Torch is required by the safetensors-backed fixtures below.
torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

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


def _write_adapter(path: Path, b: torch.Tensor, a: torch.Tensor) -> None:
    save_file(_adapter_tensors(b, a), str(path))


def _write_eval_dataset(path: Path, entries: list[dict[str, str]]) -> None:
    path.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Phase 1a regression — angle-only path (back-compat)
# ---------------------------------------------------------------------------


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
    _write_adapter(prior_path, b1, a1)
    _write_adapter(new_path, b2, a2)

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
    _write_adapter(prior_path, b1, a1)
    _write_adapter(new_path, b2, a2)

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


# ---------------------------------------------------------------------------
# Phase 1b — full gate (angle + win-rate, AND-logic rollback)
# ---------------------------------------------------------------------------


def _patch_generate_fn(monkeypatch, callable_: object) -> str:
    """Install ``callable_`` under a fresh module path and return its locator."""
    import types

    module_name = "tests_mock_gen_fn_module"
    mod = types.ModuleType(module_name)
    mod.generate = callable_  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, mod)
    return f"{module_name}:generate"


def test_full_gate_pass_high_angle(tmp_path: Path, monkeypatch) -> None:
    """angle > 30° → gate passes regardless of winrate drop."""
    mod = _load_script_module()

    # Orthogonal-ish deltas → high angle.
    b1 = torch.tensor([[1.0], [0.0]])
    a1 = torch.tensor([[1.0, 0.0]])
    b2 = torch.tensor([[0.0], [1.0]])
    a2 = torch.tensor([[0.0, 1.0]])

    prior_path = tmp_path / "prior.safetensors"
    new_path = tmp_path / "new.safetensors"
    _write_adapter(prior_path, b1, a1)
    _write_adapter(new_path, b2, a2)

    # Eval dataset where generated output never matches references →
    # measured winrate = 0.0 → drop = 0.9 (huge). But angle is high.
    eval_path = tmp_path / "eval.jsonl"
    _write_eval_dataset(
        eval_path,
        [
            {"prompt": "p1", "reference": "EXPECTED_REF_A"},
            {"prompt": "p2", "reference": "EXPECTED_REF_B"},
        ],
    )

    def fake_generate(prompt, adapter):  # noqa: ARG001
        return "nothing matches here"

    locator = _patch_generate_fn(monkeypatch, fake_generate)

    out_path = tmp_path / "report.json"
    rc = mod.main(
        [
            "--prior-adapter",
            str(prior_path),
            "--new-adapter",
            str(new_path),
            "--eval-dataset",
            str(eval_path),
            "--generate-fn-module",
            locator,
            "--winrate-baseline-score",
            "0.9",
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0, "high angle should pass the gate regardless of winrate drop"
    report = json.loads(out_path.read_text())
    assert report["gate_status"] == "pass"
    assert report["angle_degrees_mean"] > 30.0
    assert report["winrate_measured"] == 0.0
    assert report["winrate_drop"] > 0.03


def test_full_gate_fail_both(tmp_path: Path, monkeypatch) -> None:
    """angle < 30° AND winrate_drop > 0.03 → gate fails."""
    mod = _load_script_module()

    # Parallel deltas → very low angle.
    b1 = torch.tensor([[1.0], [0.0]])
    a1 = torch.tensor([[1.0, 0.0]])
    b2 = torch.tensor([[1.001], [0.0005]])
    a2 = torch.tensor([[1.0005, 0.0002]])

    prior_path = tmp_path / "prior.safetensors"
    new_path = tmp_path / "new.safetensors"
    _write_adapter(prior_path, b1, a1)
    _write_adapter(new_path, b2, a2)

    eval_path = tmp_path / "eval.jsonl"
    _write_eval_dataset(
        eval_path,
        [
            {"prompt": "p1", "reference": "EXPECTED_REF_A"},
            {"prompt": "p2", "reference": "EXPECTED_REF_B"},
        ],
    )

    def fake_generate(prompt, adapter):  # noqa: ARG001
        return "nothing matches here"

    locator = _patch_generate_fn(monkeypatch, fake_generate)

    out_path = tmp_path / "report.json"
    rc = mod.main(
        [
            "--prior-adapter",
            str(prior_path),
            "--new-adapter",
            str(new_path),
            "--eval-dataset",
            str(eval_path),
            "--generate-fn-module",
            locator,
            "--winrate-baseline-score",
            "0.9",
            "--output",
            str(out_path),
        ]
    )
    assert rc == 1, "low-angle + high winrate drop must fail the gate"
    report = json.loads(out_path.read_text())
    assert report["gate_status"] == "fail"
    assert report["angle_degrees_mean"] < 30.0
    assert report["winrate_drop"] > 0.03
    assert report["warning"]  # non-empty


def test_full_gate_pass_low_drop(tmp_path: Path, monkeypatch) -> None:
    """angle < 30° BUT winrate_drop < 0.03 → gate passes (AND-logic)."""
    mod = _load_script_module()

    # Parallel deltas → low angle.
    b1 = torch.tensor([[1.0], [0.0]])
    a1 = torch.tensor([[1.0, 0.0]])
    b2 = torch.tensor([[1.001], [0.0005]])
    a2 = torch.tensor([[1.0005, 0.0002]])

    prior_path = tmp_path / "prior.safetensors"
    new_path = tmp_path / "new.safetensors"
    _write_adapter(prior_path, b1, a1)
    _write_adapter(new_path, b2, a2)

    # Generated output matches every reference → winrate = 1.0, drop = -0.01.
    eval_path = tmp_path / "eval.jsonl"
    _write_eval_dataset(
        eval_path,
        [
            {"prompt": "p1", "reference": "match"},
            {"prompt": "p2", "reference": "match"},
        ],
    )

    def fake_generate(prompt, adapter):  # noqa: ARG001
        return "the string match appears here"

    locator = _patch_generate_fn(monkeypatch, fake_generate)

    out_path = tmp_path / "report.json"
    rc = mod.main(
        [
            "--prior-adapter",
            str(prior_path),
            "--new-adapter",
            str(new_path),
            "--eval-dataset",
            str(eval_path),
            "--generate-fn-module",
            locator,
            "--winrate-baseline-score",
            "0.99",
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0, "low angle but stable winrate should pass (AND-logic)"
    report = json.loads(out_path.read_text())
    assert report["gate_status"] == "pass"
    assert report["angle_degrees_mean"] < 30.0
    assert report["winrate_measured"] == 1.0
    assert report["winrate_drop"] <= 0.03


def test_partial_winrate_flags_rejected(tmp_path: Path) -> None:
    """Providing only some win-rate flags → exit 2 (CLI misuse)."""
    mod = _load_script_module()

    b1 = torch.tensor([[1.0], [0.0]])
    a1 = torch.tensor([[1.0, 0.0]])
    b2 = torch.tensor([[0.0], [1.0]])
    a2 = torch.tensor([[0.0, 1.0]])

    prior_path = tmp_path / "prior.safetensors"
    new_path = tmp_path / "new.safetensors"
    _write_adapter(prior_path, b1, a1)
    _write_adapter(new_path, b2, a2)

    rc = mod.main(
        [
            "--prior-adapter",
            str(prior_path),
            "--new-adapter",
            str(new_path),
            # Only baseline, missing dataset and module → misuse.
            "--winrate-baseline-score",
            "0.8",
        ]
    )
    assert rc == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
