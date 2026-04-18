"""Tests for ``ForgettingEvaluator.check_all_previous`` automatic mode.

Phase 1b wires the automatic path (previously raised ``NotImplementedError``)
to loop over prior stacks and call ``measure_forgetting_signal`` for each.
All tests use mocked generate_fn closures; no real models.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Adapter math is PyTorch-backed.
torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

from src.eval.forgetting import (  # noqa: E402
    ForgettingEvaluator,
    GradientSubspaceAnalyzer,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _adapter_tensors(b: torch.Tensor, a: torch.Tensor) -> dict[str, torch.Tensor]:
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


def _orthogonal_adapter(path: Path) -> None:
    # delta = [[0,0],[0,1]]
    _write_adapter(path, torch.tensor([[0.0], [1.0]]), torch.tensor([[0.0, 1.0]]))


def _parallel_adapter(path: Path, jitter: float = 0.0) -> None:
    b = torch.tensor([[1.0 + jitter], [0.0]])
    a = torch.tensor([[1.0, 0.0]])
    _write_adapter(path, b, a)


# ---------------------------------------------------------------------------
# check_all_previous no longer raises NotImplementedError
# ---------------------------------------------------------------------------


def test_check_all_previous_no_longer_raises(tmp_path: Path) -> None:
    """Automatic mode must not raise NotImplementedError (phase 1b)."""
    fe = ForgettingEvaluator(
        stack_evaluator=None,
        analyzer=GradientSubspaceAnalyzer(),
    )

    new_adapter = tmp_path / "new.safetensors"
    _orthogonal_adapter(new_adapter)

    prior_dir = tmp_path / "priors"
    prior_dir.mkdir()
    for sid in ("stack-01", "stack-02"):
        _parallel_adapter(prior_dir / f"{sid}.safetensors")

    reports = fe.check_all_previous(
        trained_stacks=["stack-01", "stack-02"],
        new_stack_id="stack-03",
        adapter_path_fn=lambda sid: prior_dir / f"{sid}.safetensors",
        new_adapter_path=new_adapter,
    )
    # Angle-only mode (no generate_fn) → angle_only_partial → passed True.
    assert len(reports) == 2
    assert all(r.new_stack_id == "stack-03" for r in reports)
    assert all(r.passed for r in reports)


# ---------------------------------------------------------------------------
# Aggregates across 2-3 mocked prior stacks
# ---------------------------------------------------------------------------


def test_check_all_previous_aggregates_pass_and_fail(tmp_path: Path) -> None:
    """Mix of orthogonal (safe) and parallel (risky) priors — only the risky
    one with a large winrate drop should rollback."""
    fe = ForgettingEvaluator(
        stack_evaluator=None,
        analyzer=GradientSubspaceAnalyzer(),
    )

    new_adapter = tmp_path / "new.safetensors"
    # New adapter is parallel to a "risky" prior.
    _parallel_adapter(new_adapter)

    prior_dir = tmp_path / "priors"
    prior_dir.mkdir()

    # stack-01: orthogonal → high angle → always passes.
    _orthogonal_adapter(prior_dir / "stack-01.safetensors")
    # stack-02: parallel → low angle → passes only if winrate stable.
    _parallel_adapter(prior_dir / "stack-02.safetensors")
    # stack-03: parallel → low angle → will be paired with a high winrate drop.
    _parallel_adapter(prior_dir / "stack-03.safetensors")

    eval_dir = tmp_path / "evals"
    eval_dir.mkdir()
    for sid in ("stack-01", "stack-02", "stack-03"):
        _write_eval_dataset(
            eval_dir / f"{sid}.jsonl",
            [
                {"prompt": "p1", "reference": "match"},
                {"prompt": "p2", "reference": "match"},
            ],
        )

    # generate_fn: returns "match" for stack-01 & stack-02, nonsense for stack-03
    # via the adapter path (which is the NEW adapter — same for all calls).
    # So we emulate the per-stack differential via a shared counter: each
    # call advances one prompt. We instead key on the prior stack by
    # inspecting the current winrate_baselines below — simpler to control
    # via baselines themselves:
    #   - stack-01 baseline 0.0  -> measured=1.0 -> drop=-1.0 (no regression)
    #   - stack-02 baseline 1.0  -> measured=1.0 -> drop= 0.0 (no regression)
    #   - stack-03 baseline 1.0  -> measured=0.0 -> drop= 1.0 (regression)
    # We make the generate_fn return "match" when the eval dataset contains
    # prompt "p1" (stack-01/02) and nonsense otherwise. Since all three
    # datasets share the same content, we need a different lever.
    #
    # Use a closure that consults a mutable bucket driven by the baseline
    # map — but measure_forgetting_signal calls generate_fn per-prompt with
    # (prompt, adapter). We don't know which prior we're on.
    #
    # Easiest: monkeypatch — write DIFFERENT eval files with references
    # unique per prior, and have generate_fn match only some.
    _write_eval_dataset(
        eval_dir / "stack-01.jsonl",
        [{"prompt": "p", "reference": "alpha"}],
    )
    _write_eval_dataset(
        eval_dir / "stack-02.jsonl",
        [{"prompt": "p", "reference": "beta"}],
    )
    _write_eval_dataset(
        eval_dir / "stack-03.jsonl",
        [{"prompt": "p", "reference": "gamma"}],
    )

    def fake_generate(prompt, adapter):  # noqa: ARG001
        # Always returns "alpha beta" — matches stack-01 and stack-02 refs
        # but NOT stack-03 ("gamma" not in output).
        return "alpha beta"

    reports = fe.check_all_previous(
        trained_stacks=["stack-01", "stack-02", "stack-03"],
        new_stack_id="stack-04",
        eval_data_dir=eval_dir,
        adapter_path_fn=lambda sid: prior_dir / f"{sid}.safetensors",
        new_adapter_path=new_adapter,
        generate_fn=fake_generate,
        winrate_baselines={
            "stack-01": 1.0,  # measured=1.0, drop=0.0 → safe (also orthogonal)
            "stack-02": 1.0,  # measured=1.0, drop=0.0 → safe
            "stack-03": 1.0,  # measured=0.0, drop=1.0 → regression
        },
    )

    assert len(reports) == 3
    by_id = {r.stack_id: r for r in reports}

    # stack-01: orthogonal (high angle) → pass regardless.
    assert by_id["stack-01"].passed is True
    assert by_id["stack-01"].should_rollback is False

    # stack-02: low angle but no winrate drop → pass (AND-logic).
    assert by_id["stack-02"].passed is True
    assert by_id["stack-02"].should_rollback is False

    # stack-03: low angle AND winrate drop → rollback.
    assert by_id["stack-03"].passed is False
    assert by_id["stack-03"].should_rollback is True
    assert by_id["stack-03"].winrate_drop > 0.03


def test_check_all_previous_precomputed_mode_still_works(tmp_path: Path) -> None:
    """Pre-computed ``results=[...]`` mode is the legacy path and must still
    work without touching the filesystem."""
    fe = ForgettingEvaluator(
        stack_evaluator=None,
        analyzer=GradientSubspaceAnalyzer(),
    )

    reports = fe.check_all_previous(
        trained_stacks=["stack-01", "stack-02"],
        new_stack_id="stack-03",
        results=[
            {
                "stack_id": "stack-01",
                "angle": 50.0,
                "winrate_base": 0.80,
                "winrate_adapted": 0.79,
            },
            {
                "stack_id": "stack-02",
                "angle": 25.0,
                "winrate_base": 0.82,
                "winrate_adapted": 0.74,
            },
        ],
    )
    assert len(reports) == 2
    assert reports[0].passed is True
    assert reports[1].should_rollback is True


def test_check_all_previous_requires_adapter_fn_in_auto_mode() -> None:
    """Automatic mode without adapter_path_fn / new_adapter_path → ValueError."""
    fe = ForgettingEvaluator(
        stack_evaluator=None,
        analyzer=GradientSubspaceAnalyzer(),
    )
    with pytest.raises(ValueError):
        fe.check_all_previous(
            trained_stacks=["stack-01"],
            new_stack_id="stack-02",
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
