"""Unit tests for the ``--consolidate-on-warning`` branch of post_train_gate.

Patches the subprocess calls so neither ``validate_adapter_health`` nor
``measure_forgetting`` actually execute — the first is forced to pass, the
second is forced to fail. The test then exercises the consolidation branch
and asserts graceful fallback to exit 2 when ``kiki_oniric`` is unavailable
(by monkey-patching ``sys.modules`` so the lazy import raises ImportError).

On hosts where ``kiki_oniric`` *is* installed (kxkm-ai via
``/tmp/dream-of-kiki``), this test still produces exit 2 because we actively
shadow the package import to ``None``, forcing the ImportError branch.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "post_train_gate.py"


def _load_module():
    """Import ``scripts/post_train_gate.py`` as a stand-alone module."""
    spec = importlib.util.spec_from_file_location(
        "post_train_gate_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None, SCRIPT_PATH
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_consolidate_on_warning_falls_back_when_kiki_oniric_missing(
    tmp_path, monkeypatch, capsys
):
    """Gate failure + missing kiki_oniric => exit 2 with a clear warning."""
    module = _load_module()

    # Shadow the kiki_oniric package so the lazy import inside
    # _attempt_consolidation raises ImportError even on hosts where it is
    # installed. Setting the module to None is the canonical way to force an
    # ImportError for a specific dotted path.
    for name in (
        "kiki_oniric",
        "kiki_oniric.dream",
        "kiki_oniric.dream.episode",
        "kiki_oniric.dream.runtime",
        "kiki_oniric.substrates",
        "kiki_oniric.substrates.micro_kiki",
    ):
        monkeypatch.setitem(sys.modules, name, None)

    # Fake adapter files — ``_run`` is patched below so contents don't matter.
    new_adapter = tmp_path / "new" / "adapters.safetensors"
    new_adapter.parent.mkdir(parents=True)
    new_adapter.write_bytes(b"\x00")
    prior_adapter = tmp_path / "prior" / "adapters.safetensors"
    prior_adapter.parent.mkdir(parents=True)
    prior_adapter.write_bytes(b"\x00")

    output_dir = tmp_path / "results"

    # Patch _run so step 1 (health) passes and step 2 (forgetting) fails.
    # Also write a minimal report JSON at the path step 2 would have written
    # so the consolidation helper can json.loads() it.
    gate_report = output_dir / f"gate-{new_adapter.parent.name}.json"
    call_log: list[list[str]] = []

    def fake_run(cmd):
        call_log.append(cmd)
        last = cmd[1] if len(cmd) > 1 else ""
        if "validate_adapter_health" in last:
            return 0, "adapter health OK\n"
        if "measure_forgetting" in last:
            output_dir.mkdir(parents=True, exist_ok=True)
            gate_report.write_text(
                '{"angle_degrees_mean": 12.3, '
                '"gate_status_aggregate": "fail"}\n'
            )
            return 1, "forgetting gate failed (angle 12.3 < 30)\n"
        return 0, ""

    monkeypatch.setattr(module, "_run", fake_run)

    snapshot_path = tmp_path / "substrate.npz"  # intentionally does not exist

    argv = [
        "--new-adapter",
        str(new_adapter),
        "--prior-adapter",
        str(prior_adapter),
        "--output-dir",
        str(output_dir),
        "--consolidate-on-warning",
        "--substrate-snapshot",
        str(snapshot_path),
    ]

    rc = module.main(argv)

    captured = capsys.readouterr()
    combined = captured.out + captured.err

    assert rc == 2, (
        f"expected exit 2 (rollback fallback), got {rc}\n{combined}"
    )
    assert "--consolidate-on-warning set" in combined
    assert "dream-of-kiki package not installed" in combined
    assert "falling back to rollback behaviour" in combined
    assert not snapshot_path.exists(), (
        "snapshot must not be written on fallback path"
    )

    # Sanity: both subprocess steps were invoked in the expected order.
    assert any("validate_adapter_health" in c[1] for c in call_log)
    assert any("measure_forgetting" in c[1] for c in call_log)


def test_consolidate_on_warning_off_by_default_still_exits_2(
    tmp_path, monkeypatch, capsys
):
    """Without the flag, a gate failure must still exit 2 (unchanged behaviour)."""
    module = _load_module()

    new_adapter = tmp_path / "new" / "adapters.safetensors"
    new_adapter.parent.mkdir(parents=True)
    new_adapter.write_bytes(b"\x00")
    prior_adapter = tmp_path / "prior" / "adapters.safetensors"
    prior_adapter.parent.mkdir(parents=True)
    prior_adapter.write_bytes(b"\x00")

    output_dir = tmp_path / "results"

    def fake_run(cmd):
        last = cmd[1] if len(cmd) > 1 else ""
        if "validate_adapter_health" in last:
            return 0, ""
        if "measure_forgetting" in last:
            return 1, "gate failed\n"
        return 0, ""

    monkeypatch.setattr(module, "_run", fake_run)

    rc = module.main(
        [
            "--new-adapter",
            str(new_adapter),
            "--prior-adapter",
            str(prior_adapter),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert rc == 2, f"expected exit 2, got {rc}\n{combined}"
    # No consolidation messages should appear in the default path.
    assert "--consolidate-on-warning set" not in combined
    assert "dream-of-kiki package not installed" not in combined


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
