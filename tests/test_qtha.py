from __future__ import annotations

from src.stacks.qtha import QTHAConfig, estimate_qtha_params


def test_default_config():
    cfg = QTHAConfig()
    assert cfg.bond_dim == 8
    assert cfg.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_config_frozen():
    cfg = QTHAConfig()
    try:
        cfg.bond_dim = 16
        assert False, "should be frozen"
    except AttributeError:
        pass


def test_custom_bond_dim():
    cfg = QTHAConfig(bond_dim=16)
    assert cfg.bond_dim == 16


def test_estimate_params_default():
    params = estimate_qtha_params(hidden_dim=3584, bond_dim=8, num_layers=28)
    assert params == 8 * 3584 * 4 * 28
    assert params < 5_000_000  # should be well under 5M


def test_estimate_params_higher_bond():
    p8 = estimate_qtha_params(3584, 8, 28)
    p16 = estimate_qtha_params(3584, 16, 28)
    assert p16 == 2 * p8


def test_qtha_much_smaller_than_lora():
    """QTHA bond-8 should be ~3-4x smaller than LoRA rank-16."""
    qtha = estimate_qtha_params(3584, 8, 28, num_modules=4)
    lora = 2 * 16 * 3584 * 4 * 28  # A + B per module
    assert qtha < lora
    ratio = lora / qtha
    assert 2.5 < ratio < 5.0


def test_pilot_script_help():
    import subprocess
    result = subprocess.run(
        ["uv", "run", "python", "scripts/train_qtha_stack.py", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "qtha" in result.stdout.lower() or "bond" in result.stdout.lower()


def test_pilot_dry_run():
    import subprocess
    result = subprocess.run(
        ["uv", "run", "python", "scripts/train_qtha_stack.py", "--domain", "reasoning",
         "--output", "/tmp/qtha-test.json"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    import json
    from pathlib import Path
    report = json.loads(Path("/tmp/qtha-test.json").read_text())
    assert report["domain"] == "reasoning"
    assert "parameter_comparison" in report
