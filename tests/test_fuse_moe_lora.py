"""Test fuse_moe_lora batch mode argument parsing."""
import subprocess
import sys


def test_fuse_all_stacks_flag_exists():
    """--all-stacks flag should be accepted."""
    result = subprocess.run(
        [sys.executable, "scripts/fuse_moe_lora.py", "--help"],
        capture_output=True, text=True,
    )
    assert "--all-stacks" in result.stdout or "--all-stacks" in result.stderr


def test_build_key_mapping_import():
    """build_key_mapping should be importable."""
    sys.path.insert(0, "scripts")
    from fuse_moe_lora import build_key_mapping
    # Empty inputs should return empty mapping
    result = build_key_mapping([], set())
    assert result == {}
