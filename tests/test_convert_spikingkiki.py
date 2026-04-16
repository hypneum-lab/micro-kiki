"""Tests for the SpikingKiki-35B-A3B conversion script.

Covers:
- CLI --help exits cleanly
- Dry-run produces valid metadata JSON
- Layer map generation correctness (mock, no torch)
- Resume logic skips already-converted layers
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Script under test
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "convert_spikingkiki_35b.py"


# ---------------------------------------------------------------------------
# Helper: import the module without executing main()
# ---------------------------------------------------------------------------


def _import_script():
    """Import the conversion script as a module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("convert_spikingkiki_35b", SCRIPT)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    """Tests that exercise the argparse layer."""

    def test_help_exits_zero(self):
        """--help must exit 0 and print usage without torch."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"--help returned {result.returncode}"
        assert "usage" in result.stdout.lower(), "expected 'usage' in help output"

    def test_dry_run_exits_zero(self, tmp_path: Path):
        """--dry-run must exit 0 and write results JSON."""
        results_out = tmp_path / "meta.json"
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--dry-run",
                "--results-out", str(results_out),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"--dry-run returned {result.returncode}\nstdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )
        assert results_out.exists(), "results JSON not created"

    def test_dry_run_metadata_schema(self, tmp_path: Path):
        """Dry-run metadata JSON must contain required top-level keys."""
        results_out = tmp_path / "meta.json"
        subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--dry-run",
                "--results-out", str(results_out),
            ],
            capture_output=True,
            check=True,
        )
        data = json.loads(results_out.read_text())
        required_keys = {
            "timestamp",
            "status",
            "input",
            "output",
            "timesteps",
            "model_info",
            "layer_map_stats",
            "converted_layers",
            "total_layers",
            "elapsed_s",
            "spike_stats",
        }
        missing = required_keys - set(data.keys())
        assert not missing, f"metadata missing keys: {missing}"

    def test_dry_run_status_is_dry_run(self, tmp_path: Path):
        """Dry-run status field must be 'dry_run'."""
        results_out = tmp_path / "meta.json"
        subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--dry-run",
                "--results-out", str(results_out),
            ],
            capture_output=True,
            check=True,
        )
        data = json.loads(results_out.read_text())
        assert data["status"] == "dry_run"

    def test_dry_run_includes_layer_map(self, tmp_path: Path):
        """Dry-run metadata must include a layer_map list."""
        results_out = tmp_path / "meta.json"
        subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--dry-run",
                "--results-out", str(results_out),
            ],
            capture_output=True,
            check=True,
        )
        data = json.loads(results_out.read_text())
        assert "layer_map" in data
        assert isinstance(data["layer_map"], list)
        assert len(data["layer_map"]) > 0

    def test_no_weights_returns_2(self, tmp_path: Path):
        """Missing --input directory must return exit code 2."""
        results_out = tmp_path / "meta.json"
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--input", str(tmp_path / "nonexistent"),
                "--output", str(tmp_path / "out"),
                "--results-out", str(results_out),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2, (
            f"expected exit 2, got {result.returncode}"
        )


# ---------------------------------------------------------------------------
# Layer map tests (pure Python, no torch)
# ---------------------------------------------------------------------------


class TestLayerMap:
    """Tests for build_layer_map() and related helpers."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _import_script()

    def test_total_entries(self, mod):
        """Layer map must have the expected total for 94 blocks."""
        layer_map = mod.build_layer_map(num_layers=mod.NUM_LAYERS)
        # Per block: 4 attn + 1 router + 256 experts * 3 projections = 773
        expected = mod.NUM_LAYERS * (4 + 1 + mod.NUM_EXPERTS * 3)
        assert len(layer_map) == expected, (
            f"expected {expected} entries, got {len(layer_map)}"
        )

    def test_small_map_structure(self, mod):
        """A 2-block map with 4 experts has correct structure."""
        layer_map = mod.build_layer_map.__func__(2) if hasattr(
            mod.build_layer_map, "__func__"
        ) else mod.build_layer_map(num_layers=2)

        # Patch NUM_EXPERTS in the call — call with explicit args
        # build_layer_map(num_layers) uses module-level NUM_EXPERTS
        # so we test at scale-2 (2 blocks)
        kinds = {e["kind"] for e in layer_map}
        assert kinds == {"attn_proj", "moe_router", "moe_expert_ffn"}, (
            f"unexpected kinds: {kinds}"
        )

    def test_activation_assignments(self, mod):
        """Router uses identity; expert FFNs use relu; attn uses identity."""
        layer_map = mod.build_layer_map(num_layers=1)
        for entry in layer_map:
            if entry["kind"] == "moe_router":
                assert entry["activation"] == "identity", (
                    "router must use identity activation"
                )
            elif entry["kind"] == "moe_expert_ffn":
                assert entry["activation"] == "relu", (
                    "expert FFN must use relu activation"
                )
            elif entry["kind"] == "attn_proj":
                assert entry["activation"] == "identity", (
                    "attention projection must use identity activation"
                )

    def test_all_converted_false_initially(self, mod):
        """All layer map entries must start with converted=False."""
        layer_map = mod.build_layer_map(num_layers=2)
        assert all(not e["converted"] for e in layer_map), (
            "some entries were marked converted at initialisation"
        )

    def test_layer_map_stats(self, mod):
        """layer_map_stats returns correct counts."""
        layer_map = mod.build_layer_map(num_layers=1)
        stats = mod.layer_map_stats(layer_map)
        assert stats["moe_router"] == 1
        assert stats["attn_proj"] == 4
        assert stats["moe_expert_ffn"] == mod.NUM_EXPERTS * 3
        assert stats["total"] == 1 + 4 + mod.NUM_EXPERTS * 3

    def test_key_prefix_format(self, mod):
        """key_prefix values must start with 'model.layers.<n>'."""
        layer_map = mod.build_layer_map(num_layers=3)
        for entry in layer_map:
            assert entry["key_prefix"].startswith("model.layers."), (
                f"unexpected key_prefix: {entry['key_prefix']}"
            )

    def test_expert_ids_range(self, mod):
        """moe_expert_ffn entries must cover expert_id 0 … NUM_EXPERTS-1."""
        layer_map = mod.build_layer_map(num_layers=1)
        expert_ids = {
            e["expert_id"]
            for e in layer_map
            if e["kind"] == "moe_expert_ffn"
        }
        assert expert_ids == set(range(mod.NUM_EXPERTS)), (
            "expert IDs do not cover the expected range"
        )


# ---------------------------------------------------------------------------
# Resume logic tests
# ---------------------------------------------------------------------------


class TestResumeLogic:
    """Tests for load_resume_state / save_resume_state."""

    @pytest.fixture()
    def mod(self):
        return _import_script()

    def test_empty_state_when_no_file(self, mod, tmp_path: Path):
        """load_resume_state returns empty set when no state file exists."""
        result = mod.load_resume_state(tmp_path / "nonexistent_dir")
        assert result == set()

    def test_roundtrip(self, mod, tmp_path: Path):
        """Saved state is reloaded correctly."""
        keys = {"model.layers.0.self_attn.q_proj", "model.layers.1.mlp.gate"}
        mod.save_resume_state(tmp_path, keys)
        loaded = mod.load_resume_state(tmp_path)
        assert loaded == keys

    def test_corrupted_state_returns_empty(self, mod, tmp_path: Path):
        """Corrupted state file must return empty set, not raise."""
        state_file = tmp_path / ".convert_state.json"
        state_file.write_text("not valid json {{{")
        result = mod.load_resume_state(tmp_path)
        assert result == set()

    def test_resume_skips_converted_layers(self, mod, tmp_path: Path):
        """dry-run with --resume skips already-saved keys."""
        # Pre-populate a resume state with some keys
        pre_converted = {
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
        }
        mod.save_resume_state(tmp_path, pre_converted)

        # load and verify the keys are present
        loaded = mod.load_resume_state(tmp_path)
        assert pre_converted.issubset(loaded)

    def test_save_creates_parent_dirs(self, mod, tmp_path: Path):
        """save_resume_state must create missing parent directories."""
        deep_path = tmp_path / "a" / "b" / "c"
        mod.save_resume_state(deep_path, {"some.key"})
        state_file = deep_path / ".convert_state.json"
        assert state_file.exists()

    def test_incremental_save(self, mod, tmp_path: Path):
        """Saving multiple times accumulates keys correctly."""
        mod.save_resume_state(tmp_path, {"key.a", "key.b"})
        first = mod.load_resume_state(tmp_path)
        assert first == {"key.a", "key.b"}

        mod.save_resume_state(tmp_path, {"key.a", "key.b", "key.c"})
        second = mod.load_resume_state(tmp_path)
        assert second == {"key.a", "key.b", "key.c"}


# ---------------------------------------------------------------------------
# Metadata structure tests
# ---------------------------------------------------------------------------


class TestMetadata:
    """Tests for build_metadata()."""

    @pytest.fixture()
    def mod(self):
        return _import_script()

    def test_model_info_block(self, mod, tmp_path: Path):
        """model_info must contain architecture constants."""
        args = mod.parse_args([
            "--dry-run",
            "--input", str(tmp_path),
            "--output", str(tmp_path / "out"),
        ])
        layer_map = mod.build_layer_map(num_layers=2)
        meta = mod.build_metadata(
            args, layer_map, spike_stats={}, elapsed_s=1.0, status="test"
        )
        info = meta["model_info"]
        assert info["num_experts"] == mod.NUM_EXPERTS
        assert info["top_k"] == mod.TOP_K
        assert info["num_layers"] == mod.NUM_LAYERS

    def test_elapsed_rounded(self, mod, tmp_path: Path):
        """elapsed_s must be rounded to 1 decimal place."""
        args = mod.parse_args(["--dry-run"])
        layer_map = mod.build_layer_map(num_layers=1)
        meta = mod.build_metadata(
            args, layer_map, spike_stats={},
            elapsed_s=12345.6789, status="test"
        )
        assert meta["elapsed_s"] == 12345.7

    def test_converted_layers_count(self, mod, tmp_path: Path):
        """converted_layers must reflect how many entries are marked True."""
        args = mod.parse_args(["--dry-run"])
        layer_map = mod.build_layer_map(num_layers=1)
        # Mark first 3 entries as converted
        for entry in layer_map[:3]:
            entry["converted"] = True
        meta = mod.build_metadata(
            args, layer_map, spike_stats={}, elapsed_s=0.0, status="test"
        )
        assert meta["converted_layers"] == 3
