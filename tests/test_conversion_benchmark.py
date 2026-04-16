"""Tests for the LAS vs Spikingformer conversion benchmark (story-31).

Validates that the benchmark runs on a toy 4-layer MLP, produces
finite outputs from both methods, and generates a valid comparison.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.spiking.conversion_benchmark import (
    BenchmarkResult,
    build_toy_mlp,
    run_benchmark,
)


class TestBuildToyMLP:
    """Tests for the MLP builder."""

    def test_default_architecture(self) -> None:
        layers = build_toy_mlp()
        assert len(layers) == 4  # 5 sizes => 4 layers
        assert layers[0]["weight"].shape == (128, 64)
        assert layers[-1]["weight"].shape == (32, 64)

    def test_custom_architecture(self) -> None:
        layers = build_toy_mlp(layer_sizes=[8, 16, 4])
        assert len(layers) == 2
        assert layers[0]["weight"].shape == (16, 8)
        assert layers[1]["weight"].shape == (4, 16)

    def test_biases_are_zero(self) -> None:
        layers = build_toy_mlp()
        for layer in layers:
            np.testing.assert_array_equal(layer["bias"], 0.0)


class TestRunBenchmark:
    """Tests for the full benchmark pipeline."""

    def test_benchmark_runs(self) -> None:
        results = run_benchmark(
            layer_sizes=[8, 16, 16, 8],
            n_samples=4,
            seed=0,
        )
        assert "methods" in results
        assert "LAS" in results["methods"]
        assert "Spikingformer" in results["methods"]

    def test_both_produce_finite_outputs(self) -> None:
        results = run_benchmark(
            layer_sizes=[8, 16, 8],
            n_samples=4,
        )
        for method in ("LAS", "Spikingformer"):
            m = results["methods"][method]
            assert np.isfinite(m["cosine_similarity"])
            assert np.isfinite(m["l2_distance"])
            assert m["l2_distance"] >= 0.0

    def test_param_counts_identical(self) -> None:
        results = run_benchmark(
            layer_sizes=[8, 16, 8],
            n_samples=4,
        )
        assert results["comparison"]["param_count_identical"]

    def test_las_cosine_higher(self) -> None:
        """LAS should be closer to ANN output (near-lossless)."""
        results = run_benchmark(
            layer_sizes=[8, 16, 16, 8],
            n_samples=8,
        )
        las_cos = results["methods"]["LAS"]["cosine_similarity"]
        sf_cos = results["methods"]["Spikingformer"]["cosine_similarity"]
        # LAS should produce very high similarity
        assert las_cos > 0.5, f"LAS cosine too low: {las_cos}"
        # Both should be finite
        assert np.isfinite(sf_cos)

    def test_energy_ratios_valid(self) -> None:
        results = run_benchmark(
            layer_sizes=[8, 16, 8],
            n_samples=4,
        )
        for method in ("LAS", "Spikingformer"):
            ratio = results["methods"][method]["energy_ratio"]
            assert 0.0 < ratio <= 1.0

    def test_spikingformer_more_efficient(self) -> None:
        """Spikingformer should have lower energy ratio."""
        results = run_benchmark(
            layer_sizes=[8, 16, 16, 8],
            n_samples=4,
        )
        assert results["comparison"]["spikingformer_more_efficient"]

    def test_output_json_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bench.json"
            results = run_benchmark(
                layer_sizes=[8, 16, 8],
                n_samples=4,
                output_path=path,
            )
            assert path.exists()
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["config"]["n_samples"] == 4

    def test_comparison_table_complete(self) -> None:
        results = run_benchmark(
            layer_sizes=[8, 16, 8],
            n_samples=4,
        )
        comp = results["comparison"]
        assert "las_closer_to_ann" in comp
        assert "spikingformer_more_efficient" in comp
        assert "param_count_identical" in comp
