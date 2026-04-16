"""Tests for the energy estimator (story-32).

Validates theoretical FLOPs-to-spikes energy estimation, per-layer
and aggregate computation, and JSON metadata ingestion.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.spiking.energy_estimator import (
    EnergyEstimator,
    LayerEnergy,
    estimate_from_metadata,
)


class TestLayerEnergy:
    """Tests for per-layer energy computation."""

    def test_basic_computation(self) -> None:
        est = EnergyEstimator()
        result = est.estimate_layer("fc1", 64, 128, active_ratio=0.5)
        assert result.mac_count == 2 * 64 * 128
        assert result.spike_ops == int(result.mac_count * 0.5)
        assert result.energy_ratio == 0.5
        assert result.energy_saving_pct == 50.0

    def test_fully_dense(self) -> None:
        est = EnergyEstimator()
        result = est.estimate_layer("fc1", 32, 32, active_ratio=1.0)
        assert result.energy_ratio == 1.0
        assert result.energy_saving_pct == 0.0

    def test_fully_sparse(self) -> None:
        est = EnergyEstimator()
        result = est.estimate_layer("fc1", 32, 32, active_ratio=0.0)
        assert result.spike_ops == 0
        assert result.energy_saving_pct == 100.0

    def test_invalid_ratio_rejected(self) -> None:
        est = EnergyEstimator()
        with pytest.raises(ValueError, match="active_ratio"):
            est.estimate_layer("bad", 8, 8, active_ratio=1.5)

    def test_to_dict(self) -> None:
        est = EnergyEstimator()
        result = est.estimate_layer("fc1", 16, 32, active_ratio=0.3)
        d = result.to_dict()
        assert d["name"] == "fc1"
        assert d["in_features"] == 16
        assert d["out_features"] == 32
        assert isinstance(d["energy_ratio"], float)


class TestEnergyEstimator:
    """Tests for full model energy estimation."""

    def test_two_layer_model(self) -> None:
        est = EnergyEstimator()
        layers = [
            {"name": "fc1", "in_features": 64, "out_features": 128,
             "active_ratio": 0.4},
            {"name": "fc2", "in_features": 128, "out_features": 32,
             "active_ratio": 0.6},
        ]
        results = est.estimate_model(layers)
        assert len(results["per_layer"]) == 2
        agg = results["aggregate"]
        assert agg["total_mac_count"] > 0
        assert agg["total_spike_ops"] > 0
        assert 0.0 < agg["energy_ratio"] < 1.0
        assert agg["energy_saving_pct"] > 0.0

    def test_aggregate_macs_sum_correctly(self) -> None:
        est = EnergyEstimator()
        layers = [
            {"name": "a", "in_features": 8, "out_features": 16,
             "active_ratio": 0.5},
            {"name": "b", "in_features": 16, "out_features": 4,
             "active_ratio": 0.5},
        ]
        results = est.estimate_model(layers)
        expected_macs = 2 * 8 * 16 + 2 * 16 * 4
        assert results["aggregate"]["total_mac_count"] == expected_macs

    def test_energy_nj_computed(self) -> None:
        est = EnergyEstimator(mac_energy_pj=4.6, spike_energy_pj=0.9)
        layers = [
            {"name": "fc1", "in_features": 100, "out_features": 100,
             "active_ratio": 0.5},
        ]
        results = est.estimate_model(layers)
        agg = results["aggregate"]
        # ANN: 20000 MACs * 4.6 pJ / 1000 = 92 nJ
        assert abs(agg["ann_energy_nj"] - 92.0) < 0.1
        # SNN: 10000 ops * 0.9 pJ / 1000 = 9.0 nJ
        assert abs(agg["snn_energy_nj"] - 9.0) < 0.1

    def test_estimate_from_shapes(self) -> None:
        est = EnergyEstimator()
        shapes = [(64, 128), (128, 64), (64, 32)]
        ratios = [0.3, 0.5, 0.7]
        results = est.estimate_from_shapes(shapes, ratios)
        assert len(results["per_layer"]) == 3
        # Check layer names auto-generated
        assert results["per_layer"][0]["name"] == "layer_0"

    def test_shapes_default_ratios(self) -> None:
        est = EnergyEstimator()
        results = est.estimate_from_shapes([(8, 16)])
        assert results["per_layer"][0]["active_ratio"] == 0.5

    def test_shapes_mismatch_rejected(self) -> None:
        est = EnergyEstimator()
        with pytest.raises(ValueError, match="length"):
            est.estimate_from_shapes([(8, 16)], [0.5, 0.3])


class TestEstimateFromMetadata:
    """Tests for JSON metadata ingestion."""

    def test_flat_list_metadata(self) -> None:
        metadata = [
            {"name": "layer_0", "in_features": 64,
             "out_features": 128, "active_ratio": 0.4},
            {"name": "layer_1", "in_features": 128,
             "out_features": 32, "spike_rate": 0.3},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(metadata, f)
            f.flush()
            results = estimate_from_metadata(f.name)

        assert len(results["per_layer"]) == 2
        assert results["per_layer"][1]["active_ratio"] == 0.3

    def test_nested_dict_metadata(self) -> None:
        metadata = {
            "model": "SpikingKiki-27B",
            "layers": [
                {"name": "attn_0", "in_features": 2048,
                 "out_features": 2048, "active_ratio": 0.35},
                {"name": "mlp_0", "in_features": 2048,
                 "out_features": 8192, "active_ratio": 0.45},
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(metadata, f)
            f.flush()
            results = estimate_from_metadata(f.name)

        assert len(results["per_layer"]) == 2
        agg = results["aggregate"]
        assert agg["energy_saving_pct"] > 0.0

    def test_output_written(self) -> None:
        metadata = [
            {"name": "fc", "in_features": 8,
             "out_features": 8, "active_ratio": 0.5},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = Path(tmpdir) / "meta.json"
            out_path = Path(tmpdir) / "energy.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f)
            results = estimate_from_metadata(meta_path, out_path)
            assert out_path.exists()
            with open(out_path) as f:
                loaded = json.load(f)
            assert "aggregate" in loaded

    def test_three_spikingkiki_models(self) -> None:
        """Simulate energy estimation for 3 SpikingKiki models."""
        for model_name, layers_data in [
            ("SpikingKiki-27B", [
                {"name": "attn", "in_features": 3584,
                 "out_features": 3584, "active_ratio": 0.32},
                {"name": "mlp", "in_features": 3584,
                 "out_features": 18944, "active_ratio": 0.41},
            ]),
            ("SpikingKiki-122B", [
                {"name": "attn", "in_features": 5120,
                 "out_features": 5120, "active_ratio": 0.28},
                {"name": "mlp", "in_features": 5120,
                 "out_features": 13824, "active_ratio": 0.35},
            ]),
            ("SpikingKiki-LargeOpus", [
                {"name": "attn", "in_features": 6144,
                 "out_features": 6144, "active_ratio": 0.30},
                {"name": "mlp", "in_features": 6144,
                 "out_features": 16384, "active_ratio": 0.38},
            ]),
        ]:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(layers_data, f)
                f.flush()
                results = estimate_from_metadata(f.name)

            agg = results["aggregate"]
            assert agg["total_mac_count"] > 0
            assert agg["energy_saving_pct"] > 50.0, (
                f"{model_name}: saving {agg['energy_saving_pct']:.1f}%"
            )
