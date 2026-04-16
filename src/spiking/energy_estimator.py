"""Energy estimator — theoretical FLOPs to spikes conversion.

Story-32 implementation. Computes theoretical energy savings from
spiking neural network execution vs standard ANN inference.

Methodology:
- ANN energy ~ MAC count = 2 * in_features * out_features per layer
- SNN energy ~ spike-op count = active_ratio * MAC count
  (only non-zero activations trigger computation)
- Energy ratio = spike_ops / mac_count (< 1.0 means savings)

The estimator reads layer shapes and activation sparsity stats
(e.g., from LIF metadata) and computes per-layer + aggregate
energy estimates.

Public surface:

- :class:`EnergyEstimator` — main estimator class
- :func:`estimate_from_metadata` — convenience for JSON metadata
- :dataclass:`LayerEnergy` — per-layer energy breakdown
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "EnergyEstimator",
    "LayerEnergy",
    "estimate_from_metadata",
]


@dataclass
class LayerEnergy:
    """Per-layer energy breakdown."""

    name: str
    in_features: int
    out_features: int
    mac_count: int
    active_ratio: float  # fraction of non-zero activations
    spike_ops: int  # active_ratio * mac_count
    energy_ratio: float  # spike_ops / mac_count
    energy_saving_pct: float  # (1 - energy_ratio) * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "mac_count": self.mac_count,
            "active_ratio": self.active_ratio,
            "spike_ops": self.spike_ops,
            "energy_ratio": self.energy_ratio,
            "energy_saving_pct": self.energy_saving_pct,
        }


@dataclass
class EnergyEstimator:
    """Theoretical energy estimator for ANN vs SNN comparison.

    Parameters
    ----------
    mac_energy_pj : float
        Energy per MAC operation in picojoules. Default 4.6 pJ
        (typical 45nm CMOS, Horowitz 2014).
    spike_energy_pj : float
        Energy per spike accumulate operation in picojoules.
        Default 0.9 pJ (neuromorphic hardware estimate).
    """

    mac_energy_pj: float = 4.6
    spike_energy_pj: float = 0.9

    def estimate_layer(
        self,
        name: str,
        in_features: int,
        out_features: int,
        active_ratio: float = 0.5,
    ) -> LayerEnergy:
        """Estimate energy for a single layer.

        Parameters
        ----------
        name : str
            Layer identifier.
        in_features : int
            Input dimension.
        out_features : int
            Output dimension.
        active_ratio : float
            Fraction of non-zero activations (0.0 = fully sparse,
            1.0 = fully dense). Default 0.5 for ReLU networks.

        Returns
        -------
        LayerEnergy with computed metrics.
        """
        if active_ratio < 0.0 or active_ratio > 1.0:
            raise ValueError(
                f"active_ratio must be in [0, 1], got {active_ratio}"
            )

        mac_count = 2 * in_features * out_features
        spike_ops = int(mac_count * active_ratio)
        energy_ratio = active_ratio  # spike_ops / mac_count
        saving_pct = (1.0 - energy_ratio) * 100.0

        return LayerEnergy(
            name=name,
            in_features=in_features,
            out_features=out_features,
            mac_count=mac_count,
            active_ratio=active_ratio,
            spike_ops=spike_ops,
            energy_ratio=energy_ratio,
            energy_saving_pct=saving_pct,
        )

    def estimate_model(
        self,
        layers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Estimate energy for a full model.

        Parameters
        ----------
        layers : list[dict]
            Each dict has keys: ``name``, ``in_features``,
            ``out_features``, ``active_ratio``.

        Returns
        -------
        dict with per-layer results and aggregate summary.
        """
        results = []
        total_macs = 0
        total_spike_ops = 0

        for layer_spec in layers:
            layer_energy = self.estimate_layer(
                name=layer_spec["name"],
                in_features=layer_spec["in_features"],
                out_features=layer_spec["out_features"],
                active_ratio=layer_spec.get("active_ratio", 0.5),
            )
            results.append(layer_energy)
            total_macs += layer_energy.mac_count
            total_spike_ops += layer_energy.spike_ops

        aggregate_ratio = (
            total_spike_ops / total_macs if total_macs > 0 else 1.0
        )

        # Absolute energy in nanojoules
        ann_energy_nj = total_macs * self.mac_energy_pj / 1000.0
        snn_energy_nj = total_spike_ops * self.spike_energy_pj / 1000.0
        actual_ratio = snn_energy_nj / ann_energy_nj if ann_energy_nj > 0 else 1.0

        return {
            "per_layer": [r.to_dict() for r in results],
            "aggregate": {
                "total_mac_count": total_macs,
                "total_spike_ops": total_spike_ops,
                "compute_ratio": aggregate_ratio,
                "ann_energy_nj": ann_energy_nj,
                "snn_energy_nj": snn_energy_nj,
                "energy_ratio": actual_ratio,
                "energy_saving_pct": (1.0 - actual_ratio) * 100.0,
                "mac_energy_pj": self.mac_energy_pj,
                "spike_energy_pj": self.spike_energy_pj,
            },
        }

    def estimate_from_shapes(
        self,
        shapes: list[tuple[int, int]],
        active_ratios: list[float] | None = None,
    ) -> dict[str, Any]:
        """Convenience: estimate from a list of (in, out) tuples.

        Parameters
        ----------
        shapes : list[tuple[int, int]]
            Layer shapes as (in_features, out_features).
        active_ratios : list[float] | None
            Per-layer activation sparsity. If None, defaults to 0.5.
        """
        if active_ratios is None:
            active_ratios = [0.5] * len(shapes)
        if len(active_ratios) != len(shapes):
            raise ValueError("active_ratios length must match shapes")

        layers = []
        for i, ((in_f, out_f), ratio) in enumerate(
            zip(shapes, active_ratios)
        ):
            layers.append({
                "name": f"layer_{i}",
                "in_features": in_f,
                "out_features": out_f,
                "active_ratio": ratio,
            })
        return self.estimate_model(layers)


def estimate_from_metadata(
    metadata_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Estimate energy from a LIF metadata JSON file.

    The metadata file should contain a list of layer entries with:
    - ``name``: layer name
    - ``in_features``: input dimension
    - ``out_features``: output dimension
    - ``active_ratio`` or ``spike_rate``: activation sparsity

    Parameters
    ----------
    metadata_path : str | Path
        Path to the LIF metadata JSON.
    output_path : str | Path | None
        If set, write results to this file.

    Returns
    -------
    dict with energy estimates.
    """
    path = Path(metadata_path)
    with open(path) as f:
        metadata = json.load(f)

    # Support both flat list and nested structures
    if isinstance(metadata, dict):
        layers_data = metadata.get("layers", metadata.get("layer_stats", []))
    elif isinstance(metadata, list):
        layers_data = metadata
    else:
        raise ValueError(f"unexpected metadata format: {type(metadata)}")

    layers = []
    for entry in layers_data:
        active = entry.get(
            "active_ratio",
            entry.get("spike_rate", 0.5),
        )
        layers.append({
            "name": entry.get("name", f"layer_{len(layers)}"),
            "in_features": entry["in_features"],
            "out_features": entry["out_features"],
            "active_ratio": active,
        })

    estimator = EnergyEstimator()
    results = estimator.estimate_model(layers)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

    return results
