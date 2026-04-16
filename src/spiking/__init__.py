"""Spiking-neural-network conversion tools (v0.3 Phase N-IV).

This package hosts the LAS (Lossless ANN→SNN) converter plus primitives
used to transform pretrained transformer layers into spiking equivalents.

See ``docs/specs/las-conversion-framework.md`` for the design spec.
"""

from src.spiking.las_converter import (
    LASConverter,
    SpikingLinear,
    convert_linear,
    verify_equivalence,
)
from src.spiking.lif_neuron import LIFNeuron, rate_encode

__all__ = [
    "LASConverter",
    "SpikingLinear",
    "convert_linear",
    "verify_equivalence",
    "LIFNeuron",
    "rate_encode",
]
