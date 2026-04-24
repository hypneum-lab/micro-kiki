"""micro-kiki ↔ dream-of-kiki integration package.

Exposes :class:`MicroKikiSubstrate`, a prototype implementation of the
``kiki_oniric`` framework-C substrate ABI (see
``github.com/hypneum-lab/dream-of-kiki``, spec C-v0.7.0+PARTIAL) backed
by an mlx_lm base model + LoRA adapter + Aeon Atlas/Trace records.

Spike 2 scope : validates the API shape on a small mock model
(1.5B-4bit class). Full 35B wiring + conformance test runs land in
phase 3-4 (see ``docs/research/2026-04-21-dream-of-kiki-substrate-spec.md``).
"""

from __future__ import annotations

from src.dream.substrate import (
    MICROKIKI_SUBSTRATE_NAME,
    MICROKIKI_SUBSTRATE_VERSION,
    MicroKikiSubstrate,
    microkiki_substrate_components,
)

__all__ = [
    "MICROKIKI_SUBSTRATE_NAME",
    "MICROKIKI_SUBSTRATE_VERSION",
    "MicroKikiSubstrate",
    "microkiki_substrate_components",
]
