"""Unit tests for :mod:`src.spiking.lif_neuron`."""

from __future__ import annotations

import numpy as np
import pytest

from src.spiking.lif_neuron import LIFNeuron, rate_encode


def test_lif_threshold_validation() -> None:
    with pytest.raises(ValueError):
        LIFNeuron(threshold=0.0)


def test_lif_tau_validation() -> None:
    with pytest.raises(ValueError):
        LIFNeuron(threshold=1.0, tau=1.5)


def test_lif_fires_at_threshold() -> None:
    """A constant current equal to threshold must fire every step."""
    neuron = LIFNeuron(threshold=1.0, tau=1.0)
    current = np.ones((5, 1), dtype=np.float64)
    spikes, v = neuron.simulate(current)
    assert spikes.sum() == pytest.approx(5.0)
    # Soft reset means the membrane ends near 0.
    assert abs(float(v.item())) < 1e-9


def test_lif_rate_code_recovers_activation() -> None:
    """Feeding rate-encoded ``a`` recovers ``a`` as spike-count * threshold.

    With ``T = 16`` steps and a bounded activation in [0, max_rate],
    ``sum(spikes) * threshold`` reconstructs the activation exactly (no
    leak, soft reset). We verify on a handful of target values.
    """
    T = 16
    max_rate = 1.0
    thr = max_rate / T
    neuron = LIFNeuron(threshold=thr, tau=1.0)
    for a in [0.0, 0.125, 0.5, 0.875, 1.0]:
        current = rate_encode(np.array([a]), timesteps=T, max_rate=max_rate)
        spikes, _ = neuron.simulate(current)
        reconstructed = spikes.sum() * thr
        # With T=16 the quantisation error is at most thr = 1/16 ≈ 0.0625.
        assert abs(reconstructed - a) <= thr + 1e-9, (a, reconstructed)


def test_rate_encode_clips_above_max_rate() -> None:
    current = rate_encode(np.array([2.5]), timesteps=10, max_rate=1.0)
    # After clipping to max_rate=1.0, each step gets 0.1.
    assert current.shape == (10, 1)
    assert np.allclose(current, 0.1)
