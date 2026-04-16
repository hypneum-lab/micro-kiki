"""Leaky Integrate-and-Fire (LIF) neuron primitive.

Minimal numpy-only implementation used as the activation replacement in
the LAS (Lossless ANN→SNN) converter. See arxiv 2505.09659 §3 for the
rate-coded variant we reproduce here.

Semantics
---------
Given an input current ``I_t`` at each timestep ``t``:

    V_t = tau * V_{t-1} + I_t
    spike_t = 1 if V_t >= threshold else 0
    V_t -= spike_t * threshold   # soft reset (preserves residual)

The neuron approximates a (positive) ReLU activation when driven with a
constant positive current over a sufficient time window: the mean spike
rate matches ``min(I / threshold, 1 / threshold)``.

The primitive is deliberately framework-light: pure numpy on the hot
path so tests never need torch. A torch-compatible wrapper lives in
``las_converter.SpikingLinear``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LIFNeuron", "rate_encode"]


@dataclass
class LIFNeuron:
    """Leaky integrate-and-fire neuron (soft-reset variant).

    Parameters
    ----------
    threshold:
        Membrane-potential threshold for firing. Spike emitted when
        ``V_t >= threshold``; membrane decremented by ``threshold``.
    tau:
        Leak factor in ``[0, 1]``. ``tau=1.0`` is a pure
        integrate-and-fire (no leak, used for lossless rate codes).
    v_init:
        Initial membrane potential (default 0).
    """

    threshold: float = 1.0
    tau: float = 1.0
    v_init: float = 0.0

    def __post_init__(self) -> None:
        if self.threshold <= 0:
            raise ValueError("threshold must be > 0")
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError("tau must be in [0, 1]")

    def simulate(
        self,
        currents: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the neuron over ``T`` timesteps.

        Parameters
        ----------
        currents:
            Array of shape ``(T, ...)`` — per-timestep input current.
            Any trailing shape is supported (the neuron is element-wise).

        Returns
        -------
        spikes:
            Boolean array of the same shape, ``True`` where a spike
            fired at that timestep.
        membrane:
            Final membrane potential (shape ``currents.shape[1:]``).
        """
        if currents.ndim == 0:
            raise ValueError("currents must have at least 1 timestep axis")
        T = currents.shape[0]
        feature_shape = currents.shape[1:]
        v = np.full(feature_shape, fill_value=self.v_init, dtype=np.float64)
        spikes = np.zeros_like(currents, dtype=np.float64)
        for t in range(T):
            v = self.tau * v + currents[t].astype(np.float64)
            fired = v >= self.threshold
            spikes[t] = fired.astype(np.float64)
            # Soft reset — subtract threshold where we fired.
            v = np.where(fired, v - self.threshold, v)
        return spikes, v


def rate_encode(
    activations: np.ndarray,
    timesteps: int,
    max_rate: float = 1.0,
) -> np.ndarray:
    """Convert a real-valued activation tensor into a rate-coded current.

    Each activation ``a`` is broadcast to a constant current
    ``a * max_rate / timesteps`` across ``timesteps`` steps. Feeding
    this into an LIF neuron with ``threshold=max_rate / timesteps``
    recovers ``min(relu(a), max_rate)`` as a mean spike rate in the
    limit of large ``timesteps``.

    Parameters
    ----------
    activations:
        Real-valued array of any shape.
    timesteps:
        Number of integration steps ``T``.
    max_rate:
        Upper clipping bound on the encoded activation magnitude.

    Returns
    -------
    current:
        Array of shape ``(T,) + activations.shape`` with per-step current.
    """
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if max_rate <= 0:
        raise ValueError("max_rate must be positive")
    clipped = np.clip(activations, 0.0, max_rate)
    per_step = clipped / float(timesteps)
    return np.broadcast_to(per_step, (timesteps,) + clipped.shape).copy()
