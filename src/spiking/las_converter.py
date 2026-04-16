"""LAS — Lossless ANN→SNN conversion (story-17 minimal implementation).

This module implements the core of arxiv 2505.09659 at the scale that
v0.3 Phase N-IV needs for the framework story: per-layer conversion of
``nn.Linear`` blocks into spiking equivalents, plus a ``convert_model``
walker that dispatches per-layer conversion for toy MLPs.

Design choices (see ``docs/specs/las-conversion-framework.md`` for the
full spec):

- **Rate-coded LIF** with soft reset. A positive pre-activation ``a`` is
  encoded as a constant per-step current ``a / T`` into an LIF neuron
  with ``threshold = max_rate / T``. Over ``T`` timesteps, the spike
  count reconstructs the ReLU-clipped activation up to a quantisation
  error of ``max_rate / T``.
- **Negative values** are not representable in a single rate channel;
  the minimal converter therefore assumes a ReLU-style non-negativity
  at conversion boundaries (matches the paper's §3.1 assumption for
  the unsigned rate code). MoE / attention extensions in later stories
  (21, 25) add signed two-channel encoding — out of scope here.
- **Numpy-first**: the unit tests never need torch. ``SpikingLinear``
  is a pure-numpy forward; a torch wrapper is exposed when torch is
  importable, but it is a thin adapter.

Public surface:

- :class:`SpikingLinear` — LAS-converted ``nn.Linear`` equivalent.
- :class:`LASConverter` — entry point with ``convert_layer``,
  ``convert_model``, ``verify_equivalence``.
- :func:`convert_linear` / :func:`verify_equivalence` — convenience
  functions that do not require instantiating the class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from src.spiking.lif_neuron import LIFNeuron, rate_encode

__all__ = [
    "LASConverter",
    "SpikingLinear",
    "convert_linear",
    "verify_equivalence",
]


# ---------------------------------------------------------------------------
# Weight extraction — supports numpy arrays, dicts, and torch.nn.Linear.
# ---------------------------------------------------------------------------


def _extract_linear(layer: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``(weight, bias)`` as numpy arrays for a linear-like layer.

    Accepted forms:

    - ``torch.nn.Linear`` (lazy import of torch to keep unit tests pure)
    - a dict with ``weight`` (and optional ``bias``) numpy arrays
    - a tuple/list ``(weight, bias)``
    """
    # Duck-typed torch.nn.Linear.
    if hasattr(layer, "weight") and hasattr(layer, "bias"):
        w = layer.weight
        b = layer.bias
        w_np = _to_numpy(w)
        b_np = _to_numpy(b) if b is not None else None
        return w_np, b_np
    if isinstance(layer, dict):
        w = layer["weight"]
        b = layer.get("bias")
        return _to_numpy(w), (_to_numpy(b) if b is not None else None)
    if isinstance(layer, (tuple, list)) and len(layer) in (1, 2):
        w = _to_numpy(layer[0])
        b = _to_numpy(layer[1]) if len(layer) == 2 and layer[1] is not None else None
        return w, b
    raise TypeError(f"unsupported layer type: {type(layer).__name__}")


def _to_numpy(x: Any) -> np.ndarray:
    """Convert ``x`` to a numpy array, handling torch tensors lazily."""
    if isinstance(x, np.ndarray):
        return x
    # torch tensor duck-type: has .detach / .cpu / .numpy
    detach = getattr(x, "detach", None)
    if callable(detach):
        t = detach()
        cpu = getattr(t, "cpu", None)
        if callable(cpu):
            t = cpu()
        return np.asarray(t.numpy())
    return np.asarray(x)


# ---------------------------------------------------------------------------
# SpikingLinear — numpy forward + optional torch convenience wrapper.
# ---------------------------------------------------------------------------


@dataclass
class SpikingLinear:
    """LAS-converted counterpart of a single ``nn.Linear`` layer.

    Forward semantics (per input sample ``x``):

    1. Compute pre-activation ``z = x @ W.T + b`` (same as ANN).
    2. Rate-encode ``relu(z)`` across ``T`` timesteps.
    3. Run the LIF neuron; sum spikes; rescale by ``threshold`` to get
       the reconstructed activation ``a_hat``.
    4. In the zero-error limit (``T → ∞``) ``a_hat == relu(z)``.

    For *identity* conversion (``activation="identity"``) we skip the
    ReLU clipping at step 2 — this is the surrogate used by Story 17's
    acceptance test where we feed pre-clipped inputs and expect an
    exact recovery.
    """

    weight: np.ndarray
    bias: np.ndarray | None
    timesteps: int = 16
    max_rate: float = 1.0
    activation: str = "relu"  # one of: "relu", "identity"
    _neuron: LIFNeuron = field(init=False)

    def __post_init__(self) -> None:
        if self.weight.ndim != 2:
            raise ValueError("weight must be 2-D")
        if self.bias is not None and self.bias.ndim != 1:
            raise ValueError("bias must be 1-D")
        if self.bias is not None and self.bias.shape[0] != self.weight.shape[0]:
            raise ValueError("bias shape does not match weight out-features")
        if self.timesteps <= 0:
            raise ValueError("timesteps must be positive")
        if self.max_rate <= 0:
            raise ValueError("max_rate must be positive")
        if self.activation not in ("relu", "identity"):
            raise ValueError("activation must be 'relu' or 'identity'")
        threshold = self.max_rate / float(self.timesteps)
        self._neuron = LIFNeuron(threshold=threshold, tau=1.0)

    @property
    def out_features(self) -> int:
        return int(self.weight.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.weight.shape[1])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run the spiking forward pass.

        Parameters
        ----------
        x:
            Array of shape ``(..., in_features)``. Last axis is the
            feature axis.

        Returns
        -------
        out:
            Array of shape ``(..., out_features)`` — the reconstructed
            activation after the spike integration.
        """
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(
                f"input last-dim {x.shape[-1]} != in_features "
                f"{self.in_features}"
            )
        # Pre-activation (same as the ANN matmul).
        z = x @ self.weight.T
        if self.bias is not None:
            z = z + self.bias
        if self.activation == "relu":
            z = np.maximum(z, 0.0)
        # Rate-encode and run the LIF neuron over T timesteps.
        current = rate_encode(z, timesteps=self.timesteps, max_rate=self.max_rate)
        spikes, _ = self._neuron.simulate(current)
        spike_count = spikes.sum(axis=0)
        # Each spike contributes `threshold` to the reconstructed value.
        return spike_count * self._neuron.threshold

    # Alias so the object is callable like a torch module.
    __call__ = forward


def convert_linear(
    layer: Any,
    timesteps: int = 16,
    max_rate: float = 1.0,
    activation: str = "relu",
) -> SpikingLinear:
    """Convenience wrapper: convert a single linear layer."""
    w, b = _extract_linear(layer)
    return SpikingLinear(
        weight=w,
        bias=b,
        timesteps=timesteps,
        max_rate=max_rate,
        activation=activation,
    )


# ---------------------------------------------------------------------------
# Tiny MLP walker for `convert_model`.
# ---------------------------------------------------------------------------


@dataclass
class SpikingMLP:
    """Simple sequential stack of :class:`SpikingLinear` layers."""

    layers: list[SpikingLinear]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    __call__ = forward


# ---------------------------------------------------------------------------
# LASConverter — entry point class.
# ---------------------------------------------------------------------------


class LASConverter:
    """Lossless ANN→SNN conversion via time-coded activation alignment.

    Minimal implementation for Story 17. Later stories extend the
    converter to MoE routing (21) and Mistral dense blocks (25).

    Parameters
    ----------
    timesteps:
        Number of LIF integration steps ``T``. Larger ``T`` shrinks the
        quantisation error ``O(1/T)``. Default 16 (paper setting for
        lossless on ViT).
    max_rate:
        Upper bound on activation magnitude. Activations clipped beyond
        this value saturate the rate code.
    """

    def __init__(
        self,
        timesteps: int = 16,
        max_rate: float = 1.0,
    ) -> None:
        if timesteps <= 0:
            raise ValueError("timesteps must be positive")
        if max_rate <= 0:
            raise ValueError("max_rate must be positive")
        self.timesteps = int(timesteps)
        self.max_rate = float(max_rate)
        self._activation_stats: dict[str, dict[str, float]] = {}

    # ---- per-layer -------------------------------------------------------

    def convert_layer(
        self,
        layer: Any,
        activation: str = "relu",
    ) -> SpikingLinear:
        """Convert a single linear-like layer (see :func:`convert_linear`)."""
        return convert_linear(
            layer,
            timesteps=self.timesteps,
            max_rate=self.max_rate,
            activation=activation,
        )

    # ---- full model ------------------------------------------------------

    def convert_model(
        self,
        model: Any,
        time_window: int | None = None,
    ) -> SpikingMLP:
        """Convert a tiny sequential model.

        Accepted shapes:

        - A list/tuple of linear-like layers (each accepted by
          :func:`convert_linear`). This is the minimal path used by
          the unit tests.
        - A torch ``nn.Sequential`` of ``nn.Linear`` layers (other
          modules are passed through as ReLU boundaries).

        MoE / attention extensions live in Story 21+. ``time_window``
        overrides the converter's ``timesteps`` for this call.
        """
        if time_window is not None:
            self.timesteps = int(time_window)

        linears = list(self._iter_linears(model))
        if not linears:
            raise ValueError("no linear-like layers found in model")
        spiking = [self.convert_layer(layer) for layer in linears]
        return SpikingMLP(layers=spiking)

    @staticmethod
    def _iter_linears(model: Any) -> Iterable[Any]:
        if isinstance(model, (list, tuple)):
            for m in model:
                yield m
            return
        # torch.nn.Sequential-like: iterable of children.
        children = getattr(model, "children", None)
        if callable(children):
            for child in children():
                # crude filter — include anything with a 2-D .weight
                w = getattr(child, "weight", None)
                if w is not None and getattr(w, "ndim", 0) == 2:
                    yield child
            return
        raise TypeError(f"unsupported model type: {type(model).__name__}")

    # ---- verification ----------------------------------------------------

    def verify_equivalence(
        self,
        ann_forward: Callable[[np.ndarray], np.ndarray],
        snn_model: SpikingLinear | SpikingMLP,
        sample_input: np.ndarray,
        tol: float = 5e-2,
    ) -> bool:
        """Check that the SNN output matches the ANN within ``tol``.

        ``ann_forward`` is any callable that takes ``sample_input`` and
        returns the ANN's activations at the conversion boundary
        (typically ``relu(x @ W.T + b)``). The comparison metric is
        **relative L2**:

            ||snn - ann||_2 / (||ann||_2 + eps) < tol

        A relative metric is used because rate-code quantisation
        introduces per-element error that scales with the activation
        magnitude, not the absolute value.
        """
        ann_out = ann_forward(sample_input)
        snn_out = snn_model(sample_input)
        diff = np.linalg.norm(snn_out - ann_out)
        norm = np.linalg.norm(ann_out) + 1e-12
        rel = float(diff / norm)
        return rel < tol

    # ---- stats -----------------------------------------------------------

    def activation_stats(self) -> dict[str, dict[str, float]]:
        """Return per-layer activation bounds (populated by extensions)."""
        return dict(self._activation_stats)


def verify_equivalence(
    ann_forward: Callable[[np.ndarray], np.ndarray],
    snn_model: SpikingLinear | SpikingMLP,
    sample_input: np.ndarray,
    tol: float = 5e-2,
) -> bool:
    """Module-level convenience wrapper around ``LASConverter.verify``."""
    converter = LASConverter()
    return converter.verify_equivalence(ann_forward, snn_model, sample_input, tol=tol)
