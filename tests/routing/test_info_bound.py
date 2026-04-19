"""Tests for src/routing/info_bound.py — information-theoretic upper bound on VQC accuracy."""
from __future__ import annotations

import math

import numpy as np
import pytest


def test_holevo_cap_returns_n_bits():
    from src.routing.info_bound import holevo_capacity_bits

    assert holevo_capacity_bits(n_qubits=1) == pytest.approx(1.0, abs=1e-9)
    assert holevo_capacity_bits(n_qubits=4) == pytest.approx(4.0, abs=1e-9)
    assert holevo_capacity_bits(n_qubits=10) == pytest.approx(10.0, abs=1e-9)


def test_fano_bound_monotone_in_mi():
    """More mutual information → lower error bound (monotone decreasing)."""
    from src.routing.info_bound import fano_error_lower_bound

    k = 10
    h_y = math.log2(k)  # uniform prior
    err_low_mi = fano_error_lower_bound(mi_bits=0.5, h_y=h_y, n_classes=k)
    err_hi_mi = fano_error_lower_bound(mi_bits=3.0, h_y=h_y, n_classes=k)
    assert err_low_mi > err_hi_mi, (
        f"expected low MI → higher error floor, got {err_low_mi:.3f} vs {err_hi_mi:.3f}"
    )


def test_fano_bound_zero_mi_gives_near_chance_floor():
    """With I(M;Y) = 0, Fano yields error ≈ 1 − 1/K (can't do better than chance)."""
    from src.routing.info_bound import fano_error_lower_bound

    k = 10
    h_y = math.log2(k)
    err = fano_error_lower_bound(mi_bits=0.0, h_y=h_y, n_classes=k)
    # Fano: P_err ≥ (H(Y) − MI − 1) / log2(K-1) = (log2(10) − 0 − 1) / log2(9) ≈ 0.733
    assert 0.70 < err < 0.80, f"expected ~0.73 near-chance floor, got {err:.3f}"


def test_fano_bound_saturated_mi_gives_zero_error():
    """With I(M;Y) = H(Y) (perfect info), Fano bound drops to 0."""
    from src.routing.info_bound import fano_error_lower_bound

    k = 10
    h_y = math.log2(k)
    err = fano_error_lower_bound(mi_bits=h_y, n_classes=k, h_y=h_y)
    assert err <= 0.0 + 1e-9, f"expected 0 error floor at saturated MI, got {err:.3f}"


def test_acc_upper_bound_for_10_class_4_qubit():
    """Our target case: 4-qubit PauliZ measurements on 10-class routing.

    Holevo says MI ≤ 4 bits. Since H(Y) = log2(10) ≈ 3.32 bits, MI can saturate.
    The bound is then controlled by (i) actual MI between projected embeddings
    and class labels, not the Holevo cap.
    """
    from src.routing.info_bound import acc_upper_bound

    # Case: MI estimated at 1.0 bit → Fano err floor, acc_max = 1 − err_floor
    acc_max = acc_upper_bound(n_qubits=4, n_classes=10, mi_estimate_bits=1.0)
    # Fano: err ≥ (3.32 − 1.0 − 1) / log2(9) ≈ 0.417, so acc ≤ 0.583
    assert 0.55 < acc_max < 0.65, f"expected ~0.58, got {acc_max:.3f}"


def test_estimate_mi_discrete_on_separable_data():
    """MI estimator should saturate at H(Y) when features perfectly separate classes."""
    from src.routing.info_bound import estimate_mi_bits

    rng = np.random.default_rng(0)
    n_classes = 4
    # Class label determines the bin perfectly
    y = rng.integers(0, n_classes, size=400)
    x = y + rng.normal(0, 0.01, size=400)  # near-deterministic mapping
    mi = estimate_mi_bits(x.reshape(-1, 1), y)
    h_y = math.log2(n_classes)
    assert abs(mi - h_y) < 0.2, f"expected ~{h_y:.2f} bits (H(Y)), got {mi:.3f}"


def test_estimate_mi_zero_on_independent_data():
    """MI estimator should be near 0 when features are independent of labels."""
    from src.routing.info_bound import estimate_mi_bits

    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, size=400)
    x = rng.normal(size=(400, 3))  # pure noise, independent of y
    mi = estimate_mi_bits(x, y)
    assert mi < 0.3, f"expected < 0.3 bits for independent features, got {mi:.3f}"
