"""Tests for src/routing/info_bound.py — information-theoretic upper bound on VQC accuracy."""
from __future__ import annotations

import math
import warnings

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


# ---------------------------------------------------------------------------
# Erratum 2026-10-25 (hypneum-lab/micro-kiki#11, master table MK-23).
# Until that date ``estimate_mi_bits`` summed per-feature marginal MIs and
# returned 150.74 bits for K = 10 classes (> log2 10 = 3.32 bits); the 0.911
# "C5" ceiling derived from it is void. The tests below fail on that formula.
# ---------------------------------------------------------------------------


def _redundant_features(n_classes: int = 10, n_per_class: int = 50,
                        n_copies: int = 50, noise: float = 0.1, seed: int = 0):
    """K separable classes seen through ``n_copies`` noisy copies of one coordinate.

    Every copy carries ~log2(K) bits about Y on its own, so the sum of marginal
    MIs is ~n_copies * log2(K) while the joint I(X; Y) cannot exceed log2(K).
    """
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(n_classes), n_per_class)
    X = y[:, None] + noise * rng.normal(size=(y.size, n_copies))
    return X, y


def test_label_entropy_bits_uniform_is_log2_k():
    from src.routing.info_bound import label_entropy_bits

    y = np.repeat(np.arange(10), 50)
    assert label_entropy_bits(y) == pytest.approx(math.log2(10), abs=1e-9)
    assert label_entropy_bits(np.zeros(20, dtype=int)) == pytest.approx(0.0, abs=1e-12)


def test_joint_mi_estimate_never_exceeds_label_entropy():
    """Regression for #11: I(X; Y) <= H(Y) <= log2(K) must hold for any X.

    Fails on the pre-2026-10-25 formula (sum of marginal MIs ~ 166 bits here).
    """
    from src.routing.info_bound import estimate_mi_bits

    X, y = _redundant_features()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the H(Y) cap may legitimately fire
        mi = estimate_mi_bits(X, y)
    assert mi <= math.log2(10) + 1e-9, f"MI estimate {mi:.2f} bits exceeds log2(10)"
    assert mi > 2.5, f"classes are separable, expected MI near H(Y), got {mi:.2f}"


def test_sum_of_marginal_mi_violates_label_entropy():
    """Documents the defect: the withdrawn formula blows far past H(Y)."""
    from sklearn.feature_selection import mutual_info_classif

    X, y = _redundant_features()
    marginal_sum_bits = (
        mutual_info_classif(X, y, n_neighbors=3, random_state=0).sum() / math.log(2)
    )
    assert marginal_sum_bits > 10 * math.log2(10), (
        f"expected tens of bits from the marginal sum, got {marginal_sum_bits:.2f}"
    )


def test_acc_upper_bound_caps_mi_above_label_entropy_with_warning():
    """The 150.74-bit input of results/c5-info-bound.json must warn and be capped."""
    from src.routing.info_bound import InfoBoundWarning, acc_upper_bound

    with pytest.warns(InfoBoundWarning):
        acc = acc_upper_bound(n_qubits=4, n_classes=10, mi_estimate_bits=150.74)
    assert acc == pytest.approx(1.0)  # vacuous bound, not an artefact of 150 bits


def test_acc_upper_bound_holevo_cap_inactive_when_qubits_exceed_log2_k():
    """N = 4 >= log2(10): the Holevo term cannot bind; the bound is data-driven only."""
    from src.routing.info_bound import acc_upper_bound

    acc_4q = acc_upper_bound(n_qubits=4, n_classes=10, mi_estimate_bits=2.0)
    acc_40q = acc_upper_bound(n_qubits=40, n_classes=10, mi_estimate_bits=2.0)
    assert acc_4q == pytest.approx(acc_40q)


def test_estimate_mi_joint_is_capped_and_warns_when_raw_exceeds_entropy():
    from src.routing.info_bound import InfoBoundWarning, estimate_mi_bits

    # Two perfectly separated classes, tiny sample: the k-NN estimator is biased
    # upward and can exceed H(Y) = 1 bit; the cap must then engage and warn.
    y = np.array([0] * 8 + [1] * 8)
    X = (y * 10.0).reshape(-1, 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mi = estimate_mi_bits(X, y)
    assert mi <= 1.0 + 1e-9
    assert all(issubclass(w.category, InfoBoundWarning) for w in caught)
