"""Information-theoretic upper bound on VQC test accuracy.

Combines Holevo capacity (N qubits -> at most N bits of classical information
extractable) with Fano's inequality (classifier error floor given I(M; Y)) to
produce a single function ``acc_upper_bound(n_qubits, n_classes, mi_estimate_bits)``.

Erratum 2026-10-25 (hypneum-lab/micro-kiki#11, ``docs/ERRATA.md`` E-2026-10-25-02).
Until this revision ``estimate_mi_bits`` returned the *sum of per-feature
marginal* mutual informations (``mutual_info_classif(X, y).sum()``). That
quantity is not an estimate of the joint I(X; Y): it grows with the number of
(redundant) features and reached 150.74 bits for K = 10 classes on 384-d
embeddings (``results/c5-info-bound.json``), whereas I(X; Y) <= H(Y) <= log2 K
= 3.32 bits. Every bound derived from it (the 0.911 "C5" ceiling) is void.

The estimator now computes the joint MI with the Ross (2014) k-NN estimator on
the full feature vector and caps the result at the empirical label entropy
H(Y), emitting an explicit :class:`InfoBoundWarning` when the raw estimate
exceeds it. :func:`acc_upper_bound` applies the same cap.

Reference: B. C. Ross, "Mutual Information between Discrete and Continuous
Data Sets", PLoS ONE 9(2): e87357, 2014.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

_TOL_BITS = 1e-9


class InfoBoundWarning(RuntimeWarning):
    """Raised when a mutual-information input exceeds the label entropy H(Y)."""


def label_entropy_bits(y: np.ndarray) -> float:
    """Empirical entropy H(Y) of a discrete label vector, in bits."""
    _, counts = np.unique(np.asarray(y).ravel(), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def holevo_capacity_bits(n_qubits: int) -> float:
    """Holevo bound: max classical info extractable from N qubits = N bits."""
    return float(n_qubits)


def fano_error_lower_bound(mi_bits: float, n_classes: int, h_y: float | None = None) -> float:
    """Fano inequality lower bound on classification error.

    P_err >= (H(Y) - I(M; Y) - 1) / log2(K - 1)

    Args:
        mi_bits: estimated mutual information I(M; Y) in bits.
        n_classes: K.
        h_y: entropy of the label distribution (defaults to uniform = log2(K)).

    Returns:
        Lower bound on P_err, clipped to [0, 1].
    """
    if h_y is None:
        h_y = math.log2(n_classes)
    if n_classes <= 1:
        return 0.0
    bound = (h_y - mi_bits - 1.0) / math.log2(n_classes - 1)
    return float(max(0.0, min(1.0, bound)))


def acc_upper_bound(n_qubits: int, n_classes: int, mi_estimate_bits: float,
                    h_y: float | None = None) -> float:
    """Upper bound on test accuracy: 1 - Fano_error(min(MI, Holevo, H(Y))).

    The MI is capped twice. At the Holevo capacity N: the circuit cannot carry
    more than N bits of class information. At H(Y): no observation carries more
    information about Y than Y has, so an MI estimate above H(Y) is estimator
    bias, not information (this second cap is the fix for #11; crossing it
    emits an :class:`InfoBoundWarning`).

    Two consequences worth stating. When N >= log2 K (e.g. N = 4, K = 10) the
    Holevo cap does not bind and the bound is driven by the data-dependent MI
    alone. When the MI saturates H(Y) the bound is vacuous (1.0).
    """
    if h_y is None:
        h_y = math.log2(n_classes)
    if mi_estimate_bits > h_y + _TOL_BITS:
        warnings.warn(
            f"mutual information {mi_estimate_bits:.3f} bits exceeds H(Y) = {h_y:.3f} bits; "
            "capping at H(Y). A value above H(Y) is estimator bias, not information "
            "(docs/ERRATA.md E-2026-10-25-02).",
            InfoBoundWarning,
            stacklevel=2,
        )
    effective_mi = min(mi_estimate_bits, holevo_capacity_bits(n_qubits), h_y)
    err_floor = fano_error_lower_bound(effective_mi, n_classes, h_y=h_y)
    return 1.0 - err_floor


def _joint_mi_nats(X: np.ndarray, y: np.ndarray, n_neighbors: int,
                   rng: np.random.Generator) -> float:
    """Ross (2014) k-NN estimate of I(X; Y), continuous X in R^d, discrete Y.

    Multivariate form of the estimator behind scikit-learn's
    ``mutual_info_classif`` (which is per-feature only): for each sample, the
    distance to its k-th neighbour *within its own class* defines a ball; the
    estimate compares the number of same-class points to the number of points
    of any class inside that ball. Chebyshev metric; features scaled to unit
    variance and jittered to break ties, as scikit-learn does.
    """
    from scipy.special import digamma
    from sklearn.neighbors import KDTree

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y).ravel()
    n_samples = X.shape[0]
    if y.shape[0] != n_samples:
        raise ValueError(f"X has {n_samples} rows but y has {y.shape[0]} labels")

    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    X = X / std
    X = X + 1e-10 * np.maximum(1.0, np.abs(X).mean(axis=0)) * rng.standard_normal(X.shape)

    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples)
    k_all = np.empty(n_samples)
    for label in np.unique(y):
        mask = y == label
        count = int(mask.sum())
        if count > 1:
            k = min(n_neighbors, count - 1)
            tree = KDTree(X[mask], metric="chebyshev")
            dist, _ = tree.query(X[mask], k=k + 1)  # column 0 is the point itself
            radius[mask] = np.nextafter(dist[:, -1], 0)
            k_all[mask] = k
        label_counts[mask] = count

    keep = label_counts > 1  # singleton classes carry no neighbourhood
    n_eff = int(keep.sum())
    if n_eff == 0:
        return 0.0
    tree = KDTree(X[keep], metric="chebyshev")
    m_all = tree.query_radius(X[keep], radius[keep], count_only=True)
    mi = (digamma(n_eff) + np.mean(digamma(k_all[keep]))
          - np.mean(digamma(label_counts[keep])) - np.mean(digamma(m_all)))
    return float(max(0.0, mi))


def estimate_mi_bits(X: np.ndarray, y: np.ndarray, *, n_neighbors: int = 3,
                     random_state: int = 0) -> float:
    """Estimate the joint I(X; Y) in bits (continuous X, discrete Y).

    Joint k-NN estimate (Ross 2014) on the full feature vector, clipped to
    [0, H(Y)]. Crossing H(Y) triggers an :class:`InfoBoundWarning` because it
    can only come from estimator bias or variance. This replaces the sum of
    per-feature marginal MIs used until 2026-10-25, which is unbounded in the
    number of features and reached 150.74 bits for a 3.32-bit label (#11).
    """
    rng = np.random.default_rng(random_state)
    raw_bits = _joint_mi_nats(X, y, n_neighbors, rng) / math.log(2)
    h_y = label_entropy_bits(y)
    if raw_bits > h_y + _TOL_BITS:
        warnings.warn(
            f"joint MI estimate {raw_bits:.3f} bits exceeds H(Y) = {h_y:.3f} bits; "
            "capping at H(Y). A value above H(Y) is estimator bias, not information "
            "(docs/ERRATA.md E-2026-10-25-02).",
            InfoBoundWarning,
            stacklevel=2,
        )
        return h_y
    return raw_bits
