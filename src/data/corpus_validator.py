"""Corpus validator: cluster real-dialogue embeddings and match to existing taxonomy.

Uses HDBSCAN (density-based) so we don't force a fixed K; noise points (-1)
remain unlabelled. Matches clusters to the fixed 10-domain taxonomy via
Hungarian assignment on a confusion-matrix-style overlap cost.
"""
from __future__ import annotations

import numpy as np


def cluster_embeddings_hdbscan(X: np.ndarray, *, min_cluster_size: int = 20,
                                min_samples: int | None = None) -> np.ndarray:
    """Run HDBSCAN on X, return cluster labels (noise = -1)."""
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
        metric="euclidean",
    )
    return clusterer.fit_predict(X)


def match_clusters_to_domains(true_domain: np.ndarray, cluster_id: np.ndarray,
                               n_domains: int) -> dict:
    """Hungarian assignment: each cluster -> one domain, maximising overlap.

    Returns dict with 'assignment' (cluster_id -> domain_idx), 'mean_overlap'
    (per-domain accuracy under the assignment), 'per_cluster_overlap' (list).
    """
    from scipy.optimize import linear_sum_assignment

    valid_mask = cluster_id != -1
    td = true_domain[valid_mask]
    ci = cluster_id[valid_mask]

    unique_clusters = sorted(set(int(c) for c in ci))
    n_clusters = len(unique_clusters)
    conf = np.zeros((n_clusters, n_domains), dtype=np.int64)
    for ck_idx, ck in enumerate(unique_clusters):
        mask = ci == ck
        for d in range(n_domains):
            conf[ck_idx, d] = int(((td == d) & mask).sum())

    if n_clusters >= n_domains:
        cost = -conf[:, :n_domains]
        row_ind, col_ind = linear_sum_assignment(cost)
        assignment = {unique_clusters[r]: int(col_ind[i]) for i, r in enumerate(row_ind)}
    else:
        cost = -conf.T
        row_ind, col_ind = linear_sum_assignment(cost[:n_clusters])
        assignment = {unique_clusters[int(c)]: int(r) for r, c in zip(row_ind, col_ind)}

    overlaps = []
    for ck, d in assignment.items():
        ck_mask = ci == ck
        overlaps.append(float((td[ck_mask] == d).mean()) if ck_mask.sum() > 0 else 0.0)

    return {
        "assignment": assignment,
        "mean_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "per_cluster_overlap": overlaps,
        "n_noise_points": int((~valid_mask).sum()),
        "n_valid_points": int(valid_mask.sum()),
    }
