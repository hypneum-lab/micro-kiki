"""Uniform baseline runner for Phase C1: classical classifiers + TorchVQCRouter.

All baselines return the same dict: {name, accuracy, macro_f1, train_time_s, n_params}.
Used by scripts/bench_classical_vs_vqc.py to produce apples-to-apples comparisons.
"""
from __future__ import annotations

import time

import numpy as np

_KNOWN = {"stratified", "logreg", "logreg_pca", "mlp", "torch_vqc"}


def run_classical_baseline(
    name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    *,
    seed: int = 0,
    pca_dim: int = 4,
    hidden_dim: int = 64,
    n_qubits: int = 4,
    n_layers: int = 6,
    epochs: int = 300,
    lr: float = 0.05,
    weight_decay: float = 1e-4,
    max_iter: int = 2000,
) -> dict:
    """Train + eval one baseline. Returns uniform dict for aggregation."""
    if name not in _KNOWN:
        raise ValueError(f"unknown baseline {name!r} — must be one of {_KNOWN}")

    from sklearn.metrics import accuracy_score, f1_score

    t0 = time.perf_counter()

    if name == "stratified":
        from sklearn.dummy import DummyClassifier
        clf = DummyClassifier(strategy="stratified", random_state=seed)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        n_params = 0  # non-parametric

    elif name == "logreg":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=max_iter, random_state=seed)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        # coef_: (n_classes, n_features); intercept_: (n_classes,)
        n_params = int(clf.coef_.size + clf.intercept_.size)

    elif name == "logreg_pca":
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        pca = PCA(n_components=pca_dim, random_state=seed)
        Xtr_p = pca.fit_transform(X_tr)
        Xte_p = pca.transform(X_te)
        clf = LogisticRegression(max_iter=max_iter, random_state=seed)
        clf.fit(Xtr_p, y_tr)
        y_pred = clf.predict(Xte_p)
        # PCA params (components_) + LogReg params
        n_params = int(pca.components_.size + clf.coef_.size + clf.intercept_.size)

    elif name == "mlp":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(
            hidden_layer_sizes=(hidden_dim,),
            max_iter=max_iter,
            random_state=seed,
            solver="adam",
            early_stopping=False,
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        # sum weights + biases across all layers
        n_params = int(sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_))

    elif name == "torch_vqc":
        import torch
        from src.routing.torch_vqc_router import TorchVQCRouter
        model = TorchVQCRouter(
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_classes=int(max(y_tr.max(), y_te.max())) + 1,
            lr=lr,
            seed=seed,
            input_dim=int(X_tr.shape[1]),
            weight_decay=weight_decay,
        )
        Xt = torch.from_numpy(X_tr).double()  # type: ignore[name-defined]
        yt = torch.from_numpy(y_tr.astype(np.int64))  # type: ignore[name-defined]
        model.train_batched(Xt, yt, epochs=epochs)
        with torch.no_grad():  # type: ignore[name-defined]
            Xe = torch.from_numpy(X_te).double()  # type: ignore[name-defined]
            y_pred = model.predict(Xe).numpy()
        n_params = int(sum(p.numel() for p in model.parameters()))

    train_time = time.perf_counter() - t0

    acc = float(accuracy_score(y_te, y_pred))
    f1 = float(f1_score(y_te, y_pred, average="macro"))

    return {
        "name": name,
        "accuracy": acc,
        "macro_f1": f1,
        "train_time_s": train_time,
        "n_params": n_params,
    }
