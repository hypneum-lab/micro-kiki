#!/usr/bin/env python3
"""End-to-end benchmark: PennyLane QuantumRouter vs TorchVQCRouter on real data.

Same preprocessing and data as eval_text_jepa_vqc.py (MiniLM baseline embeddings
+ 80/20 split) — the ONLY difference is the VQC backend. Reports accuracy
parity (should match within noise) and wallclock speedup.

Usage:
    uv run python scripts/bench_torch_vqc_e2e.py \
        --data-dir data/final \
        --domains dsp,electronics,emc,embedded \
        --max-per-domain 25 \
        --epochs 5 \
        --backbone models/niche-embeddings
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.routing.quantum_router import QuantumRouter, QuantumRouterConfig
from src.routing.text_jepa.dataset import load_domain_corpus
from src.routing.torch_vqc_router import TorchVQCRouter

logger = logging.getLogger(__name__)


def _baseline_embed(backbone: str, seq_len: int, samples) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(str(backbone), device="cpu")
    tok = st.tokenizer
    transformer = st[0].auto_model.to("cpu")

    embs = []
    for s in samples:
        enc = tok(s.text, return_tensors="pt", truncation=True, max_length=seq_len, padding="max_length")
        with torch.no_grad():
            out = transformer(**enc).last_hidden_state
        embs.append(out.squeeze(0).mean(dim=0).cpu().numpy())
    return np.stack(embs, axis=0)


def _eval_pennylane(embs, labels, n_classes, epochs, seed):
    cfg = QuantumRouterConfig(n_qubits=4, n_layers=6, n_classes=n_classes)
    vqc = QuantumRouter(cfg)
    t0 = time.perf_counter()
    vqc.train(embs, labels.astype(int), epochs=epochs)
    train_time = time.perf_counter() - t0
    return vqc, train_time


def _eval_torch(embs, labels, n_classes, epochs, seed, device="cpu"):
    model = TorchVQCRouter(n_qubits=4, n_layers=6, n_classes=n_classes, lr=0.01, seed=seed)
    model.to(device)
    X = torch.from_numpy(embs).double().to(device)
    y = torch.from_numpy(labels.astype(np.int64)).to(device)
    t0 = time.perf_counter()
    losses = model.train_batched(X, y, epochs=epochs)
    train_time = time.perf_counter() - t0
    return model, train_time


def _accuracy_pl(vqc, embs, labels):
    correct = 0
    for e, y in zip(embs, labels):
        qubits = vqc.circuit(vqc.weights, e)
        logits = qubits @ vqc.linear_w + vqc.linear_b
        if int(np.argmax(logits)) == int(y):
            correct += 1
    return correct / max(len(embs), 1)


def _accuracy_torch(model, embs, labels, device="cpu"):
    X = torch.from_numpy(embs).double().to(device)
    with torch.no_grad():
        preds = model.predict(X).cpu().numpy()
    return float((preds == labels).mean())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--domains", required=True)
    p.add_argument("--max-per-domain", type=int, default=25)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--backbone", default="models/niche-embeddings")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    samples = load_domain_corpus(Path(args.data_dir), domains=domains, max_per_domain=args.max_per_domain)
    if not samples:
        logger.error("no samples loaded")
        return 2
    logger.info("loaded %d samples across %d domains", len(samples), len(domains))

    dom_to_idx = {d: i for i, d in enumerate(domains)}
    labels = np.array([dom_to_idx[s.domain] for s in samples], dtype=np.int64)

    logger.info("computing baseline embeddings …")
    t_emb = time.perf_counter()
    embs = _baseline_embed(args.backbone, args.seq_len, samples)
    logger.info("embedding done in %.1fs, shape=%s", time.perf_counter() - t_emb, embs.shape)

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(embs))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    tr, te = idx[:split], idx[split:]

    n_classes = len(domains)

    logger.info("=== PennyLane VQC ===")
    pl_model, pl_train_t = _eval_pennylane(embs[tr], labels[tr], n_classes, args.epochs, args.seed)
    pl_train_acc = _accuracy_pl(pl_model, embs[tr], labels[tr])
    pl_test_acc = _accuracy_pl(pl_model, embs[te], labels[te])
    logger.info("PL train %.1fs  train_acc=%.3f  test_acc=%.3f", pl_train_t, pl_train_acc, pl_test_acc)

    logger.info("=== Torch VQC ===")
    t_model, t_train_t = _eval_torch(embs[tr], labels[tr], n_classes, args.epochs, args.seed)
    t_train_acc = _accuracy_torch(t_model, embs[tr], labels[tr])
    t_test_acc = _accuracy_torch(t_model, embs[te], labels[te])
    logger.info("T train %.1fs  train_acc=%.3f  test_acc=%.3f", t_train_t, t_train_acc, t_test_acc)

    speedup = pl_train_t / max(t_train_t, 1e-9)
    logger.info("=== SPEEDUP: %.1f× ===", speedup)

    out = {
        "n_samples": int(len(samples)),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_classes": n_classes,
        "epochs": args.epochs,
        "pennylane": {
            "train_time_s": pl_train_t,
            "train_acc": pl_train_acc,
            "test_acc": pl_test_acc,
        },
        "torch": {
            "train_time_s": t_train_t,
            "train_acc": t_train_acc,
            "test_acc": t_test_acc,
        },
        "speedup_training": speedup,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))
        logger.info("wrote %s", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
