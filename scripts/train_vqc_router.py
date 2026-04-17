#!/usr/bin/env python3
"""Train the Quantum VQC Router on domain-labelled prompts.

Pipeline:
1. Load prompts from eval/ + distilled/ + codex-generated/ + hf-extra/
2. Encode via TF-IDF character n-grams (no GPU, fast, portable)
3. Reduce to n_qubits dimensions via PCA
4. Train VQC (PennyLane parameter-shift) + linear head
5. Save weights to outputs/vqc-weights.npz

Usage:
    python3 scripts/train_vqc_router.py
    python3 scripts/train_vqc_router.py --epochs 100 --lr 0.02
    python3 scripts/train_vqc_router.py --eval-only  # load weights and evaluate
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_vqc")

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT))

# 10 niche domains (sorted) + "base" at index 10
NICHE_DOMAINS = sorted([
    "dsp", "electronics", "emc", "embedded", "freecad",
    "kicad-dsl", "platformio", "power", "spice", "stm32",
])
ALL_DOMAINS = NICHE_DOMAINS + ["base"]
DOMAIN_TO_IDX = {d: i for i, d in enumerate(ALL_DOMAINS)}

# Data source directories (relative to REPO_ROOT/data/)
DATA_SOURCES = ["eval", "distilled", "codex-generated", "hf-extra", "merged"]

# Domain aliases (map file/dir names to canonical niche names)
DOMAIN_ALIASES = {
    "kicad": "kicad-dsl",
    "kicad-pcb": "kicad-dsl",
    "spice-sim": "spice",
    "iot": "embedded",
}

# Max examples per domain to keep training tractable for VQC
MAX_PER_DOMAIN = 200


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def extract_prompt(example: dict) -> str | None:
    """Extract the user prompt from a JSONL example (various formats)."""
    if "prompt" in example:
        return example["prompt"]
    if "messages" in example:
        for msg in example["messages"]:
            if msg.get("role") == "user":
                return msg.get("content", "")
    if "instruction" in example:
        return example["instruction"]
    return None


def load_domain_prompts() -> dict[str, list[str]]:
    """Load prompts grouped by canonical domain from all data sources."""
    data_dir = REPO_ROOT / "data"
    domain_prompts: dict[str, list[str]] = {d: [] for d in ALL_DOMAINS}
    seen: dict[str, set[str]] = {d: set() for d in ALL_DOMAINS}

    for source in DATA_SOURCES:
        source_dir = data_dir / source
        if not source_dir.exists():
            continue

        # Check both <domain>/train.jsonl and <domain>.jsonl patterns
        for jsonl_path in sorted(source_dir.rglob("*.jsonl")):
            # Determine domain from path
            rel = jsonl_path.relative_to(source_dir)
            parts = rel.parts
            if len(parts) >= 2:
                raw_domain = parts[0]  # directory name
            else:
                raw_domain = jsonl_path.stem  # filename without .jsonl

            # Resolve aliases
            canonical = DOMAIN_ALIASES.get(raw_domain, raw_domain)
            if canonical not in DOMAIN_TO_IDX:
                continue  # skip non-niche domains (chat-fr, python, etc.)

            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        example = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    prompt = extract_prompt(example)
                    if prompt and len(prompt) > 10 and prompt not in seen[canonical]:
                        domain_prompts[canonical].append(prompt)
                        seen[canonical].add(prompt)

    # Cap per domain
    for domain in domain_prompts:
        if len(domain_prompts[domain]) > MAX_PER_DOMAIN:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(domain_prompts[domain]), MAX_PER_DOMAIN, replace=False)
            domain_prompts[domain] = [domain_prompts[domain][i] for i in sorted(indices)]

    return domain_prompts


# ---------------------------------------------------------------------------
# Feature extraction (TF-IDF character n-grams → PCA)
# ---------------------------------------------------------------------------

def char_ngram_features(texts: list[str], n_range: tuple[int, int] = (2, 4),
                        max_features: int = 500) -> np.ndarray:
    """Simple character n-gram TF-IDF without sklearn dependency.

    Returns (n_texts, max_features) float32 matrix.
    """
    # Build vocabulary from all texts
    ngram_counts: dict[str, int] = {}
    for text in texts:
        text_lower = text.lower()[:500]  # cap length
        seen_in_doc: set[str] = set()
        for n in range(n_range[0], n_range[1] + 1):
            for i in range(len(text_lower) - n + 1):
                ng = text_lower[i:i + n]
                if ng not in seen_in_doc:
                    ngram_counts[ng] = ngram_counts.get(ng, 0) + 1
                    seen_in_doc.add(ng)

    # Keep top max_features by document frequency
    sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: -x[1])
    vocab = {ng: idx for idx, (ng, _) in enumerate(sorted_ngrams[:max_features])}
    vocab_size = len(vocab)

    # Build TF-IDF matrix
    n_docs = len(texts)
    idf = np.zeros(vocab_size, dtype=np.float32)
    for ng, idx in vocab.items():
        df = ngram_counts[ng]
        idf[idx] = np.log((n_docs + 1) / (df + 1)) + 1.0

    matrix = np.zeros((n_docs, vocab_size), dtype=np.float32)
    for doc_idx, text in enumerate(texts):
        text_lower = text.lower()[:500]
        tf: dict[int, float] = {}
        total = 0
        for n in range(n_range[0], n_range[1] + 1):
            for i in range(len(text_lower) - n + 1):
                ng = text_lower[i:i + n]
                if ng in vocab:
                    v_idx = vocab[ng]
                    tf[v_idx] = tf.get(v_idx, 0) + 1
                    total += 1
        if total > 0:
            for v_idx, count in tf.items():
                matrix[doc_idx, v_idx] = (count / total) * idf[v_idx]

    # L2 normalize rows
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms

    return matrix


def pca_reduce(matrix: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple PCA via SVD. Returns (reduced, components, mean)."""
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    # Truncated SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]
    reduced = centered @ components.T
    return reduced, components, mean


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_vqc(args: argparse.Namespace) -> None:
    """Full training pipeline."""
    from src.routing.quantum_router import QuantumRouter, QuantumRouterConfig

    # 1. Load data
    logger.info("Loading domain prompts...")
    domain_prompts = load_domain_prompts()

    texts: list[str] = []
    labels: list[int] = []
    for domain, prompts in domain_prompts.items():
        idx = DOMAIN_TO_IDX[domain]
        for p in prompts:
            texts.append(p)
            labels.append(idx)

    labels_arr = np.array(labels)
    logger.info("Loaded %d prompts across %d domains:", len(texts),
                len([d for d, p in domain_prompts.items() if p]))
    for domain, prompts in domain_prompts.items():
        if prompts:
            logger.info("  %-15s %4d examples (label=%d)", domain, len(prompts),
                        DOMAIN_TO_IDX[domain])

    if len(texts) < 20:
        logger.error("Not enough data to train VQC (need >=20, got %d)", len(texts))
        return

    # 2. Feature extraction
    logger.info("Extracting char n-gram features (n=2..4, max=%d)...", args.max_features)
    features = char_ngram_features(texts, max_features=args.max_features)
    logger.info("  Feature matrix: %s", features.shape)

    # 3. PCA reduction to n_qubits dimensions
    n_qubits = args.n_qubits
    logger.info("PCA reduction: %d → %d dimensions...", features.shape[1], n_qubits)
    reduced, pca_components, pca_mean = pca_reduce(features, n_qubits)
    logger.info("  Reduced: %s, variance captured by %d PCs", reduced.shape, n_qubits)

    # Scale to [0, 2π] for AngleEmbedding
    r_min, r_max = reduced.min(), reduced.max()
    if r_max - r_min > 0:
        reduced = (reduced - r_min) / (r_max - r_min) * 2 * np.pi
    else:
        reduced = np.zeros_like(reduced)

    # 4. Train/val split (80/20 stratified)
    rng = np.random.default_rng(42)
    indices = np.arange(len(texts))
    rng.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train, y_train = reduced[train_idx], labels_arr[train_idx]
    X_val, y_val = reduced[val_idx], labels_arr[val_idx]
    logger.info("Train: %d, Val: %d", len(X_train), len(X_val))

    # 5. Initialize VQC
    config = QuantumRouterConfig(
        n_qubits=n_qubits,
        n_layers=args.n_layers,
        n_classes=len(ALL_DOMAINS),
        learning_rate=args.lr,
    )
    router = QuantumRouter(config)

    # 6. Train with mini-batches for speed
    logger.info("Training VQC: %d epochs, lr=%.4f, %d qubits, %d layers...",
                args.epochs, args.lr, n_qubits, args.n_layers)
    t0 = time.time()

    best_val_acc = 0.0
    best_weights = None
    best_linear_w = None
    best_linear_b = None
    patience_counter = 0

    for epoch in range(args.epochs):
        # Shuffle training data
        perm = rng.permutation(len(X_train))
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        # Train one epoch
        epoch_loss = 0.0
        batch_size = min(args.batch_size, len(X_train))
        n_batches = max(1, len(X_train) // batch_size)

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, len(X_train))
            batch_X = X_shuffled[start:end]
            batch_y = y_shuffled[start:end]

            for emb, label in zip(batch_X, batch_y):
                qubits = router.circuit(router.weights, emb)
                logits = qubits @ router.linear_w + router.linear_b
                probs = _softmax(logits)
                loss = -float(np.log(probs[int(label)] + 1e-12))
                epoch_loss += loss

                # Gradient for linear head
                d_logits = probs.copy()
                d_logits[int(label)] -= 1.0
                router.linear_w -= config.learning_rate * np.outer(qubits, d_logits)
                router.linear_b -= config.learning_rate * d_logits

                # Parameter-shift for VQC weights
                shift = np.pi / 2
                dl_dq = np.dot(router.linear_w, d_logits)

                for idx in np.ndindex(*router.weights.shape):
                    w_plus = router.weights.copy()
                    w_plus[idx] += shift
                    w_minus = router.weights.copy()
                    w_minus[idx] -= shift

                    q_plus = router.circuit(w_plus, emb)
                    q_minus = router.circuit(w_minus, emb)
                    dq_dw = (q_plus - q_minus) / 2.0
                    grad = float(np.dot(dl_dq, dq_dw))
                    router.weights[idx] -= config.learning_rate * grad

        avg_loss = epoch_loss / max(len(X_train), 1)

        # Validation accuracy
        val_correct = 0
        val_confs: list[float] = []
        for emb, label in zip(X_val, y_val):
            qubits = router.circuit(router.weights, emb)
            logits = qubits @ router.linear_w + router.linear_b
            probs = _softmax(logits)
            pred = int(np.argmax(probs))
            conf = float(probs[pred])
            val_confs.append(conf)
            if pred == label:
                val_correct += 1

        val_acc = val_correct / max(len(X_val), 1)
        avg_conf = np.mean(val_confs)

        logger.info(
            "Epoch %3d/%d  loss=%.4f  val_acc=%.1f%% (%d/%d)  avg_conf=%.3f",
            epoch + 1, args.epochs, avg_loss, val_acc * 100,
            val_correct, len(X_val), avg_conf,
        )

        # Early stopping with patience
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = router.weights.copy()
            best_linear_w = router.linear_w.copy()
            best_linear_b = router.linear_b.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, args.patience)
                break

    elapsed = time.time() - t0
    logger.info("Training done in %.1fs, best val_acc=%.1f%%", elapsed, best_val_acc * 100)

    # Restore best weights
    if best_weights is not None:
        router.weights = best_weights
        router.linear_w = best_linear_w
        router.linear_b = best_linear_b

    # 7. Save weights + PCA transform
    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = out_dir / "vqc-weights.npz"
    router.save(weights_path)

    # Save PCA transform separately (needed at inference time)
    pca_path = out_dir / "vqc-pca.npz"
    np.savez(pca_path,
             components=pca_components,
             mean=pca_mean,
             scale_min=r_min,
             scale_max=r_max,
             max_features=args.max_features)
    logger.info("PCA transform saved to %s", pca_path)

    # 8. Final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)
    _evaluate(router, X_val, y_val, labels_arr, domain_prompts)


def _evaluate(router, X_val, y_val, all_labels, domain_prompts) -> None:
    """Print per-domain accuracy on validation set."""
    from collections import Counter

    predictions: list[int] = []
    confidences: list[float] = []

    for emb in X_val:
        qubits = router.circuit(router.weights, emb)
        logits = qubits @ router.linear_w + router.linear_b
        probs = _softmax(logits)
        predictions.append(int(np.argmax(probs)))
        confidences.append(float(probs[predictions[-1]]))

    # Per-domain breakdown
    domain_stats: dict[str, dict[str, int]] = {}
    for pred, true, conf in zip(predictions, y_val, confidences):
        true_name = ALL_DOMAINS[true]
        pred_name = ALL_DOMAINS[pred]
        if true_name not in domain_stats:
            domain_stats[true_name] = {"correct": 0, "total": 0}
        domain_stats[true_name]["total"] += 1
        if pred == true:
            domain_stats[true_name]["correct"] += 1

    total_correct = sum(1 for p, t in zip(predictions, y_val) if p == t)
    total = len(y_val)

    logger.info("\n%-15s %6s %6s %8s", "Domain", "Correct", "Total", "Accuracy")
    logger.info("─" * 45)
    for domain in ALL_DOMAINS:
        if domain in domain_stats:
            s = domain_stats[domain]
            acc = s["correct"] / max(s["total"], 1) * 100
            logger.info("%-15s %6d %6d %7.1f%%", domain, s["correct"], s["total"], acc)

    logger.info("─" * 45)
    logger.info("%-15s %6d %6d %7.1f%%", "TOTAL", total_correct, total,
                total_correct / max(total, 1) * 100)
    logger.info("\nAvg confidence: %.3f", np.mean(confidences))


def eval_only(args: argparse.Namespace) -> None:
    """Load weights and evaluate on all data."""
    from src.routing.quantum_router import QuantumRouter, QuantumRouterConfig

    weights_path = REPO_ROOT / "outputs" / "vqc-weights.npz"
    pca_path = REPO_ROOT / "outputs" / "vqc-pca.npz"

    if not weights_path.exists() or not pca_path.exists():
        logger.error("Weights or PCA not found. Run training first.")
        return

    # Load data
    domain_prompts = load_domain_prompts()
    texts: list[str] = []
    labels: list[int] = []
    for domain, prompts in domain_prompts.items():
        for p in prompts:
            texts.append(p)
            labels.append(DOMAIN_TO_IDX[domain])

    # Load PCA
    pca_data = np.load(pca_path)
    features = char_ngram_features(texts, max_features=int(pca_data["max_features"]))
    centered = features - pca_data["mean"]
    reduced = centered @ pca_data["components"].T
    r_min, r_max = float(pca_data["scale_min"]), float(pca_data["scale_max"])
    if r_max - r_min > 0:
        reduced = (reduced - r_min) / (r_max - r_min) * 2 * np.pi

    # Load router
    config = QuantumRouterConfig(n_qubits=args.n_qubits, n_layers=args.n_layers)
    router = QuantumRouter(config)
    router.load(weights_path)

    labels_arr = np.array(labels)
    _evaluate(router, reduced, labels_arr, labels_arr, domain_prompts)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Quantum VQC Router")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--max-features", type=int, default=500,
                        help="Max char n-gram features before PCA")
    parser.add_argument("--eval-only", action="store_true",
                        help="Load existing weights and evaluate only")
    args = parser.parse_args()

    if args.eval_only:
        eval_only(args)
    else:
        train_vqc(args)


if __name__ == "__main__":
    main()
