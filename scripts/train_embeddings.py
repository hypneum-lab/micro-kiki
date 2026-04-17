#!/usr/bin/env python3
"""Fine-tune domain-specific embedding model for Aeon memory recall.

Uses sentence-transformers with two loss functions:
1. MultipleNegativesRankingLoss (MNRL) -- in-batch negatives
2. TripletLoss -- hard negatives from confusing domain pairs

Usage:
    python3 scripts/train_embeddings.py --all
    python3 scripts/train_embeddings.py --dry-run
    python3 scripts/train_embeddings.py --domains spice emc power
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FINAL_DATA = REPO_ROOT / "data" / "final"
OUTPUT_DIR = REPO_ROOT / "models" / "niche-embeddings"

NICHE_DOMAINS = [
    "kicad-dsl", "spice", "emc", "stm32", "embedded",
    "freecad", "platformio", "power", "dsp", "electronics",
]

HARD_NEGATIVE_PAIRS = [
    ("embedded", "stm32"),
    ("spice", "power"),
    ("kicad-dsl", "electronics"),
    ("embedded", "platformio"),
]

MIN_DOMAIN_EXAMPLES = 100
MAX_DOMAIN_EXAMPLES = 2000


def load_texts_from_file(path: Path, min_len: int = 20) -> list[str]:
    """Extract text strings from a JSONL file."""
    texts: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "messages" in d:
                for m in d["messages"]:
                    c = m.get("content", "")
                    if len(c) >= min_len:
                        texts.append(c)
            elif "prompt" in d:
                p = d["prompt"]
                if len(p) >= min_len:
                    texts.append(p)
            elif "instruction" in d:
                p = d["instruction"]
                if len(p) >= min_len:
                    texts.append(p)
    return texts


def load_all_domain_texts(domains: list[str]) -> dict[str, list[str]]:
    """Load texts for each domain from data/final/."""
    domain_texts: dict[str, list[str]] = {}
    for domain in domains:
        train_file = FINAL_DATA / domain / "train.jsonl"
        if not train_file.exists():
            logger.warning("No data for %s at %s", domain, train_file)
            domain_texts[domain] = []
            continue
        texts = load_texts_from_file(train_file)
        if len(texts) > MAX_DOMAIN_EXAMPLES:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(texts), MAX_DOMAIN_EXAMPLES, replace=False)
            texts = [texts[i] for i in sorted(idx)]
        if 0 < len(texts) < MIN_DOMAIN_EXAMPLES:
            texts = oversample(texts, MIN_DOMAIN_EXAMPLES, seed=42)
        domain_texts[domain] = texts
        logger.info("  %-15s %4d texts", domain, len(texts))
    return domain_texts


def oversample(texts: list[str], target: int, seed: int = 42) -> list[str]:
    """Repeat texts to reach target count."""
    if len(texts) >= target:
        return texts
    rng = np.random.default_rng(seed)
    extra = rng.choice(len(texts), target - len(texts), replace=True)
    return texts + [texts[i] for i in extra]


def build_mnrl_pairs(domain_texts: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Build (anchor, positive) pairs from same-domain texts for MNRL."""
    pairs: list[tuple[str, str]] = []
    for domain, texts in domain_texts.items():
        if len(texts) < 2:
            continue
        for i in range(0, len(texts) - 1, 2):
            pairs.append((texts[i], texts[i + 1]))
    return pairs


def build_hard_negative_triplets(
    domain_texts: dict[str, list[str]],
) -> list[tuple[str, str, str]]:
    """Build (anchor, positive, negative) triplets from confusing pairs."""
    triplets: list[tuple[str, str, str]] = []
    rng = np.random.default_rng(42)
    for domain_a, domain_b in HARD_NEGATIVE_PAIRS:
        texts_a = domain_texts.get(domain_a, [])
        texts_b = domain_texts.get(domain_b, [])
        if len(texts_a) < 2 or len(texts_b) < 1:
            continue
        for i in range(0, len(texts_a) - 1, 2):
            neg_idx = int(rng.integers(0, len(texts_b)))
            triplets.append((texts_a[i], texts_a[i + 1], texts_b[neg_idx]))
        for i in range(0, len(texts_b) - 1, 2):
            neg_idx = int(rng.integers(0, len(texts_a)))
            triplets.append((texts_b[i], texts_b[i + 1], texts_a[neg_idx]))
    return triplets


def train(domains: list[str], args: argparse.Namespace) -> None:
    """Train embedding model with MNRL + TripletLoss."""
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
    except ImportError:
        logger.error(
            "sentence-transformers required. Install: "
            "pip install 'micro-kiki[embeddings]'"
        )
        sys.exit(1)

    logger.info("Loading domain texts...")
    domain_texts = load_all_domain_texts(domains)
    active = {d: t for d, t in domain_texts.items() if t}
    if not active:
        logger.error("No training data found.")
        return

    mnrl_pairs = build_mnrl_pairs(domain_texts)
    logger.info("MNRL pairs: %d", len(mnrl_pairs))
    mnrl_examples = [InputExample(texts=[a, p]) for a, p in mnrl_pairs]

    triplets = build_hard_negative_triplets(domain_texts)
    logger.info("Hard negative triplets: %d", len(triplets))
    triplet_examples = [InputExample(texts=[a, p, n]) for a, p, n in triplets]

    logger.info("Loading all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    mnrl_loader = DataLoader(mnrl_examples, shuffle=True, batch_size=args.batch_size)
    mnrl_loss = losses.MultipleNegativesRankingLoss(model)
    train_objectives = [(mnrl_loader, mnrl_loss)]

    if triplet_examples:
        triplet_loader = DataLoader(triplet_examples, shuffle=True, batch_size=args.batch_size)
        triplet_loss = losses.TripletLoss(
            model=model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=args.margin,
        )
        train_objectives.append((triplet_loader, triplet_loss))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Training: %d epochs, batch=%d, margin=%.2f", args.epochs, args.batch_size, args.margin)
    t0 = time.time()
    model.fit(
        train_objectives=train_objectives,
        epochs=args.epochs,
        warmup_steps=int(0.1 * len(mnrl_loader) * args.epochs),
        output_path=str(OUTPUT_DIR),
        show_progress_bar=True,
    )
    logger.info("Training done in %.1fs", time.time() - t0)
    logger.info("Model saved to %s", OUTPUT_DIR)
    _eval_separation(model, domain_texts)


def _eval_separation(model, domain_texts: dict[str, list[str]]) -> None:
    """Print intra-domain vs inter-domain cosine similarity."""
    from sentence_transformers import util
    domains_with_data = [d for d, t in domain_texts.items() if len(t) >= 4]
    if len(domains_with_data) < 2:
        logger.warning("Not enough domains for eval")
        return
    intra_scores: list[float] = []
    inter_scores: list[float] = []
    for domain in domains_with_data[:5]:
        texts = domain_texts[domain][:10]
        embs = model.encode(texts)
        sims = util.cos_sim(embs, embs).numpy()
        n = len(texts)
        for i in range(n):
            for j in range(i + 1, n):
                intra_scores.append(float(sims[i][j]))
    for i, d1 in enumerate(domains_with_data[:5]):
        for d2 in domains_with_data[i + 1: i + 3]:
            e1 = model.encode(domain_texts[d1][:5])
            e2 = model.encode(domain_texts[d2][:5])
            sims = util.cos_sim(e1, e2).numpy()
            inter_scores.extend(sims.flatten().tolist())
    avg_intra = np.mean(intra_scores) if intra_scores else 0
    avg_inter = np.mean(inter_scores) if inter_scores else 0
    logger.info("Eval: intra-domain=%.3f (target>0.7), inter-domain=%.3f (target<0.3)", avg_intra, avg_inter)


def dry_run(domains: list[str]) -> None:
    """Show what would be trained without loading models."""
    logger.info("DRY RUN")
    domain_texts = load_all_domain_texts(domains)
    mnrl_pairs = build_mnrl_pairs(domain_texts)
    triplets = build_hard_negative_triplets(domain_texts)
    logger.info("MNRL pairs: %d", len(mnrl_pairs))
    logger.info("Hard negative triplets: %d", len(triplets))
    logger.info("Output: %s", OUTPUT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train domain-specific embeddings for Aeon memory.")
    parser.add_argument("--all", action="store_true", help="Train on all niche domains")
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.3, help="TripletLoss margin")
    args = parser.parse_args()
    domains = args.domains or (NICHE_DOMAINS if args.all else [])
    if not domains:
        parser.print_help()
        return
    if args.dry_run:
        dry_run(domains)
    else:
        train(domains, args)


if __name__ == "__main__":
    main()
