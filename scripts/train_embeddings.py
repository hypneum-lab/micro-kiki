#!/usr/bin/env python3
"""Fine-tune domain-specific embedding model for Aeon memory recall.

Replaces the hash-based stub in src/memory/aeon.py with a real embedding
model trained on niche domain text. Uses mlx-tune EmbeddingSFTTrainer
if available, falls back to sentence-transformers otherwise.

Usage::

    uv run python scripts/train_embeddings.py --all
    uv run python scripts/train_embeddings.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MERGED_DATA = PROJECT_ROOT / "data" / "merged"
KIKI_DATA = Path.home() / "KIKI-Mac_tunner" / "data" / "micro-kiki"
OUTPUT_DIR = PROJECT_ROOT / "models" / "niche-embeddings"

NICHE_DOMAINS = [
    "kicad-dsl", "spice", "emc", "stm32", "embedded",
    "freecad", "platformio", "power", "dsp", "electronics",
]


def load_texts(domain: str, max_n: int = 2000) -> list[str]:
    """Extract text from domain training data for embedding training."""
    texts: list[str] = []
    for p in [
        MERGED_DATA / domain / "train.jsonl",
        KIKI_DATA / domain / "train.jsonl",
    ]:
        if p.exists():
            with open(p) as f:
                for line in f:
                    d = json.loads(line.strip())
                    if "messages" in d:
                        for m in d["messages"]:
                            if m.get("content") and len(m["content"]) > 20:
                                texts.append(m["content"])
                    elif "prompt" in d:
                        texts.append(d["prompt"])
            break
    return texts[:max_n]


def build_pairs(domains: list[str]) -> tuple[list[tuple[str, str]], list[int]]:
    """Build (anchor, positive) pairs with domain labels.

    Same-domain pairs are positive (label 1), cross-domain pairs are
    created implicitly during contrastive training.
    """
    pairs: list[tuple[str, str]] = []
    labels: list[int] = []

    for domain in domains:
        texts = load_texts(domain)
        if len(texts) < 2:
            logger.warning("Not enough texts for %s (%d)", domain, len(texts))
            continue
        # Create same-domain positive pairs (consecutive texts)
        for i in range(0, len(texts) - 1, 2):
            pairs.append((texts[i], texts[i + 1]))
            labels.append(1)
        logger.info("%s: %d positive pairs from %d texts", domain, len(texts) // 2, len(texts))

    return pairs, labels


def train_with_mlx_tune(pairs: list[tuple[str, str]], labels: list[int]) -> None:
    """Train using mlx-tune EmbeddingSFTTrainer."""
    try:
        import mlx.core as mx
        mx.set_memory_limit(460 * 1024**3)
        mx.set_cache_limit(32 * 1024**3)
    except ImportError:
        pass

    try:
        from mlx_tune import FastEmbeddingModel, EmbeddingSFTConfig, EmbeddingSFTTrainer
    except ImportError:
        raise RuntimeError(
            "mlx-tune EmbeddingSFTTrainer not available. "
            "Install: pip install mlx-tune"
        )

    logger.info("Loading embedding model...")
    model, tokenizer = FastEmbeddingModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Format as dataset
    dataset = [
        {"anchor": a, "positive": p, "label": l}
        for (a, p), l in zip(pairs, labels)
    ]

    trainer = EmbeddingSFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        learning_rate=2e-5,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        output_dir=str(OUTPUT_DIR),
    )
    trainer.train()
    logger.info("Embedding model saved to %s", OUTPUT_DIR)


def dry_run(domains: list[str]) -> None:
    """Show what would be trained without loading models."""
    logger.info("DRY RUN — no model loading")
    total_pairs = 0
    for domain in domains:
        texts = load_texts(domain, max_n=100)
        n_pairs = len(texts) // 2
        total_pairs += n_pairs
        logger.info("  %s: %d texts → %d pairs", domain, len(texts), n_pairs)
    logger.info("Total: %d positive pairs across %d domains", total_pairs, len(domains))
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("Model: sentence-transformers/all-MiniLM-L6-v2 (or Qwen3-Embedding)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train domain-specific embeddings for Aeon memory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--all", action="store_true", help="Train on all niche domains")
    parser.add_argument("--domains", nargs="+", default=None, help="Specific domains")
    parser.add_argument("--dry-run", action="store_true", help="Show plan only")
    args = parser.parse_args()

    domains = args.domains or (NICHE_DOMAINS if args.all else [])
    if not domains:
        parser.print_help()
        return

    if args.dry_run:
        dry_run(domains)
        return

    pairs, labels = build_pairs(domains)
    if not pairs:
        logger.error("No training pairs generated. Check data paths.")
        return

    logger.info("Training on %d pairs from %d domains", len(pairs), len(domains))

    try:
        train_with_mlx_tune(pairs, labels)
    except RuntimeError as e:
        logger.warning("mlx-tune failed: %s", e)
        logger.info("Falling back to sentence-transformers (if installed)")
        try:
            from sentence_transformers import SentenceTransformer, InputExample, losses
            from torch.utils.data import DataLoader

            model = SentenceTransformer("all-MiniLM-L6-v2")
            train_examples = [
                InputExample(texts=[a, p], label=float(l))
                for (a, p), l in zip(pairs, labels)
            ]
            dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
            loss = losses.CosineSimilarityLoss(model)

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            model.fit(
                train_objectives=[(dataloader, loss)],
                epochs=3,
                output_path=str(OUTPUT_DIR),
            )
            logger.info("Saved to %s", OUTPUT_DIR)
        except ImportError:
            logger.error("Neither mlx-tune nor sentence-transformers available")
            sys.exit(1)


if __name__ == "__main__":
    main()
