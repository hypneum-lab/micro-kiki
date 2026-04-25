"""Router ablation: compare 3 routing strategies on the same validation set.

Usage:
    python scripts/eval_router_ablation.py \
        --valid data/router-v4/valid.jsonl \
        --router-weights output/router-v4 \
        --output results/ablation-router/router-ablation.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# --- Strategy 1: Keyword-based (domain name as keyword, word-boundary match) ---
def _keyword_route(text: str, domain_keywords: dict[str, list[str]]) -> list[str]:
    """Score text against domain keywords with word-boundary matching."""
    scores: dict[str, float] = {}
    text_lower = text.lower()
    for domain, keywords in domain_keywords.items():
        score = sum(
            1.0 for kw in keywords
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
        )
        if score >= 1.0:
            scores[domain] = score
    if not scores:
        return []
    ranked = sorted(scores, key=scores.get, reverse=True)
    return ranked[:4]


# --- Strategy 2: MiniLM+MLP (current router V4) ---
def _load_minilm_router(weights_dir: Path):
    """Load the trained router."""
    from safetensors.numpy import load_file
    from sentence_transformers import SentenceTransformer

    meta = json.loads((weights_dir / "meta.json").read_text())
    tensors = load_file(str(weights_dir / "router.safetensors"))
    w0 = tensors["0.weight"]
    b0 = tensors["0.bias"]
    w1 = tensors["3.weight"]
    b1 = tensors["3.bias"]
    domains = meta["domains"]
    encoder = SentenceTransformer(
        meta.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    )

    def route(text: str) -> list[str]:
        emb = encoder.encode(text, normalize_embeddings=True)
        h = np.maximum(w0 @ emb + b0, 0)
        logits = w1 @ h + b1
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        active = np.where(probs > 0.12)[0]
        if len(active) == 0:
            return []
        active = active[np.argsort(probs[active])[::-1]][:4]
        return [domains[i] for i in active]

    return route


# --- Strategy 3: Random baseline ---
def _random_route(domains: list[str]) -> list[str]:
    return [random.choice(domains)]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--router-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/ablation-router/router-ablation.json"))
    args = parser.parse_args()

    # Load validation data
    examples = []
    for line in args.valid.read_text().strip().split("\n"):
        if not line:
            continue
        e = json.loads(line)
        examples.append({"prompt": e["prompt"], "domain": e["domain"]})
    logger.info("Loaded %d validation examples", len(examples))

    domains = sorted(set(e["domain"] for e in examples))
    logger.info("Domains: %d unique", len(domains))

    # Load MiniLM router
    logger.info("Loading MiniLM router from %s", args.router_weights)
    minilm_route = _load_minilm_router(args.router_weights)
    logger.info("MiniLM router loaded")

    # Build keyword config: domain name as keyword (domain name + hyphen-free variant)
    domain_keywords = {d: [d.replace("-", " "), d] for d in domains}

    # Run all strategies
    results: dict[str, list[dict]] = {"keyword": [], "minilm": [], "random": []}
    strategies = ["keyword", "minilm", "random"]

    t0 = time.time()
    for i, ex in enumerate(examples):
        prompt, gold = ex["prompt"], ex["domain"]

        # Keyword
        kw_pred = _keyword_route(prompt, domain_keywords)
        results["keyword"].append({"gold": gold, "pred": kw_pred})

        # MiniLM
        ml_pred = minilm_route(prompt)
        results["minilm"].append({"gold": gold, "pred": ml_pred})

        # Random
        rnd_pred = _random_route(domains)
        results["random"].append({"gold": gold, "pred": rnd_pred})

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            logger.info("Processed %d/%d (%.1fs)", i + 1, len(examples), elapsed)

    # Compute metrics
    summary = {}
    for strategy in strategies:
        preds = results[strategy]
        top1 = sum(1 for p in preds if p["pred"] and p["pred"][0] == p["gold"]) / len(preds)
        top3 = sum(1 for p in preds if p["gold"] in p["pred"][:3]) / len(preds)
        coverage = sum(1 for p in preds if len(p["pred"]) > 0) / len(preds)
        summary[strategy] = {
            "top1": round(top1, 4),
            "top3": round(top3, 4),
            "coverage": round(coverage, 4),
            "n": len(preds),
        }
        logger.info(
            "%s: top1=%.3f  top3=%.3f  coverage=%.3f",
            strategy, top1, top3, coverage,
        )

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_examples": len(examples),
        "n_domains": len(domains),
        "strategies": summary,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    logger.info("Results saved to %s", args.output)

    # Print summary table
    print("\n=== Router Ablation Results ===")
    print(f"{'Strategy':<12} {'Top-1':>8} {'Top-3':>8} {'Coverage':>10}")
    print("-" * 42)
    for strat, m in summary.items():
        print(f"{strat:<12} {m['top1']:>8.3f} {m['top3']:>8.3f} {m['coverage']:>10.3f}")
    print(f"\nn={len(examples)}, {len(domains)} domains")


if __name__ == "__main__":
    main()
