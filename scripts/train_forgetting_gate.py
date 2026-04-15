#!/usr/bin/env python
"""Train the AeonSleep forgetting-gate MLP (story-9).

Reproducible trainer for the small MLP in
:mod:`src.cognitive.forgetting_gate`. Writes the dataset to
``data/forgetting-pairs.jsonl`` and the trained weights to
``models/forgetting-gate.npz`` (gitignored except the dataset).

Usage::

    uv run python scripts/train_forgetting_gate.py
    uv run python scripts/train_forgetting_gate.py \
        --n 4000 --epochs 400 --out models/forgetting-gate.npz

The script is deterministic given a fixed ``--seed``. It prints an
F1 score on a held-out 20% test split and exits with status 1 if F1
falls below the acceptance threshold (0.85).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.cognitive.forgetting_gate import (
    ForgettingGate,
    f1_score,
    generate_synthetic_pairs,
    read_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=2000,
                   help="total synthetic pairs to generate")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.85,
                   help="minimum F1 on test split")
    p.add_argument("--data-out", type=Path,
                   default=Path("data/forgetting-pairs.jsonl"))
    p.add_argument("--weights-out", type=Path,
                   default=Path("models/forgetting-gate.npz"))
    p.add_argument("--report-out", type=Path,
                   default=Path("results/forgetting-gate-train.json"))
    p.add_argument("--skip-data-write", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    feats, labels = generate_synthetic_pairs(n=args.n, seed=args.seed)

    if not args.skip_data_write:
        write_jsonl(args.data_out, feats, labels)
        # Sanity check the IO roundtrip.
        feats, labels = read_jsonl(args.data_out)

    split = int(0.8 * len(feats))
    train_x, test_x = feats[:split], feats[split:]
    train_y, test_y = labels[:split], labels[split:]

    gate = ForgettingGate(seed=args.seed)
    history = gate.fit(
        train_x, train_y,
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        seed=args.seed, verbose=False,
    )
    preds = gate.predict(test_x)
    score = f1_score(test_y, preds)

    args.weights_out.parent.mkdir(parents=True, exist_ok=True)
    gate.params.save(args.weights_out)

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "n_total": len(feats),
        "n_train": split,
        "n_test": len(feats) - split,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "final_loss": history[-1],
        "f1": score,
        "threshold": args.threshold,
    }
    args.report_out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if score < args.threshold:
        print(f"FAIL: F1 {score:.3f} < threshold {args.threshold}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
