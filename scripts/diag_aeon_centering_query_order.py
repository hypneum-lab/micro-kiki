#!/usr/bin/env python3
"""Diagnostic: does the synthetic centering gain depend on evaluation-time state?

`LatentMLP.forward` updates the centering running mean on every call, including
the single-query calls made by `predict_next` during evaluation. This script
re-scores the synthetic conditions of `scripts/eval_aeon_predictor.py`
(stack-structured stream, seed 0) on the same trained weights while changing
only how that state evolves during evaluation:

  as_run       queries in stream order, predictive call then null call
               (the protocol of eval_aeon_predictor.py)
  null_first   same order, null call before the predictive call
  reversed     queries in reverse stream order
  shuffled     queries in a random order (one entry per shuffle seed)
  frozen       centering state restored to its post-training value before
               every call (no evaluation-time update)

The "frozen" variant is not confounded by the update that `forward` still
performs inside each call: with a batch of one and momentum 0.9 the output is
out - (0.9*m + 0.1*out) = 0.9*(out - m), i.e. the output centered by the frozen
mean m up to a positive scale, and cosine retrieval is scale-invariant.

and, for shared centering only, a control without any MLP:

  no_mlp_ema   retrieval with (raw query - EMA of the raw queries seen so far)

It also records, per query (in stream order), the rank of the gold turn and
which turn holds second place (the previous turn i-1, the gold turn i+1, or
another one), and `final_train_loss` for three (epochs, lr) settings, to
document which settings reproduce the committed April result files.

numpy emits spurious "encountered in matmul" RuntimeWarnings on some BLAS
builds (they occur here for every condition, including C); only that message
is silenced, and the JSON records that weights and outputs are finite.

Exploratory, single seed, same limits as the runs it re-scores (query stored in
the index, queries part of the training pairs). No new training recipe.

Usage:
    uv run python scripts/diag_aeon_centering_query_order.py \\
        --out results/2026-11-24-aeon-centering-query-order.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_aeon_predictor import (  # noqa: E402
    _build_stack_structured_stream,
    _reciprocal_rank,
)

from src.memory.aeon_predictor import AeonPredictor, PredictorConfig  # noqa: E402
from src.memory.aeonsleep import AeonSleep  # noqa: E402

N_STACKS = 16
MOMENTUM = 0.9

# label -> (use_centering, per_stack_centering, epochs, lr, committed result file)
CONDITIONS: dict[str, tuple[bool, bool, int, float, str]] = {
    "C": (False, False, 50, 1e-3, "2026-04-19-aeon-baseline-no-centering.json"),
    "D": (True, False, 50, 1e-3, "2026-04-19-aeon-baseline-shared-centering.json"),
    "F1": (True, True, 50, 1e-3, "2026-04-19-aeon-poc-F1-per-stack.json"),
    "F2": (True, True, 300, 5e-3, "2026-04-19-aeon-poc-F2-per-stack-long.json"),
}


def _build(
    use_centering: bool,
    per_stack: bool,
    epochs: int,
    lr: float,
    seed: int,
    n_turns: int,
    dim: int,
    cold_start: int,
) -> tuple[AeonSleep, AeonPredictor, list[np.ndarray], list[int], float]:
    """Same construction as eval_aeon_predictor.py --stream stack-structured."""
    palace = AeonSleep(dim=dim)
    cfg = PredictorConfig(
        dim=dim,
        hidden=min(256, dim),
        n_stacks=N_STACKS,
        cold_start_threshold=cold_start,
        seed=seed,
        use_centering=use_centering,
        centering_momentum=MOMENTUM,
        per_stack_centering=per_stack,
    )
    pred = AeonPredictor(palace=palace, config=cfg)
    palace.attach_predictor(pred)
    stream, transition_stacks = _build_stack_structured_stream(
        n_turns, dim, n_stacks=N_STACKS, seed=seed
    )
    stack_ids = [0] + transition_stacks
    t0 = datetime(2026, 4, 17, 10, 0)
    for i, h in enumerate(stream):
        pred.ingest_latent(f"t{i}", h, ts=t0 + timedelta(seconds=i), stack_id=stack_ids[i])
    history = pred.fit_on_buffer(lr=lr, epochs=epochs, batch_size=32)
    return palace, pred, stream, stack_ids, float(history[-1])


def _snapshot(pred: AeonPredictor) -> dict[str, np.ndarray]:
    state = {"shared": pred.mlp._running_mean.copy()}
    if pred.mlp.per_stack_centering:
        state["per_stack"] = pred.mlp._running_means_per_stack.copy()
    return state


def _restore(pred: AeonPredictor, state: dict[str, np.ndarray]) -> None:
    pred.mlp._running_mean = state["shared"].copy()
    if "per_stack" in state:
        pred.mlp._running_means_per_stack = state["per_stack"].copy()


def _top5(palace: AeonSleep, vec: np.ndarray) -> list[str]:
    return [hit.episode_id for hit in palace.recall(vec.tolist(), k=5)]


def _rank_hist(ids_per_query: list[list[str]], golds: list[str]) -> list[int]:
    """Counts of gold rank: index 0 = absent from top 5, index r = rank r."""
    hist = [0] * 6
    for ids, gold in zip(ids_per_query, golds):
        hist[ids.index(gold) + 1 if gold in ids else 0] += 1
    return hist


def _per_query(ids_by_turn: dict[int, list[str]]) -> dict:
    """Per-query detail, in stream order whatever the scoring order.

    gold_ranks: one digit per query, rank of turn i+1 in the top 5 (0 = absent).
    second_place: which turn is at rank 2 (previous turn i-1, gold turn i+1, other).
    """
    ranks = []
    second = {"previous_turn": 0, "gold_turn": 0, "other": 0}
    for i in sorted(ids_by_turn):
        ids = ids_by_turn[i]
        gold = f"t{i + 1}"
        ranks.append(str(ids.index(gold) + 1 if gold in ids else 0))
        if ids[1] == f"t{i - 1}":
            second["previous_turn"] += 1
        elif ids[1] == gold:
            second["gold_turn"] += 1
        else:
            second["other"] += 1
    return {"gold_ranks": "".join(ranks), "second_place": second}


def _score(
    palace: AeonSleep,
    pred: AeonPredictor,
    stream: list[np.ndarray],
    stack_ids: list[int],
    order: list[int],
    post_train: dict[str, np.ndarray],
    *,
    null_first: bool = False,
    frozen: bool = False,
) -> dict:
    _restore(pred, post_train)
    pred_ids: list[list[str]] = []
    null_ids: list[list[str]] = []
    golds: list[str] = []
    query_top1 = 0
    pred_by_turn: dict[int, list[str]] = {}
    null_by_turn: dict[int, list[str]] = {}
    finite = True
    for i in order:
        h_q = stream[i]
        calls = [("pred", stack_ids[i]), ("null", -1)]
        if null_first:
            calls.reverse()
        got: dict[str, list[str]] = {}
        for tag, sid in calls:
            if frozen:
                _restore(pred, post_train)
            vec = pred.predict_next(h_q, horizon=1, stack_id=sid)
            finite = finite and bool(np.isfinite(vec).all())
            got[tag] = _top5(palace, vec)
        pred_by_turn[i] = got["pred"]
        null_by_turn[i] = got["null"]
        pred_ids.append(got["pred"])
        null_ids.append(got["null"])
        golds.append(f"t{i + 1}")
        query_top1 += got["pred"][0] == f"t{i}"
    pred_rr = [_reciprocal_rank(ids, g) for ids, g in zip(pred_ids, golds)]
    null_rr = [_reciprocal_rank(ids, g) for ids, g in zip(null_ids, golds)]
    return {
        "predictive_mrr": float(np.mean(pred_rr)),
        "null_stack_mrr": float(np.mean(null_rr)),
        "win_rate_stack_vs_null": float(np.mean([p > n for p, n in zip(pred_rr, null_rr)])),
        "predictive_gold_rank_hist": _rank_hist(pred_ids, golds),
        "null_gold_rank_hist": _rank_hist(null_ids, golds),
        "predictive_query_itself_at_rank_1": int(query_top1),
        "predictive_per_query": _per_query(pred_by_turn),
        "null_per_query": _per_query(null_by_turn),
        "outputs_all_finite": finite,
    }


def _no_mlp_ema(
    palace: AeonSleep,
    stream: list[np.ndarray],
    order: list[int],
    init: np.ndarray,
    updates_per_query: int,
) -> dict:
    """Raw query minus an EMA of the raw queries seen so far. No predictor."""
    mean = init.astype(np.float32).copy()
    ids_all: list[list[str]] = []
    golds: list[str] = []
    by_turn: dict[int, list[str]] = {}
    for i in order:
        h_q = stream[i].astype(np.float32)
        for _ in range(updates_per_query):
            mean = (MOMENTUM * mean + (1.0 - MOMENTUM) * h_q).astype(np.float32)
        ids_all.append(_top5(palace, h_q - mean))
        by_turn[i] = ids_all[-1]
        golds.append(f"t{i + 1}")
    rr = [_reciprocal_rank(ids, g) for ids, g in zip(ids_all, golds)]
    return {
        "mrr": float(np.mean(rr)),
        "gold_rank_hist": _rank_hist(ids_all, golds),
        "per_query": _per_query(by_turn),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-turns", type=int, default=1000)
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--cold-start", type=int, default=500)
    ap.add_argument("--shuffle-seeds", default="1,2,3,4,5")
    ap.add_argument("--results-dir", type=Path, default=_PROJECT_ROOT / "results")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    warnings.filterwarnings(
        "ignore", message=".*encountered in matmul", category=RuntimeWarning
    )

    held_start = max(1, args.n_turns - args.n_queries - 1)
    natural = [i for i in range(held_start, held_start + args.n_queries) if i + 1 < args.n_turns]
    shuffle_seeds = [int(s) for s in args.shuffle_seeds.split(",") if s.strip()]

    payload: dict = {
        "script": "scripts/diag_aeon_centering_query_order.py",
        "date": "2026-11-24",
        "status": "exploratory diagnostic, single seed, re-scores existing conditions",
        "seed": args.seed,
        "n_turns": args.n_turns,
        "n_queries": len(natural),
        "dim": args.dim,
        "centering_momentum": MOMENTUM,
        "stream_type": "stack-structured",
        "shuffle_seeds": shuffle_seeds,
        "conditions": {},
    }

    for label, (use_c, per_stack, epochs, lr, committed) in CONDITIONS.items():
        palace, pred, stream, stack_ids, loss = _build(
            use_c, per_stack, epochs, lr, args.seed, args.n_turns, args.dim, args.cold_start
        )
        post_train = _snapshot(pred)

        def score(order: list[int], **kw) -> dict:
            return _score(palace, pred, stream, stack_ids, order, post_train, **kw)

        base_ids = [_top5(palace, stream[i]) for i in natural]
        golds = [f"t{i + 1}" for i in natural]
        entry: dict = {
            "use_centering": use_c,
            "per_stack_centering": per_stack,
            "epochs": epochs,
            "lr": lr,
            "final_train_loss": loss,
            "weights_all_finite": bool(
                all(
                    np.isfinite(getattr(pred.mlp, name)).all()
                    for name in ("w1", "b1", "w2", "b2", "w3", "b3")
                )
            ),
            "post_train_shared_running_mean_norm": float(np.linalg.norm(post_train["shared"])),
            "baseline_mrr": float(
                np.mean([_reciprocal_rank(ids, g) for ids, g in zip(base_ids, golds)])
            ),
            "baseline_gold_rank_hist": _rank_hist(base_ids, golds),
            "baseline_query_itself_at_rank_1": int(
                sum(ids[0] == f"t{i}" for ids, i in zip(base_ids, natural))
            ),
            "baseline_per_query": _per_query(dict(zip(natural, base_ids))),
            "variants": {
                "as_run": score(natural),
                "null_first": score(natural, null_first=True),
                "reversed": score(natural[::-1]),
                "frozen": score(natural, frozen=True),
                "shuffled": {},
            },
        }
        if "per_stack" in post_train:
            entry["post_train_null_slot_norm"] = float(np.linalg.norm(post_train["per_stack"][0]))
        for s in shuffle_seeds:
            order = list(natural)
            np.random.default_rng(s).shuffle(order)
            entry["variants"]["shuffled"][str(s)] = score(order)

        if use_c and not per_stack:
            entry["no_mlp_ema"] = {
                f"init_{init_name}_updates_{k}": _no_mlp_ema(palace, stream, natural, init, k)
                for init_name, init in (
                    ("post_train_mean", post_train["shared"]),
                    ("zero", np.zeros(args.dim, dtype=np.float32)),
                )
                for k in (1, 2)
            }
            as_run_ranks = entry["variants"]["as_run"]["predictive_per_query"]["gold_ranks"]
            for ema in entry["no_mlp_ema"].values():
                ema["gold_ranks_identical_to_as_run_predictive"] = bool(
                    ema["per_query"]["gold_ranks"] == as_run_ranks
                )

        committed_path = args.results_dir / committed
        if committed_path.exists():
            ref = json.loads(committed_path.read_text())
            run = entry["variants"]["as_run"]
            entry["committed_file"] = f"results/{committed}"
            entry["as_run_matches_committed_file"] = bool(
                run["predictive_mrr"] == ref["predictive_mrr"]
                and run["null_stack_mrr"] == ref["null_stack_mrr"]
                and run["win_rate_stack_vs_null"] == ref["win_rate_stack_vs_null"]
                and loss == ref["final_train_loss"]
                and entry["baseline_mrr"] == ref["baseline_mrr"]
            )
        payload["conditions"][label] = entry

    # Which (epochs, lr) reproduce the committed final_train_loss values.
    # The loss does not depend on the centering / LayerNorm flags.
    payload["training_settings_check"] = {
        "50_epochs_lr_1e-3": payload["conditions"]["C"]["final_train_loss"],
        "300_epochs_lr_5e-3": payload["conditions"]["F2"]["final_train_loss"],
        "300_epochs_lr_1e-3": _build(
            False, False, 300, 1e-3, args.seed, args.n_turns, args.dim, args.cold_start
        )[4],
    }

    text = json.dumps(payload, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
