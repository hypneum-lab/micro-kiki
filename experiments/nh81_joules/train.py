"""Train one NH-81 arm (dense ``ann1`` or spiking ``spkt``) with MLX.

Both arms see the identical batch sequence (same seed, same sampler).
Every ``--ckpt-every`` steps the weights are saved to ``ckpt_<step>.npz``
and the val / test losses (nats/token) are evaluated on fixed windows.
Memory is capped (default 6 GB) so the run stays inside the macM1 budget.

Usage (macM1, nice 19):
    python train.py --arm ann1 --data split/ --out runs/ann1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from data import eval_windows
from model_mlx import LM, n_params

EVAL_SEED = {"val": 1, "test": 2}
EVAL_COUNT = 512
EVAL_BS = 32


def held_out_loss(model: LM, data: np.ndarray, seq: int, seed: int) -> tuple[float, float]:
    starts = eval_windows(data.size, seq, EVAL_COUNT, seed)
    tot, n, fire = 0.0, 0, []
    for i in range(0, len(starts), EVAL_BS):
        st = starts[i:i + EVAL_BS]
        x = mx.array(np.stack([data[s:s + seq] for s in st]))
        y = mx.array(np.stack([data[s + 1:s + seq + 1] for s in st]))
        stats: list = []
        logits = model(x, stats=stats)
        ce = nn.losses.cross_entropy(logits, y, reduction="sum")
        tot += ce.item()
        n += y.size
        if stats:
            fire.append(mx.stack(stats).mean().item())
    return tot / n, float(np.mean(fire)) if fire else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ann1", "spkt"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ticks", type=int, default=6)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--mem-gb", type=float, default=6.0)
    ap.add_argument("--spike-l1", type=float, default=0.0,
                    help="weight of the firing-rate penalty (BSK HL_SPIKE_L1)")
    a = ap.parse_args()

    mx.set_memory_limit(int(a.mem_gb * 2**30))
    mx.set_cache_limit(int(2**30))
    info = json.load(open(os.path.join(a.data, "split_info.json")))
    train = np.fromfile(os.path.join(a.data, "train.i32"), dtype=np.int32)
    held = {k: np.fromfile(os.path.join(a.data, f"{k}.i32"), dtype=np.int32) for k in ("val", "test")}
    rng = np.random.default_rng(a.seed)
    mx.random.seed(a.seed)
    model = LM(info["vocab"], arm=a.arm, ticks=a.ticks)
    mx.eval(model.parameters())
    seq = model.cfg["SEQ"]
    opt = optim.AdamW(learning_rate=a.lr, betas=[0.9, 0.999], weight_decay=0.0)
    os.makedirs(a.out, exist_ok=True)
    json.dump({**model.cfg, **vars(a), "params": n_params(model)},
              open(os.path.join(a.out, "config.json"), "w"), indent=1)
    log = open(os.path.join(a.out, "log.jsonl"), "a")

    def loss_fn(m, x, y):
        stats: list = []
        ce = nn.losses.cross_entropy(m(x, train=True, stats=stats), y, reduction="mean")
        if a.spike_l1 > 0 and stats:
            return ce + a.spike_l1 * mx.stack(stats).mean()
        return ce

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        opt.update(model, grads)
        return loss

    def checkpoint(t: int) -> None:
        rec = {"step": t, "time": time.time()}
        for k, d in held.items():
            rec[f"{k}_nats"], rec[f"{k}_fire"] = held_out_loss(model, d, seq, EVAL_SEED[k])
        model.save_weights(os.path.join(a.out, f"ckpt_{t:05d}.npz"))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    t0 = time.time()
    for t in range(1, a.steps + 1):
        ix = rng.integers(0, train.size - seq - 1, size=a.bs)
        x = mx.array(np.stack([train[i:i + seq] for i in ix]))
        y = mx.array(np.stack([train[i + 1:i + seq + 1] for i in ix]))
        loss = step(x, y)
        mx.eval(state)
        if t % 100 == 0:
            print(f"step {t} loss={loss.item():.3f} s/step={(time.time() - t0) / t:.3f} "
                  f"peak={mx.get_peak_memory() / 2**30:.2f}GB", flush=True)
        if t % a.ckpt_every == 0 or t == a.steps:
            checkpoint(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
