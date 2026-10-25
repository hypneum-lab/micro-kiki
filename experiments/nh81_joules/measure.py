"""NH-81 energy protocol on macM1 (preregistration section 5).

Interleaved pairs: 20 s rest window, then one dense run and one spiking
run, order alternating between pairs. Every window is appended to
``measure.jsonl`` with wall-clock bounds (for a PI-run powermetrics log),
IOReport GPU/ANE/CPU energy deltas, GPU utilisation samples, the OrbStack VM
%CPU and the busy Python processes. Guards: N10-C stop rule before every
pair, start guard (2 min idle GPU) retried every 10 min for up to 3 h.

Usage:
    python measure.py --dense runs/ann1/ckpt_XXXXX.npz \
        --spiking runs/spkt/ckpt_12000.npz --out results/measure.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time

import mlx.core as mx
import numpy as np

import power
from model_mlx import LM

HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = bool(int(os.environ.get("NH81_QUICK", "0")))  # code dry runs only
REST_S = 2.0 if QUICK else 20.0
TARGET_RUN_S = 2.0 if QUICK else 15.0


class UtilSampler:
    """Background ioreg GPU-utilisation sampler (every 0.5 s)."""

    def __init__(self) -> None:
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.samples.append(power.gpu_utilization())
            self._stop.wait(0.5)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


class Log:
    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.f = open(path, "a")

    def __call__(self, **rec) -> None:
        rec["logged"] = time.time()
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        print(json.dumps(rec), flush=True)


def load(path: str, arm: str) -> LM:
    m = LM(arm=arm)
    m.load_weights(path)
    m.eval()
    mx.eval(m.parameters())
    return m


def encode_prompts() -> tuple[list[list[int]], int]:
    from tokenizers import Tokenizer

    cfg = json.load(open(os.path.join(HERE, "prompts.json")))
    tk = Tokenizer.from_file(os.path.expanduser(os.environ.get("NH81_BPE", "~/scratch/nh81/bsk_bpe.json")))
    return [tk.encode(p).ids for p in cfg["prompts"]], int(cfg["new_tokens"])


class Workload:
    """W1 generation (batch 1, greedy, sliding window) and W2 scoring."""

    def __init__(self, model: LM, mode: str, prompts, new_tokens, test: np.ndarray) -> None:
        self.model, self.prompts, self.new_tokens = model, prompts, new_tokens
        self.f = mx.compile(lambda x: model(x, mode=mode))
        rng = np.random.default_rng(3)
        st = rng.integers(0, test.size - 129, size=32 * 64)
        self.score_batches = [mx.array(np.stack([test[s:s + 128] for s in st[i:i + 32]]))
                              for i in range(0, len(st), 32)]

    def generate(self, passes: int) -> int:
        n = 0
        for _ in range(passes):
            for p in self.prompts:
                ids = list(p)
                for _ in range(self.new_tokens):
                    logits = self.f(mx.array([ids[-128:]]))
                    ids.append(int(mx.argmax(logits[0, -1]).item()))
                    n += 1
        return n

    def score(self, nbatch: int) -> int:
        n = 0
        for i in range(nbatch):
            x = self.score_batches[i % len(self.score_batches)]
            mx.eval(self.f(x))
            n += x.size
        return n


def window(meter: power.IOReportEnergy, fn) -> dict:
    """Run ``fn`` (returns token count) inside an energy window."""
    vm0 = power.vm_cpu_percent()
    busy0 = power.busy_python()
    with UtilSampler() as us:
        a = meter.snapshot()
        tokens = fn()
        b = meter.snapshot()
    rails = power.rail_energy(meter.delta_joules(a, b))
    return {"t0": a[0], "t1": b[0], "dur_s": b[0] - a[0], "tokens": tokens, **rails,
            "gpu_util_mean": float(np.mean(us.samples)) if us.samples else -1,
            "gpu_util_max": max(us.samples) if us.samples else -1,
            "vm_cpu_pct": (vm0 + power.vm_cpu_percent()) / 2,
            "busy_python": [r[2] for r in busy0 + power.busy_python()]}


def rest(meter: power.IOReportEnergy) -> dict:
    return window(meter, lambda: (time.sleep(REST_S), 0)[1])


def start_guard(log: Log, max_wait_s: float = 3 * 3600) -> bool:
    t_start = time.time()
    while time.time() - t_start < max_wait_s:
        reason = power.n10c_active()
        if reason:
            log(kind="abort", reason=f"N10-C: {reason}")
            sys.exit(3)
        ok, t0, samples = True, time.time(), []
        while time.time() - t0 < 120:
            u = power.gpu_utilization()
            bp = power.busy_python()
            samples.append(u)
            if u >= 10 or bp:
                ok = False
                log(kind="guard_wait", gpu_util=u, busy_python=[r[2] for r in bp],
                    vm_cpu_pct=power.vm_cpu_percent())
                break
            time.sleep(5)
        if ok:
            log(kind="guard_ok", gpu_util_samples=samples)
            return True
        time.sleep(600)
    log(kind="abort", reason="macM1 busy for 3 h, not measured")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True)
    ap.add_argument("--spiking", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--test", default=os.path.expanduser("~/scratch/nh81/split/test.i32"))
    ap.add_argument("--skip-guard", action="store_true", help="tests only")
    a = ap.parse_args()

    mx.set_memory_limit(6 * 2**30)
    log = Log(a.out)
    log(kind="start", dense=a.dense, spiking=a.spiking, pid=os.getpid(),
        powermetrics_log=os.path.exists(os.path.expanduser("~/scratch/nh81/pm.plist")))
    if not a.skip_guard and not start_guard(log):
        return 2
    meter = power.IOReportEnergy()
    prompts, new_tokens = encode_prompts()
    test = np.fromfile(a.test, dtype=np.int32)
    dense = load(a.dense, "ann1")
    spk = load(a.spiking, "spkt")
    wl = {"dense": Workload(dense, "fused", prompts, new_tokens, test),
          "fused": Workload(spk, "fused", prompts, new_tokens, test),
          "event": Workload(spk, "event", prompts, new_tokens, test)}

    # Unmeasured warm-up: visits every window length once per model/path.
    for w in wl.values():
        w.generate(1)
        w.score(2)
    # Calibration on the dense model (same K / batch count for every arm).
    t = time.time(); wl["dense"].generate(1); t_gen = time.time() - t
    t = time.time(); wl["dense"].score(8); t_sc = (time.time() - t) / 8
    k_gen = max(1, math.ceil(TARGET_RUN_S / t_gen))
    n_sc = max(8, math.ceil(TARGET_RUN_S / t_sc))
    log(kind="calibration", t_gen_pass_s=t_gen, k_gen=k_gen, t_score_batch_s=t_sc, n_score=n_sc)

    blocks = [("W1", "fused"), ("W2", "fused"), ("W1", "event"), ("W2", "event")]
    for workload, path in blocks:
        run = (lambda w: (lambda: w.generate(k_gen))) if workload == "W1" else \
              (lambda w: (lambda: w.score(n_sc)))
        for i in range(a.pairs):
            reason = power.n10c_active()
            if reason:
                log(kind="abort", reason=f"N10-C: {reason}")
                return 3
            log(kind="rest", workload=workload, path=path, pair=i, **rest(meter))
            order = ["dense", path] if i % 2 == 0 else [path, "dense"]
            for arm in order:
                log(kind="run", workload=workload, path=path, pair=i, arm=arm,
                    **window(meter, run(wl[arm])))
        log(kind="rest", workload=workload, path=path, pair=a.pairs, **rest(meter))
    log(kind="done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
