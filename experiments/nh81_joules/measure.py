"""NH-81 energy protocol on macM1 (preregistration section 5).

Interleaved pairs: 20 s rest window, then one dense run and one spiking
run, order alternating between pairs. Every window is appended to
``measure.jsonl`` with wall-clock bounds (for a PI-run powermetrics log),
IOReport GPU/ANE/CPU energy deltas, GPU utilisation samples, the OrbStack VM
%CPU and the busy Python processes. Guards: N10-C stop rule before every
pair, start guard (2 min idle GPU) retried every 10 min for up to 3 h.

Hosts: ``NH81_HOST=macm1`` (default; IOReport, ``power``) or
``NH81_HOST=kxkm`` (NVML on the RTX 4090, ``power_nvml``; replication
prereg 2026-10-28: foreign GPU compute processes reject a pair, rejected
pairs are replayed, and the N10-C stop rule becomes a pause rule).

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

from model_mlx import LM

HOST = os.environ.get("NH81_HOST", "macm1")
if HOST == "kxkm":
    import power_nvml as power
else:
    import power
import analyze

HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = bool(int(os.environ.get("NH81_QUICK", "0")))  # code dry runs only
REST_S = 2.0 if QUICK else 20.0
TARGET_RUN_S = 2.0 if QUICK else 15.0
CLEAN_S = 6.0 if QUICK else 120.0          # clean time before start / resume
MAX_PAUSE_S = 3 * 3600                    # cumulative pause budget (kxkm)


class UtilSampler:
    """Background ioreg GPU-utilisation sampler (every 0.5 s)."""

    def __init__(self) -> None:
        self.samples: list[int] = []
        self.pids: set[int] = set()
        self.state: list[dict] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.samples.append(power.gpu_utilization())
            if hasattr(power, "foreign_gpu_pids"):
                self.pids |= power.foreign_gpu_pids()
                self.state.append(power.gpu_state())
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
    ids_path = os.environ.get("NH81_PROMPT_IDS")
    if ids_path:  # pre-tokenised on macM1 (prompts_ids.json), same BPE
        cfg = json.load(open(ids_path))
        return cfg["ids"], int(cfg["new_tokens"])
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
    ticks = hasattr(power, "python_ticks")
    busy0 = [] if ticks else power.busy_python()
    tk0 = power.python_ticks() if ticks else None
    with UtilSampler() as us:
        a = meter.snapshot()
        tokens = fn()
        b = meter.snapshot()
    if ticks:  # %CPU of other Python processes over the window itself
        busy = power.busy_between(tk0, power.python_ticks(), b[0] - a[0])
    else:
        busy = busy0 + power.busy_python()
    rails = power.rail_energy(meter.delta_joules(a, b))
    rec = {"t0": a[0], "t1": b[0], "dur_s": b[0] - a[0], "tokens": tokens, **rails,
           "gpu_util_mean": float(np.mean(us.samples)) if us.samples else -1,
           "gpu_util_max": max(us.samples) if us.samples else -1,
           "vm_cpu_pct": (vm0 + power.vm_cpu_percent()) / 2,
           "busy_python": [r[2] for r in busy]}
    if hasattr(power, "foreign_gpu_pids"):
        rec["foreign_gpu_pids"] = sorted(us.pids)
        for k in ("power_w", "sm_mhz", "temp_c", "mem_used_mb"):
            v = [st[k] for st in us.state]
            rec[f"{k}_mean"] = float(np.mean(v)) if v else -1
        rec["pstates"] = sorted({st["pstate"] for st in us.state})
    return rec


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
        while time.time() - t0 < CLEAN_S:
            u = power.gpu_utilization()
            bp = power.busy_python()
            samples.append(u)
            if u < 0 or u >= 10 or bp:  # u = -1: read failed, fail closed
                ok = False
                log(kind="guard_wait", gpu_util=u, busy_python=[r[2] for r in bp],
                    vm_cpu_pct=power.vm_cpu_percent())
                break
            time.sleep(5)
        if ok:
            extra = {"baseline_gpu_pids": power.set_baseline()} if hasattr(power, "set_baseline") else {}
            log(kind="guard_ok", gpu_util_samples=samples, **extra)
            return True
        time.sleep(600)
    log(kind="abort", reason="macM1 busy for 3 h, not measured")
    return False


class Pauser:
    """Pause rule (kxkm): wait until clean for CLEAN_S, within MAX_PAUSE_S."""

    def __init__(self, log: Log) -> None:
        self.log, self.used = log, 0.0

    def wait(self) -> bool:
        """True if the pair may start now; False after a pause (the caller
        then takes a fresh rest). Exits with code 4 over the pause budget."""
        if not power.pause_reason():
            return True
        t_start = time.time()
        clean_since = None
        while True:
            r = power.pause_reason()
            now = time.time()
            if r:
                clean_since = None
                self.log(kind="pause", reason=r, paused_s=self.used + now - t_start)
            elif clean_since is None:
                clean_since = now
            elif now - clean_since >= CLEAN_S:
                self.used += now - t_start
                self.log(kind="resume", paused_s=self.used)
                return False
            if self.used + now - t_start > MAX_PAUSE_S:
                self.log(kind="abort", reason="KXKM busy: pause budget (3 h) exhausted")
                raise SystemExit(4)
            time.sleep(5 if clean_since is not None else (2 if QUICK else 60))


def pair_valid(before: dict, runs: list[dict], after: dict) -> bool:
    return (analyze.rest_ok(before) and analyze.rest_ok(after) and len(runs) == 2
            and all(analyze.window_ok(r) for r in runs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True)
    ap.add_argument("--spiking", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=12, help="valid pairs per block")
    ap.add_argument("--max-attempts", type=int, default=24, help="pair attempts per block")
    ap.add_argument("--test", default=os.path.expanduser("~/scratch/nh81/split/test.i32"))
    ap.add_argument("--skip-guard", action="store_true", help="tests only")
    a = ap.parse_args()

    mx.set_memory_limit(6 * 2**30)
    log = Log(a.out)
    log(kind="start", dense=a.dense, spiking=a.spiking, pid=os.getpid(),
        host=HOST, mlx=mx.__version__, device=str(mx.default_device()),
        tf32=os.environ.get("MLX_ENABLE_TF32", "default"),
        powermetrics_log=os.path.exists(os.path.expanduser("~/scratch/nh81/pm.plist")))
    if not a.skip_guard and not start_guard(log):
        return 2
    if a.skip_guard and hasattr(power, "set_baseline"):  # dry runs only
        log(kind="guard_skipped", baseline_gpu_pids=power.set_baseline())
    meter = power.EnergyMeter() if hasattr(power, "EnergyMeter") else power.IOReportEnergy()
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
    pauser = Pauser(log) if hasattr(power, "pause_reason") else None
    for workload, path in blocks:
        run = (lambda w: (lambda: w.generate(k_gen))) if workload == "W1" else \
              (lambda w: (lambda: w.score(n_sc)))
        tag = {"workload": workload, "path": path}
        valid, i = 0, 0
        before = rest(meter)
        log(kind="rest", **tag, pair=i, **before)
        while valid < a.pairs and i < a.max_attempts:
            if pauser is None:
                reason = power.n10c_active()
                if reason:
                    log(kind="abort", reason=f"N10-C: {reason}")
                    return 3
            elif not pauser.wait():
                # paused: the logged rest is stale; attempt i gets no runs
                i += 1
                before = rest(meter)
                log(kind="rest", **tag, pair=i, **before)
                continue
            order = ["dense", path] if valid % 2 == 0 else [path, "dense"]
            runs = []
            for arm in order:
                r = window(meter, run(wl[arm]))
                log(kind="run", **tag, pair=i, arm=arm, **r)
                runs.append(r)
            after = rest(meter)
            log(kind="rest", **tag, pair=i + 1, **after)
            if pair_valid(before, runs, after):
                valid += 1
            else:
                log(kind="reject", **tag, pair=i, valid_so_far=valid)
            before, i = after, i + 1
        log(kind="block_done", **tag, valid=valid, attempts=i)
    log(kind="done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
