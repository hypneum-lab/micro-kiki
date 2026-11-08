"""GB300 rest threshold (prereg section 4.2, PI decisions 2026-10-31).

Order: exclusivity check (2 min) -> counter qualification (60 s) ->
P0 (120 s, no CUDA context of ours) -> P1 (child process: models
loaded, warm-up, dense W1 for 10 min, then 60 s idle, exit) ->
T = 1.25 x P_rest, written to ``--out``. ``--final`` takes the 60-s
read-only probe after the last pair. Numbers here never enter the
analysis.

Usage:
  python rest_probe_gb300.py --out rest_threshold.json --dense D --spiking S --test T
  python rest_probe_gb300.py --final --out final_probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gb300_rules as R  # noqa: E402

CLEAN_S = 120.0
P1_REST_S = 60.0


def _pg():
    os.environ.setdefault("NH81_REST_THRESHOLD", "")
    import power_gb300 as p
    return p


class Sampler(threading.Thread):
    """(t, cumulative J) every 1 s from the counter or the integrator."""

    def __init__(self, p, mode: str) -> None:
        super().__init__(daemon=True)
        self.p, self.mode, self.rows, self.stop = p, mode, [], threading.Event()
        self.last_faults = p.FAULTS
        self.integ = p._Integrator() if mode == "power_integrated" else None
        self.faults: list[float] = []  # times of bad reads

    def read(self) -> float | None:
        return self.integ.value() if self.integ else self.p.counter_j()

    def run(self) -> None:
        while not self.stop.is_set():
            e, now = self.read(), time.time()
            faults = self.p.FAULTS
            if e is None or (self.rows and e < self.rows[-1][1]) or faults > self.last_faults:
                self.faults.append(now)
            self.last_faults = faults
            if e is not None:
                self.rows.append((now, e))
            self.stop.wait(1.0)

    def clean(self, t0: float, t1: float) -> bool:
        """No bad read and no gap > 2 s between t0 and t1."""
        t, _ = self.between(t0, t1)
        gaps = [b - a for a, b in zip(t, t[1:])]
        return (not any(t0 <= f <= t1 for f in self.faults) and len(t) >= 2
                and t[0] - t0 <= 2 and t1 - t[-1] <= 2 and all(g <= 2.0 for g in gaps))

    def between(self, t0: float, t1: float) -> tuple[list[float], list[float]]:
        sel = [(t, e) for t, e in self.rows if t0 <= t <= t1]
        return [t for t, _ in sel], [e for _, e in sel]


def clean_reason(p, exclude: set[int]) -> str:
    fp = {x for x in p.foreign_gpu_pids() if x not in exclude}
    if fp:
        return f"foreign GPU pids {sorted(fp)}"
    if p.n10c_active():
        return f"slot job: {p.n10c_active()}"
    if p.load1() > R.LOAD1_MAX:
        return f"load average {p.load1():.2f}"
    return ""


def cpu_window(p, exclude: set[int], dur: float) -> str:
    a, sa, t0 = p.python_ticks(), p._stat_busy(), time.time()
    time.sleep(dur)
    b, sb, dt = p.python_ticks(), p._stat_busy(), time.time() - t0
    busy = R.cpu_busy(a, b, dt, p.HZ, exclude | {os.getpid()})
    ours = sum(b[x][1] - (a[x][1] if x in a else 0) for x in exclude | {os.getpid()} if x in b)
    cores = R.host_foreign_cores(sa, sb, 0, ours, p.HZ, dt)
    if busy:
        return f"busy processes {[r[2] for r in busy]}"
    if cores > R.CPU_HOST_MAX_CORES:
        return f"host CPU {cores:.2f} cores"
    return ""


def exclusivity(p, exclude: set[int] = frozenset(), budget: float = R.GUARD_BUDGET_S) -> bool:
    t_end = time.time() + budget
    while time.time() < t_end:
        r = clean_reason(p, set(exclude))
        if not r:
            r = cpu_window(p, set(exclude), CLEAN_S)
        if not r and not clean_reason(p, set(exclude)):
            return True
        print(json.dumps({"kind": "probe_wait", "reason": r}), flush=True)
        time.sleep(60)
    return False


def qualify_counter(p) -> dict:
    t0, e0 = time.time(), p.counter_j()
    reads, ws, ok, last = [], [], e0 is not None, e0
    while ok and time.time() - t0 < 60:
        time.sleep(0.5)
        e, w = p.counter_j(), p.board_power_w()
        if e is None or (last is not None and e < last) or w < 0:
            ok = False
            break
        reads.append(e)
        ws.append(w)
        last = e
    res = {"counter_ok": ok}
    if ok and reads:
        cw = (reads[-1] - e0) / (time.time() - t0)
        sw = sum(ws) / len(ws)
        res.update(counter_w=cw, sampled_w=sw, consistent=R.counter_consistent(cw, sw))
        res["counter_ok"] = res["consistent"]
    res["instrument"] = "counter" if res["counter_ok"] else "power_integrated"
    return res


def child(a) -> int:
    """P1 body: runs on the GPU, prints phase marks, never measures."""
    code = os.environ["NH81_CODE_DIR"]
    sys.path[:0] = [code]
    import power_gb300 as p
    p.check_identity()
    sys.modules["power_nvml"] = p
    os.environ["NH81_HOST"] = "kxkm"
    import mlx.core as mx
    import numpy as np
    import measure
    mx.set_memory_limit(6 * 2**30)
    prompts, new_tokens = measure.encode_prompts()
    test = np.fromfile(a.test, dtype=np.int32)
    w = measure.Workload(measure.load(a.dense, "ann1"), "fused", prompts, new_tokens, test)
    spk = measure.Workload(measure.load(a.spiking, "spkt"), "fused", prompts, new_tokens, test)
    for x in (w, spk):
        x.generate(1)
        x.score(2)
    print(json.dumps({"phase": "load_start", "t": time.time()}), flush=True)
    t0 = time.time()
    while time.time() - t0 < R.STAB_S:
        w.generate(1)
    print(json.dumps({"phase": "rest_start", "t": time.time()}), flush=True)
    time.sleep(P1_REST_S)
    print(json.dumps({"phase": "done", "t": time.time()}), flush=True)
    return 0


def p1(p, s: Sampler, a) -> dict | None:
    cmd = [sys.executable, os.path.abspath(__file__), "--child", "--dense", a.dense,
           "--spiking", a.spiking, "--test", a.test]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    marks, bad = {}, ""
    reader = threading.Thread(target=lambda: [marks.setdefault(json.loads(l)["phase"], json.loads(l)["t"])
                                              for l in proc.stdout if l.startswith('{"phase"')], daemon=True)
    reader.start()
    while proc.poll() is None:
        r = clean_reason(p, {proc.pid}) or cpu_window(p, {proc.pid}, 5.0)
        if r:
            bad = bad or r
            time.sleep(5)
    reader.join(5)
    if proc.returncode != 0 or bad or "done" not in marks:
        print(json.dumps({"kind": "p1_void", "reason": bad or f"child rc {proc.returncode}"}), flush=True)
        return None
    if not s.clean(marks["load_start"], marks["done"]):
        print(json.dumps({"kind": "p1_void", "reason": "instrument fault during P1"}), flush=True)
        return None
    tl, el = s.between(marks["rest_start"] - 60, marks["rest_start"])
    tr, er = s.between(marks["rest_start"], marks["done"])
    return {"P_load_w": (el[-1] - el[0]) / (tl[-1] - tl[0]),
            "P_rest_w": R.max_sliding_mean(tr, er, 20.0), "marks": marks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--dense", default="")
    ap.add_argument("--spiking", default="")
    ap.add_argument("--test", default="")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    if a.child:
        return child(a)
    p = _pg()
    gpu = {"pci_bus_id": os.environ["NH81_GPU_PCI_BUS_ID"], "uuid": p.uuid(),
           "name": p.device_name(), "mig": p.mig_mode()}
    if gpu["mig"] == "enabled" or gpu["mig"].startswith("error"):
        json.dump({"outcome": "not measured", "reason": f"MIG {gpu['mig']}"}, open(a.out, "w"))
        return 2
    if a.final:
        mode = json.load(open(os.environ["NH81_REST_THRESHOLD_IN"]))["instrument"]
        s = Sampler(p, mode)
        s.start()
        time.sleep(60)
        t, e = s.between(0, time.time())
        json.dump({"gpu": gpu, "instrument": mode, "mean_power_w": (e[-1] - e[0]) / (t[-1] - t[0])},
                  open(a.out, "w"), indent=1)
        return 0
    if not exclusivity(p):
        json.dump({"outcome": "not measured", "reason": "exclusivity check failed"}, open(a.out, "w"))
        return 2
    q = qualify_counter(p)
    s = Sampler(p, q["instrument"])
    s.start()
    t0 = time.time()
    time.sleep(120)
    t, e = s.between(t0, time.time())
    if not s.clean(t0, time.time()):
        json.dump({"outcome": "not measured", "reason": "instrument fault during P0"}, open(a.out, "w"))
        return 2
    ws = [(e[i + 1] - e[i]) / (t[i + 1] - t[i]) for i in range(len(t) - 1)]
    res = None
    for attempt in range(2):  # one retry after a clean 2 min (prereg 4.2)
        res = p1(p, s, a)
        if res is not None or attempt == 1 or not exclusivity(p):
            break
    if res is None:
        json.dump({"outcome": "not measured", "reason": "P1 voided twice"}, open(a.out, "w"))
        return 2
    T = R.threshold(res["P_rest_w"])
    out = {"gpu": gpu, **q, "P0_mean_w": (e[-1] - e[0]) / (t[-1] - t[0]), "P0_max_w": max(ws),
           **res, "k": R.K_REST, "T_w": T, "discriminative": res["P_load_w"] > T,
           "stab_s": R.STAB_S, "written": time.time()}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
