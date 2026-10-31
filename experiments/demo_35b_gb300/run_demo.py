"""GB300 demo: Qwen3.6-35B-A3B BF16 + micro-kiki V4 LoRA, J/token.

DEMO, NOT A SCIENTIFIC CLAIM. Prereg
``docs/prereg/demo-35b-lora-gb300/2026-10-31-demo-35b-lora-gb300.md``
(PI decisions 2026-10-31): sequential layout on one instance; all
``base`` runs first on the unwrapped model, then LoRA layers are added
and each domain's adapter is checked, loaded and run. Energy with the
NH-81 GB300 instrument (``experiments/nh81_gb300/power_gb300.py``).

Usage: python run_demo.py --base DIR --adapters DIR --prompts prompts.json --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NH81 = os.path.join(os.path.dirname(HERE), "nh81_gb300")
sys.path[:0] = [HERE, NH81, os.environ["NH81_CODE_DIR"]]

import demo_rules as D  # noqa: E402
import gb300_rules as R  # noqa: E402
import rest_probe_gb300 as RP  # noqa: E402


class Out:
    def __init__(self, d: str) -> None:
        os.makedirs(d, exist_ok=True)
        self.runs = open(os.path.join(d, "runs.jsonl"), "a")
        self.texts = open(os.path.join(d, "outputs.jsonl"), "a")

    def run(self, **r) -> None:
        r["logged"] = time.time()
        self.runs.write(json.dumps(r) + "\n")
        self.runs.flush()
        print(json.dumps({k: v for k, v in r.items() if k != "other_gpus"}), flush=True)

    def text(self, **r) -> None:
        self.texts.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.texts.flush()


def probe_before_model(p, outdir: str) -> dict:
    """Exclusivity, counter qualification and P0 (no CUDA context yet)."""
    if not RP.exclusivity(p):
        raise SystemExit("exclusivity check failed: not run")
    q = RP.qualify_counter(p)
    s = RP.Sampler(p, q["instrument"])
    s.start()
    t0 = time.time()
    time.sleep(120)
    t, e = s.between(t0, time.time())
    return {**q, "P0_mean_w": (e[-1] - e[0]) / (t[-1] - t[0]), "_sampler": s}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prompts = D.load_prompts(a.prompts)
    out = Out(a.out)
    thr_path = os.path.join(a.out, "rest_threshold.json")
    os.environ["NH81_REST_THRESHOLD"] = ""
    import power_gb300 as p

    pre = probe_before_model(p, a.out)
    sampler = pre.pop("_sampler")
    ident = p.check_identity()

    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import linear_to_lora_layers

    if mx.default_device() != mx.gpu:
        raise SystemExit("MLX default device is not the GPU")
    mx.set_memory_limit(D.MEMORY_LIMIT_GB * 2**30)
    t_load = time.time()
    model, tok = load(a.base)
    load_s = time.time() - t_load

    def ids_of(text: str) -> list[int]:
        return list(tok.apply_chat_template([{"role": "user", "content": text}],
                                            add_generation_prompt=True, enable_thinking=False,
                                            tokenize=True))

    def gen(ids: list[int], n: int) -> dict:
        last, text = None, []
        for r in stream_generate(model, tok, mx.array(ids), max_tokens=n):
            text.append(r.text)
            last = r
        return {"text": "".join(text), "prompt_tokens": last.prompt_tokens,
                "generated_tokens": last.generation_tokens, "finish": last.finish_reason,
                "peak_memory_gb": last.peak_memory}

    def run_domain(d: str, k: int) -> tuple[int, list[dict]]:
        n, res = 0, []
        for _ in range(k):
            for pr in prompts[d]:
                ids = ids_of(pr["text"])
                g = gen(ids, D.MAX_NEW_TOKENS)
                n += g["generated_tokens"]
                res.append({"prompt_id": pr["id"], "rendered_ids_sha256":
                            D.sha256_bytes(json.dumps(ids).encode()), **g})
        return n, res

    # P1 in this process: stabilisation on base runs, then 60 s idle,
    # watched for foreign GPU/CPU activity; voided once, then abandoned.
    def p1_once() -> tuple[float, float] | None:
        bad: list[str] = []
        stop = threading.Event()

        def watch():
            while not stop.is_set():
                r = RP.clean_reason(p, set()) or RP.cpu_window(p, set(), 5.0)
                if r:
                    bad.append(r)
                    stop.wait(5.0)
        th = threading.Thread(target=watch, daemon=True)
        t0 = time.time()
        th.start()
        i = 0
        while time.time() - t0 < R.STAB_S:
            run_domain(D.DOMAINS[i % len(D.DOMAINS)], 1)
            i += 1
        rest_t0 = time.time()
        time.sleep(RP.P1_REST_S)
        stop.set()
        th.join()
        if bad or not sampler.clean(t0, time.time()):
            print(json.dumps({"kind": "p1_void", "reason": bad[:3] or "instrument fault"}), flush=True)
            return None
        return rest_t0, time.time()

    w = p1_once() or (RP.exclusivity(p) and p1_once())
    if not w:
        raise SystemExit("P1 voided twice: not run")
    rest_t0, rest_t1 = w
    tl, el = sampler.between(rest_t0 - 60, rest_t0)
    tr, er = sampler.between(rest_t0, rest_t1)
    p_load = (el[-1] - el[0]) / (tl[-1] - tl[0])
    p_rest = R.max_sliding_mean(tr, er, 20.0)
    T = R.threshold(p_rest)
    thr = {"gpu": ident, **pre, "P_load_w": p_load, "P_rest_w": p_rest, "k": R.K_REST,
           "T_w": T, "discriminative": p_load > T, "stab_s": R.STAB_S, "load_s": load_s}
    json.dump(thr, open(thr_path, "w"), indent=1)
    os.environ["NH81_REST_THRESHOLD"] = thr_path
    p._CFG = None
    sampler.stop.set()
    meter = p.EnergyMeter()

    def window(fn) -> dict:
        pids: set[int] = set()
        stop = threading.Event()

        def watch():
            while not stop.is_set():
                pids.update(p.foreign_gpu_pids())
                stop.wait(0.5)
        th = threading.Thread(target=watch, daemon=True)
        tk0 = p.python_ticks()
        th.start()
        s0 = meter.snapshot()
        payload = fn()
        mx.synchronize()
        s1 = meter.snapshot()
        stop.set()
        th.join()
        busy = p.busy_between(tk0, p.python_ticks(), s1[0] - s0[0])
        rec = {"t0": s0[0], "t1": s1[0], "dur_s": s1[0] - s0[0],
               "gpu_j": s1[1]["GPU Energy"] - s0[1]["GPU Energy"],
               "busy_python": [b[2] for b in busy], "foreign_gpu_pids": sorted(pids),
               **p.window_extra()}
        return rec, payload

    def rest() -> dict:
        r, _ = window(lambda: time.sleep(D.REST_S))
        return r

    paused = [0.0]

    def wait_clean() -> None:
        t = time.time()
        while p.pause_reason():
            if paused[0] + time.time() - t > R.PAUSE_BUDGET_S:
                raise SystemExit("pause budget exhausted")
            time.sleep(60)
        paused[0] += time.time() - t

    def block(d: str, arm: str, k: int) -> None:
        gen(ids_of(prompts[d][0]["text"]), 16)  # unmeasured warm-up
        valid, att = 0, 0
        wait_clean()
        before = rest()
        while valid < D.VALID_RUNS and att < D.MAX_ATTEMPTS:
            r, (n, res) = window(lambda: run_domain(d, k))
            after = rest()
            r.update(domain=d, arm=arm, attempt=att, k_repeats=k, generated_tokens=n,
                     prompt_tokens=sum(x["prompt_tokens"] for x in res),
                     stop_reasons=[x["finish"] for x in res], layout="sequential",
                     rest_before=before, rest_after=after,
                     rest_power_w=before["gpu_j"] / before["dur_s"])
            r["j_per_token"] = D.jpt(r, before)
            r["j_per_token_gross"] = r["gpu_j"] / n if n else None
            r["valid"] = (R.rest_ok_gb300(before) and R.rest_ok_gb300(after)
                          and R.window_ok_gb300(r))
            r["short"] = r["dur_s"] < D.MIN_RUN_S
            out.run(kind="run", **r)
            for x in res:
                out.text(domain=d, arm=arm, attempt=att, prompt_id=x["prompt_id"],
                         rendered_ids_sha256=x["rendered_ids_sha256"],
                         prompt_tokens=x["prompt_tokens"], generated_tokens=x["generated_tokens"],
                         finish=x["finish"], text=x["text"],
                         text_sha256=D.sha256_bytes(x["text"].encode()))
            valid += int(r["valid"])
            att += 1
            before = after

    # Phase 1: base runs, K calibrated per domain on the base arm.
    ks = {}
    for d in D.DOMAINS:
        gen(ids_of(prompts[d][0]["text"]), 16)
        t = time.time()
        run_domain(d, 1)
        ks[d] = D.calibrate_k(time.time() - t)
        block(d, "base", ks[d])

    # Phase 2: LoRA layers once, then each adapter checked and loaded.
    linear_to_lora_layers(model, 32, D.LORA)
    want = {k: (tuple(v.shape), str(v.dtype).split(".")[-1]) for k, v in tree_flatten(model.parameters())
            if k.endswith(".lora_a") or k.endswith(".lora_b")}

    def load_adapter(d: str) -> None:
        w = mx.load(os.path.join(a.adapters, d, "adapters.safetensors"))
        got = {k: (tuple(v.shape), str(v.dtype).split(".")[-1]) for k, v in w.items()}
        probs = D.check_adapter(got, want)
        if probs or len(got) != D.EXPECTED_TENSORS_32_LAYERS:
            raise SystemExit(f"adapter {d}: {len(got)} tensors, {probs[:5]}")
        model.load_weights(list(w.items()), strict=False)
        mx.eval(model.parameters())
        mx.clear_cache()

    def logits(d: str):
        ids = mx.array([ids_of(prompts[d][0]["text"])])
        lg = model(ids)
        mx.eval(lg)
        return lg

    # Switch test A -> B -> A (bitwise identical logits for both A loads).
    A, B = D.DOMAINS[0], D.DOMAINS[1]
    load_adapter(A)
    la = logits(A)
    load_adapter(B)
    logits(B)
    load_adapter(A)
    if not bool(mx.array_equal(la, logits(A)).item()):
        raise SystemExit("switch test failed: A-B-A logits differ")
    for d in D.DOMAINS:
        load_adapter(d)
        block(d, "adapter", ks[d])

    runs = [json.loads(x) for x in open(os.path.join(a.out, "runs.jsonl"))]
    json.dump(D.summarize(runs), open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
