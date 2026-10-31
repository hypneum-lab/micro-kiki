"""Pure rules of the GB300 35B LoRA demo (no GPU, no MLX).

Demo, not a scientific claim. Prereg
``docs/prereg/demo-35b-lora-gb300/2026-10-31-demo-35b-lora-gb300.md``
(PI decisions 2026-10-31): sequential layout, 8 domains, 3 valid runs
per (domain, arm), 256 new tokens, runs >= 15 s.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics

DOMAINS = ["chat-fr", "python", "cpp", "rust", "kicad-dsl", "spice", "embedded", "power"]
PROMPTS_SHA256 = "45392249a696c1be953d2362a519bae121d97fc0c8d801c009be12a3c400ec81"
MAX_NEW_TOKENS = 256
MIN_RUN_S = 15.0
REST_S = 20.0
VALID_RUNS = 3
MAX_ATTEMPTS = 6
MEMORY_LIMIT_GB = 230
EXPECTED_TENSORS_32_LAYERS = 816
LORA = {"rank": 16, "scale": 20.0, "dropout": 0.0}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_prompts(path: str) -> dict[str, list[dict]]:
    raw = open(path, "rb").read()
    if sha256_bytes(raw) != PROMPTS_SHA256:
        raise SystemExit("prompts.json SHA-256 mismatch")
    d = json.loads(raw)
    by = {k: [] for k in DOMAINS}
    for p in d["prompts"]:
        if sha256_bytes(p["text"].encode()) != p["text_sha256"]:
            raise SystemExit(f"prompt {p['id']} SHA-256 mismatch")
        by[p["domain"]].append(p)
    if d["domains"] != DOMAINS or any(len(v) != 3 for v in by.values()):
        raise SystemExit("prompts.json does not match the frozen domain list")
    return by


def calibrate_k(seconds_one_pass: float, target: float = MIN_RUN_S) -> int:
    if seconds_one_pass <= 0:
        raise ValueError("non-positive pass time")
    return max(1, math.ceil(target / seconds_one_pass))


def check_adapter(got: dict[str, tuple[tuple[int, ...], str]],
                  want: dict[str, tuple[tuple[int, ...], str]]) -> list[str]:
    """Problems between adapter tensors and the wrapped model's LoRA
    parameters, both as {name: (shape, dtype)}; empty list = exact match."""
    probs = [f"missing {k}" for k in sorted(set(want) - set(got))]
    probs += [f"extra {k}" for k in sorted(set(got) - set(want))]
    for k in sorted(set(got) & set(want)):
        if got[k][0] != want[k][0]:
            probs.append(f"shape {k}: {got[k][0]} != {want[k][0]}")
        if got[k][1] != "float32":
            probs.append(f"dtype {k}: {got[k][1]}")
    return probs


def jpt(run: dict, rest: dict) -> float | None:
    """Net J per generated token; None if undefined (no tokens)."""
    if run["generated_tokens"] <= 0:
        return None
    p0 = rest["gpu_j"] / rest["dur_s"]
    return (run["gpu_j"] - p0 * run["dur_s"]) / run["generated_tokens"]


def summarize(runs: list[dict]) -> dict:
    """Per (domain, arm): valid runs, median and range of net J/token
    (non-positive or undefined values excluded and counted), gross J/token,
    tokens/s, short runs; ratio of medians adapter/base, descriptive."""
    out: dict = {}
    for r in runs:
        if not r.get("valid"):
            continue
        key = f"{r['domain']}|{r['arm']}"
        o = out.setdefault(key, {"domain": r["domain"], "arm": r["arm"], "valid_runs": 0,
                                 "short_runs": 0, "excluded_nonpositive": 0,
                                 "_net": [], "_gross": [], "_tps": []})
        o["valid_runs"] += 1
        o["short_runs"] += int(r["dur_s"] < MIN_RUN_S)
        if r["j_per_token"] is None or r["j_per_token"] <= 0:
            o["excluded_nonpositive"] += 1
        else:
            o["_net"].append(r["j_per_token"])
        if r["generated_tokens"] > 0:
            o["_gross"].append(r["gpu_j"] / r["generated_tokens"])
        o["_tps"].append(r["generated_tokens"] / r["dur_s"])
    for o in out.values():
        for k, name in (("_net", "j_per_token"), ("_gross", "j_per_token_gross"), ("_tps", "tokens_per_s")):
            v = o.pop(k)
            o[f"{name}_median"] = statistics.median(v) if v else None
            o[f"{name}_range"] = [min(v), max(v)] if v else None
    ratios = {}
    for d in DOMAINS:
        b, a = out.get(f"{d}|base"), out.get(f"{d}|adapter")
        if b and a and b["j_per_token_median"] and a["j_per_token_median"]:
            ratios[d] = a["j_per_token_median"] / b["j_per_token_median"]
    return {"status": "demo, not a scientific claim", "layout": "sequential",
            "cells": out, "ratio_of_medians_adapter_over_base": ratios}
