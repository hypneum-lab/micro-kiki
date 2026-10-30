"""NH-81 analysis: checkpoint matching, pair statistics, decision rule.

Implements preregistration sections 3, 5 (rejection rule), 6 and 7.

Usage:
    python analyze.py select --dense-log runs/ann1/log.jsonl --spiking-log runs/spkt/log.jsonl
    python analyze.py stats --measure measure.jsonl [--pm pm.plist] --out summary.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

ISO_MARGIN = 0.02      # nats
GPU_BG_MAX = 10.0      # % mean background GPU utilisation in a rest window (macM1)
POWER_REST_MAX_W = 90.0  # W mean board power in a rest window (KXKM, prereg r3)
MIN_PAIRS = 10


# ---------------------------------------------------------------------------
# Section 3: iso-quality
# ---------------------------------------------------------------------------

def read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def select_dense(dense_log: list[dict], spiking_log: list[dict]) -> dict:
    s = max(spiking_log, key=lambda r: r["step"])
    d_star = min(dense_log, key=lambda r: (abs(r["val_nats"] - s["val_nats"]), r["step"]))
    d_final = max(dense_log, key=lambda r: r["step"])
    gap = s["test_nats"] - d_star["test_nats"]
    return {
        "spiking_step": s["step"], "spiking_val": s["val_nats"], "spiking_test": s["test_nats"],
        "spiking_test_fire": s.get("test_fire"),
        "dense_star_step": d_star["step"], "dense_star_val": d_star["val_nats"],
        "dense_star_test": d_star["test_nats"],
        "dense_final_step": d_final["step"], "dense_final_test": d_final["test_nats"],
        "gap_test_nats": gap, "gap_final_test_nats": s["test_nats"] - d_final["test_nats"],
        "iso_quality": abs(gap) <= ISO_MARGIN,
        "iso_direction": "iso" if abs(gap) <= ISO_MARGIN else
                         ("spiking_worse" if gap > 0 else "spiking_better"),
    }


# ---------------------------------------------------------------------------
# Sections 5-6: pairs and statistics
# ---------------------------------------------------------------------------

def window_ok(r: dict) -> bool:
    """No foreign GPU compute process seen in the window (KXKM replication)."""
    return not r.get("foreign_gpu_pids")


def rest_ok(r: dict) -> bool:
    if r.get("rest_rule") == "m3pro":  # macM3 (prereg 2026-10-30-nh81-m3pro)
        w = r["gpu_j"] / r["dur_s"] if r.get("dur_s", 0) > 0 else -1.0
        return (0 <= w <= r["rest_max_w"] and not r["busy_python"]
                and 0 <= r["load1"] <= r["load_max"])
    if "foreign_gpu_pids" in r:  # KXKM/NVML record: power guard (prereg r3)
        w = r["gpu_j"] / r["dur_s"] if r.get("dur_s", 0) > 0 else -1.0
        return 0 <= w <= POWER_REST_MAX_W and not r["busy_python"] and window_ok(r)
    # A failed utilisation read (-1) rejects the window (fail closed).
    return (0 <= r["gpu_util_mean"] <= GPU_BG_MAX and r.get("gpu_util_max", 0) >= 0
            and not r["busy_python"] and window_ok(r))


def build_pairs(records: list[dict], workload: str, path: str, pm=None) -> list[dict]:
    """Valid pairs with net J/token per arm (rest-power baseline subtracted)."""
    rests = {r["pair"]: r for r in records
             if r.get("kind") == "rest" and r["workload"] == workload and r["path"] == path}
    runs = [r for r in records
            if r.get("kind") == "run" and r["workload"] == workload and r["path"] == path]
    out = []
    for i in sorted({r["pair"] for r in runs}):
        before, after = rests.get(i), rests.get(i + 1)
        arms = {r["arm"]: r for r in runs if r["pair"] == i}
        rec = {"pair": i, "valid": bool(before and after and rest_ok(before) and rest_ok(after)
                                         and len(arms) == 2
                                         and all(window_ok(r) for r in arms.values()))}
        if before:
            p0 = energy_of(before, pm)["gpu_j"] / before["dur_s"]
            rec["rest_gpu_w"] = p0
            for arm, r in arms.items():
                e = energy_of(r, pm)
                net = e["gpu_j"] + e["ane_j"] - p0 * r["dur_s"]
                key = "dense" if arm == "dense" else "spiking"
                rec[f"{key}_j_per_tok"] = net / r["tokens"]
                rec[f"{key}_gross_j_per_tok"] = (e["gpu_j"] + e["ane_j"]) / r["tokens"]
                rec[f"{key}_gross_j"], rec[f"{key}_dur_s"] = e["gpu_j"] + e["ane_j"], r["dur_s"]
                rec[f"{key}_tokens"] = r["tokens"]
                rec[f"{key}_tok_s"] = r["tokens"] / r["dur_s"]
                rec[f"{key}_cpu_w"] = e["cpu_j"] / r["dur_s"]
                rec[f"{key}_vm_cpu_pct"] = r["vm_cpu_pct"]
        if rec["valid"]:
            rec["ratio"] = rec["dense_j_per_tok"] / rec["spiking_j_per_tok"]
            rec["ratio_gross"] = rec["dense_gross_j_per_tok"] / rec["spiking_gross_j_per_tok"]
        out.append(rec)
    return out


def energy_of(rec: dict, pm) -> dict:
    if pm is None:
        return {"gpu_j": rec["gpu_j"], "ane_j": rec["ane_j"], "cpu_j": rec["cpu_j"]}
    from power import window_energy

    return window_energy(pm, rec["t0"], rec["t1"])


def bootstrap(values: np.ndarray, stat, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    boots = np.array([stat(values[row]) for row in idx])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def geomean(x: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(x))))


def fixed_reference(valid: list[dict]) -> dict:
    """Sensitivity (descriptive): one rest reference for all pairs."""
    w0 = float(np.median([p["rest_gpu_w"] for p in valid]))

    def jpt(p, k):
        return (p[f"{k}_gross_j"] - w0 * p[f"{k}_dur_s"]) / p[f"{k}_tokens"]

    r = np.array([jpt(p, "dense") / jpt(p, "spiking") for p in valid])
    return {"rest_w_median": w0, "R_fixed_ref_hat": geomean(r) if np.all(r > 0) else float("nan")}


def summarize(pairs: list[dict]) -> dict:
    valid = [p for p in pairs if p["valid"]]
    res = {"pairs_total": len(pairs), "pairs_valid": len(valid)}
    if not valid:
        return res
    r = np.array([p["ratio"] for p in valid])
    diff = np.array([p["spiking_j_per_tok"] - p["dense_j_per_tok"] for p in valid])
    res.update({
        "R_hat": geomean(r), "R_ci95": bootstrap(r, geomean),
        "R_gross_hat": geomean(np.array([p["ratio_gross"] for p in valid])),
        **fixed_reference(valid),
        "diff_j_per_tok_mean": float(diff.mean()), "diff_ci95": bootstrap(diff, np.mean),
        "dense_j_per_tok_mean": float(np.mean([p["dense_j_per_tok"] for p in valid])),
        "spiking_j_per_tok_mean": float(np.mean([p["spiking_j_per_tok"] for p in valid])),
        "dense_tok_s_mean": float(np.mean([p["dense_tok_s"] for p in valid])),
        "spiking_tok_s_mean": float(np.mean([p["spiking_tok_s"] for p in valid])),
        "rest_gpu_w_mean": float(np.mean([p["rest_gpu_w"] for p in valid])),
        "vm_cpu_pct_mean": float(np.mean([p["dense_vm_cpu_pct"] for p in valid])),
    })
    return res


# ---------------------------------------------------------------------------
# Section 7: decision
# ---------------------------------------------------------------------------

def decide(stats: dict, iso_direction: str) -> str:
    if stats.get("pairs_valid", 0) < MIN_PAIRS:
        return "inconclusive: fewer than 10 valid pairs"
    r, (lo, _hi) = stats["R_hat"], stats["R_ci95"]
    if r <= 1.1:
        if iso_direction == "spiking_better":
            return "inconclusive: withdraw not allowed (spiking better than D* by > margin)"
        return "withdraw"
    if r >= 2 and lo > 1.1:
        if iso_direction == "spiking_worse":
            return "inconclusive: reopen not allowed (spiking worse than D* by > margin)"
        return "reopen"
    return "inconclusive"


def catalogue_reading(stats: dict) -> str:
    if stats.get("pairs_valid", 0) < MIN_PAIRS:
        return "not evaluable"
    if stats["R_hat"] <= 1.1:
        return "NH-81 supported"
    if stats["R_hat"] >= 1.25 and stats["R_ci95"][0] > 1.25:
        return "NH-81 refuted"
    return "neither"


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("select")
    s1.add_argument("--dense-log", required=True)
    s1.add_argument("--spiking-log", required=True)
    s2 = sub.add_parser("stats")
    s2.add_argument("--measure", required=True)
    s2.add_argument("--selection", required=True)
    s2.add_argument("--pm")
    s2.add_argument("--out", required=True)
    s2.add_argument("--instrument", help="label for the energy source (default: IOReport/powermetrics)")
    a = ap.parse_args()
    if a.cmd == "select":
        print(json.dumps(select_dense(read_jsonl(a.dense_log), read_jsonl(a.spiking_log)), indent=1))
        return 0
    sel = json.load(open(a.selection))
    pm = None
    if a.pm and os.path.exists(a.pm):
        from power import parse_powermetrics_plist

        pm = parse_powermetrics_plist(open(a.pm, "rb").read())
    recs = read_jsonl(a.measure)
    label = a.instrument or ("powermetrics" if pm else "IOReport (provisional)")
    out = {"selection": sel, "instrument": label,
           "blocks": {}}
    for workload in ("W1", "W2"):
        for path in ("fused", "event"):
            pairs = build_pairs(recs, workload, path, pm)
            st = summarize(pairs)
            st["decision"] = decide(st, sel["iso_direction"])
            st["catalogue"] = catalogue_reading(st)
            out["blocks"][f"{workload}_{path}"] = {"stats": st, "pairs": pairs}
    out["primary"] = out["blocks"]["W1_fused"]["stats"]
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: v["stats"] for k, v in out["blocks"].items()}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
