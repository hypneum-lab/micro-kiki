#!/usr/bin/env python3
"""E2E forgetting-gate smoke on Mac Studio (real Qwen3.6-35B-A3B)."""
from __future__ import annotations
import argparse, json, logging, sys, time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("smoke_gate")

DEFAULT_PROMPTS = [
    {"prompt": "Explique en une phrase ce que fait un op-amp en configuration suiveur.", "reference": "gain unitaire buffer adaptation impédance"},
    {"prompt": "Write a Python one-liner that computes the median of a list.", "reference": "sorted median middle"},
    {"prompt": "Qu'est-ce qu'une LoRA en apprentissage profond ?", "reference": "LoRA rang bas adaptateur"},
    {"prompt": "Describe the softmax function briefly.", "reference": "softmax exponential normalization probability"},
    {"prompt": "À quoi sert un decoupling capacitor dans un circuit numérique ?", "reference": "découplage bruit alimentation filtre"},
]

def _containment(ref: str, out: str) -> float:
    if not ref: return 0.0
    toks = [t.lower() for t in ref.split() if t]
    if not toks: return 0.0
    hits = sum(1 for t in toks if t in out.lower())
    return hits / len(toks)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True, type=Path)
    p.add_argument("--prior-adapter", required=True, type=Path)
    p.add_argument("--new-adapter", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--max-tokens", type=int, default=128)
    args = p.parse_args()

    from mlx_lm import load, generate
    from mlx_lm.tuner.utils import load_adapters
    from mlx_lm.sample_utils import make_sampler
    from safetensors import safe_open
    import mlx.core as mx

    t0 = time.monotonic()
    logger.info("loading base model from %s", args.base_model)
    model, tokenizer = load(str(args.base_model))
    load_adapters(model, str(args.prior_adapter))
    logger.info("base + prior loaded in %.1fs", time.monotonic() - t0)

    prompts = DEFAULT_PROMPTS
    sampler = make_sampler(temp=0.0)
    prior_resps = []
    t1 = time.monotonic()
    for i, e in enumerate(prompts, 1):
        logger.info("[prior %d/%d] %.60s", i, len(prompts), e["prompt"])
        resp = generate(model, tokenizer, prompt=e["prompt"], max_tokens=args.max_tokens, sampler=sampler)
        prior_resps.append(resp)
    logger.info("prior pass: %.1fs", time.monotonic() - t1)

    logger.info("swapping weights -> %s", args.new_adapter)
    t2 = time.monotonic()
    wp = args.new_adapter / "adapters.safetensors"
    new_w = {}
    with safe_open(str(wp), framework="numpy") as f:
        for k in f.keys():
            new_w[k] = mx.array(f.get_tensor(k))
    model.load_weights(list(new_w.items()), strict=False)
    logger.info("swap: %.1fs", time.monotonic() - t2)

    new_resps = []
    t3 = time.monotonic()
    for i, e in enumerate(prompts, 1):
        logger.info("[new %d/%d] %.60s", i, len(prompts), e["prompt"])
        resp = generate(model, tokenizer, prompt=e["prompt"], max_tokens=args.max_tokens, sampler=sampler)
        new_resps.append(resp)
    logger.info("new pass: %.1fs", time.monotonic() - t3)

    # Scoring + angle
    per_prompt = []
    prior_scores = []
    new_scores = []
    for e, pr, nr in zip(prompts, prior_resps, new_resps):
        sp = _containment(e["reference"], pr)
        sn = _containment(e["reference"], nr)
        prior_scores.append(sp); new_scores.append(sn)
        per_prompt.append({"prompt": e["prompt"], "prior_response": pr[:300], "new_response": nr[:300], "prior_score": sp, "new_score": sn})

    # Angles via measure_forgetting_signal
    angles = {}
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.eval.forgetting import measure_forgetting_signal
        report = measure_forgetting_signal(prior_adapter_path=args.prior_adapter / "adapters.safetensors", new_adapter_path=args.new_adapter / "adapters.safetensors")
        angles = {"mean": report.get("angle_degrees_mean"), "per_module_count": len(report.get("angle_degrees_per_module", {}))}
    except Exception as ex:
        logger.warning("angle computation skipped: %s", ex)

    out = {
        "base_model": str(args.base_model),
        "prior_adapter": str(args.prior_adapter),
        "new_adapter": str(args.new_adapter),
        "n_prompts": len(prompts),
        "prior_mean_score": sum(prior_scores)/len(prior_scores),
        "new_mean_score": sum(new_scores)/len(new_scores),
        "winrate_drop": (sum(prior_scores)-sum(new_scores))/len(prior_scores),
        "angles": angles,
        "per_prompt": per_prompt,
    }
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("wrote %s", args.output)
    return 0

if __name__ == "__main__":
    sys.exit(main())
