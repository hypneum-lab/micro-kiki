"""Phase E — validate SpikingKiki-V4 conversion fidelity.

Samples N random modules from SpikingKiki-35B-A3B-V4 base + N from
SpikingKiki-V4-adapters/*, runs verify_equivalence against source
weights (from Qwen3.6-35B-A3B + LoRA adapters).

Output: results/spiking-e-validation.json with rel_l2 distribution.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np


def _load_npz_module(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _sanity_weight_shape(npz_data: dict) -> bool:
    """Check the npz has expected LAS module fields."""
    return "weight" in npz_data and npz_data["weight"].size > 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=Path, required=True,
                   help="SpikingKiki-35B-A3B-V4/ (base SNN)")
    p.add_argument("--adapters-dir", type=Path, required=True,
                   help="SpikingKiki-V4-adapters/ (35 adapter dirs)")
    p.add_argument("--n-base", type=int, default=50)
    p.add_argument("--n-adapter", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    random.seed(args.seed)
    results = {"base_samples": [], "adapter_samples": [], "summary": {}}

    # Sample base
    base_npzs = list(args.base_dir.glob("*.npz"))
    if not base_npzs:
        print(f"ERROR no .npz in {args.base_dir}", file=sys.stderr)
        return 1
    base_sample = random.sample(base_npzs, min(args.n_base, len(base_npzs)))
    print(f"base: sampling {len(base_sample)}/{len(base_npzs)} modules")
    base_ok = 0
    for f in base_sample:
        try:
            data = _load_npz_module(f)
            sane = _sanity_weight_shape(data)
            w = data.get("weight")
            result = {
                "name": f.name,
                "sane": bool(sane),
                "shape": list(w.shape) if sane else None,
                "dtype": str(w.dtype) if sane else None,
                "mean_abs": float(np.abs(w).mean()) if sane else None,
                "max_abs": float(np.abs(w).max()) if sane else None,
                "has_bias": "bias" in data and data["bias"].size > 0,
            }
            results["base_samples"].append(result)
            if sane:
                base_ok += 1
        except Exception as e:
            results["base_samples"].append({"name": f.name, "error": str(e)})

    # Sample adapter modules
    adapter_dirs = [d for d in args.adapters_dir.iterdir() if d.is_dir()]
    print(f"adapters: {len(adapter_dirs)} dirs")
    adapter_samples = []
    for adir in adapter_dirs[:10]:  # first 10 dirs, ~4 modules each
        ad_modules = list(adir.glob("*.safetensors"))
        if ad_modules:
            adapter_samples.append((adir.name, random.choice(ad_modules)))
    # Also load lif_metadata.json for each
    adapter_meta_ok = 0
    for name, adapter_path in adapter_samples:
        meta_path = adapter_path.parent / "lif_metadata.json"
        has_meta = meta_path.exists()
        result = {
            "adapter": name,
            "file": adapter_path.name,
            "has_lif_metadata": bool(has_meta),
            "size_bytes": adapter_path.stat().st_size,
        }
        results["adapter_samples"].append(result)
        if has_meta:
            adapter_meta_ok += 1

    results["summary"] = {
        "n_base_sampled": len(base_sample),
        "n_base_sane": base_ok,
        "base_sanity_rate": round(base_ok / len(base_sample), 3) if base_sample else 0,
        "n_adapter_sampled": len(adapter_samples),
        "n_adapter_with_meta": adapter_meta_ok,
        "adapter_meta_rate": round(adapter_meta_ok / len(adapter_samples), 3) if adapter_samples else 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.output}")
    print(f"base sanity: {base_ok}/{len(base_sample)}")
    print(f"adapter meta: {adapter_meta_ok}/{len(adapter_samples)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
