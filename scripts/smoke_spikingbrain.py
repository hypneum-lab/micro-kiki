#!/usr/bin/env python3
"""Smoke inference for SpikingBrain-7B (story 14 of v0.3 plan).

Downloads the SpikingBrain-7B checkpoint from ModelScope (if not already
present), loads it in BF16 via transformers, generates a short response
to a fixed prompt, and writes a JSON summary with peak memory + tok/s.

Primary variant (BF16, ~15 GB):  Panyuqi/V1-7B-base
Apache-2.0 fallback (W8ASpike):  Abel2076/SpikingBrain-7B-W8ASpike

Usage:
  uv run python scripts/smoke_spikingbrain.py
  uv run python scripts/smoke_spikingbrain.py --variant w8aspike
  uv run python scripts/smoke_spikingbrain.py --no-download  # local only
  uv run python scripts/smoke_spikingbrain.py --dry-run      # no model I/O

Output:
  results/spikingbrain-smoke.json

Exits 0 on success, 1 on smoke failure (empty / degenerate output),
2 on setup failure (missing deps, no weights).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_MODEL_DIR = Path(
    os.environ.get(
        "SPIKINGBRAIN_MODEL_DIR",
        "/Users/clems/models/spikingbrain-7b",
    )
)

VARIANTS = {
    "base": "Panyuqi/V1-7B-base",
    "sft": "Panyuqi/V1-7B-sft-s3-reasoning",
    "w8aspike": "Abel2076/SpikingBrain-7B-W8ASpike",
}

PROMPT = "hello, what are you?"
MAX_NEW_TOKENS = 32


def peak_rss_gb() -> float:
    """Return peak RSS in GiB via getrusage (process-wide)."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports ru_maxrss in bytes; Linux in KiB.
    if sys.platform == "darwin":
        return ru.ru_maxrss / (1024 ** 3)
    return ru.ru_maxrss / (1024 ** 2)


def download_via_modelscope(repo_id: str, target: Path) -> Path:
    """Download ``repo_id`` into ``target`` using the ModelScope SDK."""
    from modelscope import snapshot_download  # type: ignore

    target.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(model_id=repo_id, local_dir=str(target))
    return Path(local)


def run_smoke(model_dir: Path) -> dict:
    """Load + generate. Returns a dict for the results JSON."""
    import torch  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16

    t_load_start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    t_load_end = time.monotonic()
    load_mem_gb = peak_rss_gb()

    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    t_gen_start = time.monotonic()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    t_gen_end = time.monotonic()

    generated = out[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    tok_s = len(generated) / max(t_gen_end - t_gen_start, 1e-6)
    peak_mem_gb = peak_rss_gb()

    return {
        "prompt": PROMPT,
        "response": response,
        "response_len_chars": len(response),
        "load_time_s": round(t_load_end - t_load_start, 2),
        "gen_time_s": round(t_gen_end - t_gen_start, 3),
        "tokens_generated": int(generated.numel()),
        "tokens_s": round(tok_s, 2),
        "load_mem_gb": round(load_mem_gb, 2),
        "peak_mem_gb": round(peak_mem_gb, 2),
        "device": device,
        "dtype": "bfloat16",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        default="base",
        choices=sorted(VARIANTS.keys()),
        help="Which ModelScope repo to pull (default: base).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Local filesystem path to load / store the weights.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip ModelScope download (expect weights already present).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config + exit without touching the model.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "spikingbrain-smoke.json",
        help="Where to write the JSON summary.",
    )
    args = parser.parse_args()

    record: dict[str, object] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variant": args.variant,
        "repo_id": VARIANTS[args.variant],
        "model_dir": str(args.model_dir),
        "status": "pending",
    }

    if args.dry_run:
        record["status"] = "dry_run"
        record["notes"] = "dry-run requested; no download, no load"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        return 0

    try:
        if not args.no_download:
            print(
                f"[smoke] downloading {VARIANTS[args.variant]} "
                f"to {args.model_dir}"
            )
            download_via_modelscope(
                VARIANTS[args.variant], args.model_dir
            )
        if not args.model_dir.exists():
            record["status"] = "no_weights"
            record["error"] = f"weights missing at {args.model_dir}"
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(record, indent=2))
            return 2

        smoke = run_smoke(args.model_dir)
        record.update(smoke)
        record["status"] = (
            "ok"
            if smoke["response_len_chars"] >= 20
            and not smoke["response"].strip().startswith(PROMPT)
            else "degenerate"
        )
    except Exception as exc:  # pragma: no cover - runtime path
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0 if record["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
