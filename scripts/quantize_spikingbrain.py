#!/usr/bin/env python3
"""Q4 quantization for SpikingBrain-7B (story 15 of v0.3 plan).

Strategy:

1. Attempt llama.cpp GGUF conversion via ``convert_hf_to_gguf.py``,
   then ``llama-quantize`` with Q4_K_M.
2. If llama.cpp does not recognise the SpikingBrain / Qwen2-Spiking
   architecture (expected — the hybrid-linear attention op + PLIF
   spike node are not in llama.cpp's op set as of 2026-04), record
   the specific error and fall back to ``bitsandbytes`` 4-bit in
   PyTorch (no GGUF).
3. Emit ``results/spikingbrain-quant.json`` with
   ``{quant_method, size_gb, tokens_s, error?}``.

Target: ~3.5 GB Q4_K_M, ≥ 10 tok/s on Studio. If llama.cpp fails,
record bitsandbytes size + tok/s and mark Q4_GGUF as TODO.

Usage:
  uv run python scripts/quantize_spikingbrain.py
  uv run python scripts/quantize_spikingbrain.py --dry-run
  uv run python scripts/quantize_spikingbrain.py \\
      --model-dir /Users/clems/models/spikingbrain-7b \\
      --llama-cpp /Users/clems/llama.cpp \\
      --out-gguf /Users/clems/models/spikingbrain-7b-q4.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
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
DEFAULT_LLAMA_CPP = Path(
    os.environ.get(
        "LLAMA_CPP_DIR",
        "/Users/clems/llama.cpp",
    )
)
DEFAULT_OUT_GGUF = Path(
    os.environ.get(
        "SPIKINGBRAIN_Q4_GGUF",
        "/Users/clems/models/spikingbrain-7b-q4.gguf",
    )
)


def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def try_llama_cpp(
    model_dir: Path,
    llama_cpp: Path,
    out_gguf: Path,
) -> dict:
    """Attempt the GGUF conversion + Q4_K_M quantize path."""
    record: dict[str, object] = {"quant_method": "llama_cpp_q4_k_m"}

    convert_py = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_py.exists():
        record["status"] = "blocked"
        record["error"] = f"convert script missing: {convert_py}"
        return record

    bf16_gguf = out_gguf.with_suffix(".bf16.gguf")
    out_gguf.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: HF -> BF16 GGUF
    cmd_convert = [
        sys.executable,
        str(convert_py),
        str(model_dir),
        "--outfile",
        str(bf16_gguf),
        "--outtype",
        "bf16",
    ]
    rc, stdout, stderr = _run(cmd_convert)
    if rc != 0:
        record["status"] = "convert_failed"
        record["stage"] = "convert_hf_to_gguf"
        record["stderr_tail"] = stderr.splitlines()[-20:]
        record["error"] = (
            "llama.cpp convert_hf_to_gguf.py rejected the SpikingBrain "
            "architecture (expected — hybrid-linear attention + PLIF "
            "spike layers are not in llama.cpp's op set). See stderr_tail."
        )
        return record

    # Step 2: BF16 GGUF -> Q4_K_M
    quantize_bin = llama_cpp / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = llama_cpp / "llama-quantize"
    if not quantize_bin.exists():
        record["status"] = "blocked"
        record["error"] = f"llama-quantize binary missing near {llama_cpp}"
        return record

    cmd_quant = [str(quantize_bin), str(bf16_gguf), str(out_gguf), "Q4_K_M"]
    rc, stdout, stderr = _run(cmd_quant)
    if rc != 0 or not out_gguf.exists():
        record["status"] = "quantize_failed"
        record["stage"] = "llama-quantize"
        record["stderr_tail"] = stderr.splitlines()[-20:]
        return record

    size_gb = out_gguf.stat().st_size / (1024 ** 3)
    record.update({
        "status": "ok",
        "size_gb": round(size_gb, 2),
        "gguf_path": str(out_gguf),
    })
    return record


def try_smoke_gguf(
    out_gguf: Path,
    llama_cpp: Path,
) -> dict:
    """Quick llama-cli smoke: generate 5 tokens + measure tok/s."""
    llama_cli = llama_cpp / "build" / "bin" / "llama-cli"
    if not llama_cli.exists():
        llama_cli = llama_cpp / "llama-cli"
    if not llama_cli.exists():
        return {"smoke_status": "blocked",
                "smoke_error": f"llama-cli missing near {llama_cpp}"}

    cmd = [
        str(llama_cli),
        "-m", str(out_gguf),
        "-p", "hello",
        "-n", "5",
        "--no-warmup",
        "--no-display-prompt",
    ]
    t0 = time.monotonic()
    rc, stdout, stderr = _run(cmd)
    elapsed = time.monotonic() - t0
    tok_s = 5 / max(elapsed, 1e-6)
    return {
        "smoke_status": "ok" if rc == 0 else "failed",
        "smoke_tokens_s": round(tok_s, 2),
        "smoke_output_tail": stdout.strip().splitlines()[-5:],
    }


def try_bitsandbytes(model_dir: Path) -> dict:
    """Fallback: 4-bit bitsandbytes quantized load + generate."""
    record: dict[str, object] = {"quant_method": "bitsandbytes_nf4"}
    try:
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        record["status"] = "blocked"
        record["error"] = f"missing dep: {exc}"
        return record

    try:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tok = AutoTokenizer.from_pretrained(str(model_dir),
                                            trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            quantization_config=bnb,
            trust_remote_code=True,
            device_map="auto",
        )
        inputs = tok("hello", return_tensors="pt").to(model.device)
        t0 = time.monotonic()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5,
                                 do_sample=False)
        tok_s = 5 / max(time.monotonic() - t0, 1e-6)
        record.update({
            "status": "ok",
            "tokens_s": round(tok_s, 2),
            "response": tok.decode(
                out[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ),
        })
    except Exception as exc:  # pragma: no cover
        record["status"] = "bnb_failed"
        record["error"] = f"{type(exc).__name__}: {exc}"

    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--llama-cpp", type=Path, default=DEFAULT_LLAMA_CPP)
    parser.add_argument("--out-gguf", type=Path, default=DEFAULT_OUT_GGUF)
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "spikingbrain-quant.json",
    )
    parser.add_argument(
        "--skip-bnb-fallback",
        action="store_true",
        help="Do not try bitsandbytes if llama.cpp path fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config + exit without spawning processes.",
    )
    args = parser.parse_args()

    record: dict[str, object] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_dir": str(args.model_dir),
        "llama_cpp_dir": str(args.llama_cpp),
        "out_gguf": str(args.out_gguf),
    }

    if args.dry_run:
        record["status"] = "dry_run"
        record["plan"] = [
            f"convert {args.model_dir} -> {args.out_gguf}.bf16.gguf",
            f"quantize -> {args.out_gguf} (Q4_K_M)",
            "smoke via llama-cli (5 tokens, 'hello')",
            "fallback: bitsandbytes nf4 if llama.cpp fails",
        ]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        return 0

    if not args.model_dir.exists():
        record["status"] = "no_weights"
        record["error"] = f"weights missing at {args.model_dir}"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        return 2

    llama_cpp_result = try_llama_cpp(
        args.model_dir, args.llama_cpp, args.out_gguf
    )
    record["llama_cpp"] = llama_cpp_result

    if llama_cpp_result.get("status") == "ok":
        smoke = try_smoke_gguf(args.out_gguf, args.llama_cpp)
        record["llama_cpp"].update(smoke)
        record["status"] = "ok"
    elif args.skip_bnb_fallback:
        record["status"] = "llama_cpp_failed_no_fallback"
    else:
        record["fallback"] = try_bitsandbytes(args.model_dir)
        record["status"] = (
            "fallback_ok"
            if record["fallback"].get("status") == "ok"
            else "all_paths_failed"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0 if record["status"] in ("ok", "fallback_ok") else 1


if __name__ == "__main__":
    sys.exit(main())
