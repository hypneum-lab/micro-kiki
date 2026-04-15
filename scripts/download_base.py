"""Download and quantize Qwen3.5-4B base model."""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B"
DEFAULT_BF16_DIR = "models/qwen3.5-4b/bf16"
DEFAULT_Q4_PATH = "models/qwen3.5-4b-q4.gguf"


def download_bf16(model_id: str = DEFAULT_MODEL_ID, output_dir: str = DEFAULT_BF16_DIR) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s", model_id, out)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(out),
        ignore_patterns=["*.bin", "*.pt", "original/*"],
    )
    safetensors = list(out.glob("*.safetensors"))
    if not safetensors:
        raise FileNotFoundError(f"No safetensors files found in {out}")
    logger.info("Downloaded %d safetensors files", len(safetensors))
    return out


def quantize_q4(bf16_dir: str = DEFAULT_BF16_DIR, output_path: str = DEFAULT_Q4_PATH) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    gguf_f16 = out.with_suffix(".f16.gguf")
    logger.info("Converting HF -> GGUF F16")
    subprocess.run(
        ["python", "-m", "llama_cpp.convert_hf_to_gguf", bf16_dir, "--outfile", str(gguf_f16)],
        check=True,
    )
    logger.info("Quantizing F16 -> Q4_K_M")
    subprocess.run(["llama-quantize", str(gguf_f16), str(out), "Q4_K_M"], check=True)
    if gguf_f16.exists():
        gguf_f16.unlink()
    logger.info("Q4_K_M saved to %s (%.1f GB)", out, out.stat().st_size / 1e9)
    return out


def verify_download(bf16_dir: str = DEFAULT_BF16_DIR, q4_path: str = DEFAULT_Q4_PATH) -> dict:
    bf16 = Path(bf16_dir)
    q4 = Path(q4_path)
    result = {"bf16_exists": bf16.exists(), "q4_exists": q4.exists()}
    if bf16.exists():
        total_bf16 = sum(f.stat().st_size for f in bf16.glob("*.safetensors"))
        result["bf16_size_gb"] = total_bf16 / 1e9
        result["bf16_ok"] = 6.0 < result["bf16_size_gb"] < 12.0
    if q4.exists():
        result["q4_size_gb"] = q4.stat().st_size / 1e9
        result["q4_ok"] = 1.5 < result["q4_size_gb"] < 4.0
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Download Qwen3.5-4B base model")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--bf16-dir", default=DEFAULT_BF16_DIR)
    parser.add_argument("--q4-path", default=DEFAULT_Q4_PATH)
    parser.add_argument("--skip-quantize", action="store_true")
    args = parser.parse_args()
    download_bf16(args.model_id, args.bf16_dir)
    if not args.skip_quantize:
        quantize_q4(args.bf16_dir, args.q4_path)
    print(verify_download(args.bf16_dir, args.q4_path))
