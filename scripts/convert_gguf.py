#!/usr/bin/env python3
"""Convert fused model to GGUF with auto MLX/CUDA detection.

On Apple Silicon (MLX): uses mlx_lm to quantize to MLX format, then
optionally converts to GGUF via llama.cpp if available.

On CUDA/CPU (PyTorch): uses convert_hf_to_gguf.py + llama-quantize directly.

Usage:
    python convert_gguf_auto.py <model_dir> [--outdir <dir>] [--quant Q4_K_M]
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── Auto-detect ──────────────────────────────────────────────────────────────
PLATFORM = "cpu"
try:
    import mlx.core
    PLATFORM = "mlx"
except ImportError:
    pass

if PLATFORM != "mlx":
    try:
        import torch
        if torch.cuda.is_available():
            PLATFORM = "cuda"
        else:
            PLATFORM = "torch-cpu"
    except ImportError:
        pass

print(f"Platform: {PLATFORM}")


def find_llama_cpp():
    """Find llama.cpp tools."""
    candidates = [
        Path.home() / "llama.cpp",
        Path("/opt/homebrew"),
        Path("/usr/local"),
        Path.home() / "llama.cpp" / "build" / "bin",
    ]
    convert_script = None
    quantize_bin = None

    for base in candidates:
        c = base / "convert_hf_to_gguf.py"
        if c.exists():
            convert_script = c
        q = base / "build" / "bin" / "llama-quantize"
        if q.exists():
            quantize_bin = q
        q2 = base / "bin" / "llama-quantize"
        if q2.exists():
            quantize_bin = q2
        q3 = base / "llama-quantize"
        if q3.exists():
            quantize_bin = q3

    # Also check PATH
    if not quantize_bin:
        r = shutil.which("llama-quantize")
        if r:
            quantize_bin = Path(r)

    return convert_script, quantize_bin


def convert_mlx(model_dir, out_dir, quant):
    """Apple Silicon path: MLX quantize + optional GGUF."""
    print("\n=== MLX Path ===")

    # Step 1: Quantize with mlx_lm
    mlx_quant_dir = out_dir / "mlx-quantized"
    q_bits = {"Q4_K_M": 4, "Q8_0": 8, "Q6_K": 6}.get(quant, 4)

    print(f"Quantizing to {q_bits}-bit MLX format...")
    subprocess.run([
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", str(model_dir),
        "--mlx-path", str(mlx_quant_dir),
        "-q", "--q-bits", str(q_bits),
    ], check=True)
    print(f"MLX quantized: {mlx_quant_dir}")

    # Step 2: Try GGUF conversion via llama.cpp
    convert_script, quantize_bin = find_llama_cpp()

    if convert_script:
        print("\nConverting to GGUF F16...")
        gguf_f16 = out_dir / f"micro-kiki-v3-f16.gguf"
        try:
            subprocess.run([
                sys.executable, str(convert_script),
                str(model_dir),
                "--outfile", str(gguf_f16),
                "--outtype", "f16",
            ], check=True)
            print(f"F16 GGUF: {gguf_f16}")

            if quantize_bin:
                gguf_quant = out_dir / f"micro-kiki-v3-{quant}.gguf"
                print(f"\nQuantizing to {quant}...")
                subprocess.run([
                    str(quantize_bin),
                    str(gguf_f16),
                    str(gguf_quant),
                    quant,
                ], check=True)
                size_mb = gguf_quant.stat().st_size / (1024 * 1024)
                print(f"GGUF {quant}: {gguf_quant} ({size_mb:.0f} MB)")

                # Clean F16
                gguf_f16.unlink()
                return gguf_quant
            else:
                print("llama-quantize not found, keeping F16")
                return gguf_f16

        except subprocess.CalledProcessError as e:
            print(f"GGUF conversion failed: {e}")
            print("MLX quantized model available at:", mlx_quant_dir)
            return mlx_quant_dir
    else:
        print("convert_hf_to_gguf.py not found, MLX format only")
        return mlx_quant_dir


def convert_cuda(model_dir, out_dir, quant):
    """CUDA/PyTorch path: direct GGUF conversion."""
    print("\n=== CUDA/PyTorch Path ===")

    convert_script, quantize_bin = find_llama_cpp()

    if not convert_script:
        print("ERROR: convert_hf_to_gguf.py not found")
        print("Install llama.cpp: git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build")
        sys.exit(1)

    gguf_f16 = out_dir / "micro-kiki-v3-f16.gguf"
    print("Converting to GGUF F16...")
    subprocess.run([
        sys.executable, str(convert_script),
        str(model_dir),
        "--outfile", str(gguf_f16),
        "--outtype", "f16",
    ], check=True)
    print(f"F16 GGUF: {gguf_f16} ({gguf_f16.stat().st_size / (1024*1024):.0f} MB)")

    if quantize_bin:
        gguf_quant = out_dir / f"micro-kiki-v3-{quant}.gguf"
        print(f"\nQuantizing to {quant}...")
        subprocess.run([
            str(quantize_bin),
            str(gguf_f16),
            str(gguf_quant),
            quant,
        ], check=True)
        size_mb = gguf_quant.stat().st_size / (1024 * 1024)
        print(f"GGUF {quant}: {gguf_quant} ({size_mb:.0f} MB)")
        gguf_f16.unlink()
        return gguf_quant
    else:
        print("llama-quantize not found, keeping F16")
        return gguf_f16


def main():
    ap = argparse.ArgumentParser(description="Convert fused model to GGUF (auto MLX/CUDA)")
    ap.add_argument("model_dir", help="Path to fused HF model directory")
    ap.add_argument("--outdir", default=None, help="Output directory (default: <model_dir>/../gguf)")
    ap.add_argument("--quant", default="Q4_K_M", help="Quantization type (default: Q4_K_M)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.outdir) if args.outdir else model_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_dir}")
    print(f"Output: {out_dir}")
    print(f"Quant: {args.quant}")

    if PLATFORM == "mlx":
        result = convert_mlx(model_dir, out_dir, args.quant)
    elif PLATFORM in ("cuda", "torch-cpu"):
        result = convert_cuda(model_dir, out_dir, args.quant)
    else:
        print("ERROR: No ML framework found. Install mlx or torch.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Done! Output: {result}")
    print(f"\nTest with llama-server:")
    print(f"  llama-server -m {result} -c 4096 --port 8080")


if __name__ == "__main__":
    main()
