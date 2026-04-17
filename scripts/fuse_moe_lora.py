#!/usr/bin/env python3
"""Fuse MoE-LoRA adapters into base model — auto MLX or CUDA.

Detects platform and uses MLX (Apple Silicon) or PyTorch (CUDA/CPU).
Handles key mapping between base model and adapter namespaces.
"""
import json
import shutil
import sys
import os
from pathlib import Path

# ── Auto-detect backend ─────────────────────────────────────────────────────
BACKEND = "numpy"  # fallback
try:
    import mlx.core as mx
    BACKEND = "mlx"
except ImportError:
    pass

if BACKEND != "mlx":
    try:
        import torch
        BACKEND = "torch"
    except ImportError:
        pass

print(f"Backend: {BACKEND}")

import numpy as np
from safetensors.numpy import save_file as np_save_file

# ── Config ───────────────────────────────────────────────────────────────────
ALPHA = 32.0
RANK = 16
SCALING = ALPHA / RANK

PROJ_NAMES = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def load_tensors(path):
    """Load safetensors file with auto backend."""
    if BACKEND == "mlx":
        return mx.load(str(path))
    elif BACKEND == "torch":
        from safetensors.torch import load_file
        return load_file(str(path))
    else:
        from safetensors.numpy import safe_open
        f = safe_open(str(path), framework="numpy")
        return {k: f.get_tensor(k) for k in f.keys()}


def to_float32(t):
    if BACKEND == "mlx":
        return t.astype(mx.float32)
    elif BACKEND == "torch":
        return t.float()
    return t.astype(np.float32)


def to_numpy(t):
    if BACKEND == "mlx":
        return np.array(t.astype(mx.float16))
    elif BACKEND == "torch":
        return t.half().cpu().numpy()
    return t.astype(np.float16) if t.dtype != np.float16 else t


def matmul(a, b):
    if BACKEND == "mlx":
        return a @ b
    elif BACKEND == "torch":
        return a @ b
    return a @ b


def softmax(x):
    if BACKEND == "mlx":
        return mx.softmax(x)
    elif BACKEND == "torch":
        return torch.softmax(x, dim=-1)
    e = np.exp(x - x.max())
    return e / e.sum()


def zeros_like(t):
    if BACKEND == "mlx":
        return mx.zeros(t.shape, dtype=mx.float32)
    elif BACKEND == "torch":
        return torch.zeros_like(t, dtype=torch.float32)
    return np.zeros(t.shape, dtype=np.float32)


def build_key_mapping(base_keys, adapter_keys):
    """Build mapping: base_key -> adapter_prefix for MoE-LoRA fusion.

    Base: model.language_model.layers.X.{proj}.weight
    Adapter: language_model.model.layers.X.{proj}_moe_lora.experts.N.lora_a
    """
    mapping = {}

    for base_key in base_keys:
        for proj in PROJ_NAMES:
            # Extract layer index from base key
            # Pattern: ...layers.{N}.{proj}.weight
            proj_with_dot = proj + ".weight"
            if proj_with_dot not in base_key:
                continue

            # Find layer number
            parts = base_key.split("layers.")
            if len(parts) < 2:
                continue
            layer_part = parts[-1]
            layer_idx = layer_part.split(".")[0]
            try:
                layer_idx = int(layer_idx)
            except ValueError:
                continue

            # Build adapter prefix
            # Try both naming conventions
            for prefix_style in [
                f"language_model.model.layers.{layer_idx}.{proj}_moe_lora",
                f"model.layers.{layer_idx}.{proj}_moe_lora",
            ]:
                test_key = f"{prefix_style}.experts.0.lora_a"
                if test_key in adapter_keys:
                    mapping[base_key] = prefix_style
                    break

    return mapping


def fuse_projection(weight, adapter, prefix):
    """Fuse 4 MoE-LoRA experts weighted by router into base weight."""
    # Router weights
    router_bias_key = f"{prefix}.router_w2.bias"
    if router_bias_key in adapter:
        ew = softmax(to_float32(adapter[router_bias_key]))
    else:
        if BACKEND == "mlx":
            ew = mx.ones(4) / 4.0
        elif BACKEND == "torch":
            ew = torch.ones(4) / 4.0
        else:
            ew = np.ones(4) / 4.0

    delta = zeros_like(weight)
    for i in range(4):
        la = to_float32(adapter[f"{prefix}.experts.{i}.lora_a"])
        lb = to_float32(adapter[f"{prefix}.experts.{i}.lora_b"])
        # LoRA: delta_W = B.T @ A.T where A=(in,rank), B=(rank,out)
        # Result shape: (out, in) matches weight shape
        if BACKEND == "mlx":
            expert_delta = mx.transpose(lb) @ mx.transpose(la) * SCALING
        elif BACKEND == "torch":
            expert_delta = lb.T @ la.T * SCALING
        else:
            expert_delta = lb.T @ la.T * SCALING
        if BACKEND == "mlx":
            delta = delta + ew[i] * expert_delta
        elif BACKEND == "torch":
            delta = delta + ew[i].item() * expert_delta
        else:
            delta = delta + ew[i] * expert_delta

    fused = to_float32(weight) + delta
    # Cast back to original dtype
    if BACKEND == "mlx":
        return fused.astype(weight.dtype)
    elif BACKEND == "torch":
        return fused.to(weight.dtype)
    return fused.astype(weight.dtype)


def main():
    base_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/Qwen3.5-4B")
    adapter_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output/micro-kiki/stacks/python")
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("output/micro-kiki/gguf/fused-model")

    print(f"Base: {base_path}")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {output_dir}")

    adapter = load_tensors(adapter_path / "adapters.safetensors")
    print(f"Adapter: {len(adapter)} tensors")
    adapter_keys = set(adapter.keys())

    # Copy config files
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in base_path.iterdir():
        if f.suffix == ".safetensors":
            continue
        dst = output_dir / f.name
        if f.is_file() and not dst.exists():
            shutil.copy2(f, dst)
        elif f.is_dir() and not dst.exists():
            shutil.copytree(f, dst)

    # Process shards
    fused_count = 0
    for shard_file in sorted(base_path.glob("*.safetensors")):
        print(f"\n  {shard_file.name}...")
        base = load_tensors(shard_file)

        # Build key mapping for this shard
        mapping = build_key_mapping(base.keys(), adapter_keys)
        print(f"    Fusable projections in this shard: {len(mapping)}")

        modified = {}
        for key, weight in base.items():
            if key in mapping:
                fused = fuse_projection(weight, adapter, mapping[key])
                fused_count += 1
                modified[key] = to_numpy(fused)
            else:
                modified[key] = to_numpy(weight)

        out_path = output_dir / shard_file.name
        np_save_file(modified, str(out_path))
        print(f"    Saved ({len(modified)} tensors, {fused_count} fused total)")

    print(f"\n{'='*60}")
    print(f"Fused {fused_count} projections")
    print(f"Output: {output_dir}")
    print(f"\nConvert to GGUF:")
    if Path("/Users/clems/llama.cpp/convert_hf_to_gguf.py").exists():
        print(f"  python ~/llama.cpp/convert_hf_to_gguf.py {output_dir} --outfile {output_dir.parent}/micro-kiki-v3-f16.gguf --outtype f16")
        print(f"  ~/llama.cpp/build/bin/llama-quantize {output_dir.parent}/micro-kiki-v3-f16.gguf {output_dir.parent}/micro-kiki-v3-Q4_K_M.gguf Q4_K_M")
    else:
        print("  Install llama.cpp first")


if __name__ == "__main__":
    main()
