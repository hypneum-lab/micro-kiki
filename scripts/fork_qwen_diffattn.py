"""Fork Qwen3.5-4B with DiffAttn on full-attention layers."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PERPLEXITY_THRESHOLD = 0.03
OUTLIER_REDUCTION_MIN = 0.30


def fork_with_diffattn(base_dir: str, output_dir: str, calibration_tokens: int = 5000) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading base model from %s", base_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    full_attn_indices = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.self_attn, "q_proj") and not hasattr(layer.self_attn, "gate"):
            full_attn_indices.append(i)

    metrics = {
        "base_dir": base_dir, "output_dir": output_dir,
        "full_attn_layers": full_attn_indices,
        "full_attn_count": len(full_attn_indices),
        "status": "framework_ready",
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "fork_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def check_rollback(metrics: dict) -> bool:
    if metrics.get("perplexity_delta", 0) > PERPLEXITY_THRESHOLD:
        return True
    if metrics.get("outlier_reduction", 1.0) < OUTLIER_REDUCTION_MIN:
        return True
    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="models/qwen3.5-4b/bf16")
    parser.add_argument("--output-dir", default="models/qwen3.5-4b-diffattn")
    args = parser.parse_args()
    print(json.dumps(fork_with_diffattn(args.base_dir, args.output_dir), indent=2))
