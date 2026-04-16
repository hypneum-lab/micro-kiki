#!/usr/bin/env python3
"""LAS conversion: Qwen3.5-35B-A3B MoE → SpikingKiki-35B-A3B.

This is the most architecturally interesting conversion in the micro-kiki
pipeline: a 256-expert MoE (3B active per token) transformed into a spiking
network where expert routing is preserved losslessly via rate-coded identity
activations on the router and ReLU-coded activations on each expert FFN.

Key design choices:
- Router converted with activation="identity" — signed logits kept intact
  so top-k expert selection is preserved (same semantics as LASConverter MoE).
- Expert FFNs converted with activation="relu" — standard rate-code path.
- Attention projections converted with activation="identity" — residual
  stream stability.
- Layer-by-layer with progress logging and resumable state.
- ~40h estimated wall time on Mac Studio M3 Ultra; --resume skips layers
  already present in the output directory.

Architecture of Qwen3.5-35B-A3B per transformer block:
  - Attention: q_proj, k_proj, v_proj, o_proj
  - MoE FFN: gate (router 7168→256), 256 experts each with gate_proj +
    up_proj + down_proj (SwiGLU pattern)
  - Total ~94 transformer blocks

Usage:
  uv run python scripts/convert_spikingkiki_35b.py \\
      --input  /path/to/Qwen3.5-35B-A3B \\
      --output /path/to/SpikingKiki-35B-A3B \\
      --timesteps 128 \\
      --resume

  # Dry-run (no torch required):
  uv run python scripts/convert_spikingkiki_35b.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# Qwen3.5-35B-A3B architecture constants
NUM_LAYERS = 94
NUM_EXPERTS = 256
TOP_K = 8  # Qwen3.5-35B-A3B uses top-8 routing
HIDDEN_DIM = 7168
EXPERT_INTERMEDIATE = 2048  # per-expert intermediate dim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("convert_35b")


# ---------------------------------------------------------------------------
# Layer map generation (pure Python, no torch)
# ---------------------------------------------------------------------------


def build_layer_map(num_layers: int = NUM_LAYERS) -> list[dict[str, Any]]:
    """Return the canonical layer map for Qwen3.5-35B-A3B.

    Each entry describes one logical conversion unit with its weight key
    path inside the HuggingFace state_dict.

    Returns
    -------
    list of dicts with keys:
        layer_id   : int   — transformer block index
        kind       : str   — "attn_proj" | "moe_router" | "moe_expert_ffn"
        key_prefix : str   — state_dict key prefix
        expert_id  : int | None
        activation : str   — "identity" or "relu"
        converted  : bool  — False initially (set True when done)
    """
    entries: list[dict[str, Any]] = []
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"

        # Attention projections — identity activation to preserve residual
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            entries.append(
                {
                    "layer_id": layer_idx,
                    "kind": "attn_proj",
                    "key_prefix": f"{prefix}.self_attn.{proj}",
                    "expert_id": None,
                    "activation": "identity",
                    "converted": False,
                }
            )

        # MoE router — identity activation (signed logits must be preserved)
        entries.append(
            {
                "layer_id": layer_idx,
                "kind": "moe_router",
                "key_prefix": f"{prefix}.mlp.gate",
                "expert_id": None,
                "activation": "identity",
                "converted": False,
            }
        )

        # MoE expert FFNs — relu activation (SwiGLU gate + up + down)
        for expert_id in range(NUM_EXPERTS):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                entries.append(
                    {
                        "layer_id": layer_idx,
                        "kind": "moe_expert_ffn",
                        "key_prefix": (
                            f"{prefix}.mlp.experts.{expert_id}.{proj}"
                        ),
                        "expert_id": expert_id,
                        "activation": "relu",
                        "converted": False,
                    }
                )

    return entries


def layer_map_stats(layer_map: list[dict[str, Any]]) -> dict[str, int]:
    """Summarise the layer map by kind."""
    counts: dict[str, int] = {}
    for entry in layer_map:
        counts[entry["kind"]] = counts.get(entry["kind"], 0) + 1
    counts["total"] = len(layer_map)
    return counts


# ---------------------------------------------------------------------------
# Resume logic — check which layers are already saved
# ---------------------------------------------------------------------------


def load_resume_state(output_dir: Path) -> set[str]:
    """Return set of key_prefixes already converted (saved to disk)."""
    state_file = output_dir / ".convert_state.json"
    if not state_file.exists():
        return set()
    try:
        data = json.loads(state_file.read_text())
        return set(data.get("converted_keys", []))
    except (json.JSONDecodeError, OSError):
        log.warning("could not parse resume state — starting fresh")
        return set()


def save_resume_state(output_dir: Path, converted_keys: set[str]) -> None:
    """Persist the set of converted key_prefixes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = output_dir / ".convert_state.json"
    state_file.write_text(
        json.dumps({"converted_keys": sorted(converted_keys)}, indent=2)
    )


# ---------------------------------------------------------------------------
# Torch-dependent conversion (guarded — works without torch for testing)
# ---------------------------------------------------------------------------


def _try_import_torch() -> Any:
    try:
        import torch  # type: ignore
        return torch
    except ImportError:
        return None


def _try_import_transformers() -> Any:
    try:
        from transformers import AutoModelForCausalLM, AutoConfig  # type: ignore
        return AutoModelForCausalLM, AutoConfig
    except ImportError:
        return None, None


def _extract_linear_weights(module: Any) -> dict[str, Any]:
    """Extract weight + bias as numpy from a torch.nn.Linear."""
    import numpy as np
    w = module.weight.detach().cpu().float().numpy()
    b = (
        module.bias.detach().cpu().float().numpy()
        if module.bias is not None
        else None
    )
    return {"weight": w, "bias": b}


def convert_block_torch(
    model: Any,
    layer_idx: int,
    converter: Any,
    spike_stats: dict[str, Any],
    converted_keys: set[str],
    output_dir: Path,
    timesteps: int,
) -> dict[str, Any]:
    """Convert one transformer block, saving spiking layers to disk.

    Returns a dict of spike stats for the block (counts + activation bounds).
    """
    import numpy as np
    from src.spiking.las_converter import LASConverter

    block = model.model.layers[layer_idx]
    prefix = f"model.layers.{layer_idx}"
    block_stats: dict[str, Any] = {"layer_id": layer_idx, "projections": {}}

    # --- Attention projections ---
    attn = block.self_attn
    for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        key = f"{prefix}.self_attn.{proj_name}"
        if key in converted_keys:
            log.debug("skip (resume) %s", key)
            continue
        proj_module = getattr(attn, proj_name)
        weights = _extract_linear_weights(proj_module)
        spiking = converter.convert_layer(weights, activation="identity")
        _save_spiking_layer(spiking, key, output_dir)
        w_abs_max = float(np.abs(weights["weight"]).max())
        block_stats["projections"][proj_name] = {
            "in": spiking.in_features,
            "out": spiking.out_features,
            "w_abs_max": round(w_abs_max, 4),
        }
        converted_keys.add(key)

    # --- MoE router ---
    router_key = f"{prefix}.mlp.gate"
    if router_key not in converted_keys:
        router_weights = _extract_linear_weights(block.mlp.gate)
        spiking_router = converter.convert_layer(
            router_weights, activation="identity"
        )
        _save_spiking_layer(spiking_router, router_key, output_dir)
        block_stats["router"] = {
            "in": spiking_router.in_features,
            "out": spiking_router.out_features,
        }
        converted_keys.add(router_key)
    else:
        log.debug("skip (resume) %s", router_key)

    # --- MoE experts ---
    experts_module = block.mlp.experts
    expert_spike_counts: list[float] = []
    for expert_id in range(len(experts_module)):
        expert = experts_module[expert_id]
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            key = f"{prefix}.mlp.experts.{expert_id}.{proj_name}"
            if key in converted_keys:
                continue
            proj_module = getattr(expert, proj_name)
            weights = _extract_linear_weights(proj_module)
            spiking = converter.convert_layer(weights, activation="relu")
            _save_spiking_layer(spiking, key, output_dir)
            # Rough spike count estimate via max weight * timesteps
            est_spikes = float(
                np.abs(weights["weight"]).mean() * timesteps
            )
            expert_spike_counts.append(est_spikes)
            converted_keys.add(key)

    if expert_spike_counts:
        block_stats["expert_spike_rate_mean"] = round(
            float(np.mean(expert_spike_counts)), 4
        )

    return block_stats


def _save_spiking_layer(spiking: Any, key: str, output_dir: Path) -> None:
    """Save a SpikingLinear weight array as .npz alongside a metadata JSON."""
    import numpy as np
    safe_key = key.replace(".", "_").replace("/", "_")
    out_path = output_dir / f"{safe_key}.npz"
    np.savez_compressed(
        out_path,
        weight=spiking.weight,
        bias=spiking.bias if spiking.bias is not None else np.array([]),
        timesteps=np.array([spiking.timesteps]),
        max_rate=np.array([spiking.max_rate]),
    )


# ---------------------------------------------------------------------------
# Metadata assembly
# ---------------------------------------------------------------------------


def build_metadata(
    args: argparse.Namespace,
    layer_map: list[dict[str, Any]],
    spike_stats: dict[str, Any],
    elapsed_s: float,
    status: str,
) -> dict[str, Any]:
    """Assemble the results JSON structure."""
    map_stats = layer_map_stats(layer_map)
    converted = sum(1 for e in layer_map if e.get("converted"))
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "input": str(args.input),
        "output": str(args.output),
        "timesteps": args.timesteps,
        "model_info": {
            "name": "Qwen3.5-35B-A3B",
            "num_layers": NUM_LAYERS,
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "hidden_dim": HIDDEN_DIM,
            "expert_intermediate_dim": EXPERT_INTERMEDIATE,
        },
        "layer_map_stats": map_stats,
        "converted_layers": converted,
        "total_layers": len(layer_map),
        "elapsed_s": round(elapsed_s, 1),
        "spike_stats": spike_stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LAS conversion: Qwen3.5-35B-A3B MoE → SpikingKiki-35B-A3B"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/Volumes/studio/models/Qwen3.5-35B-A3B"
        ),
        help="Path to the HuggingFace model directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/Volumes/studio/models/SpikingKiki-35B-A3B"
        ),
        help="Output directory for converted spiking layers.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=128,
        help="LIF integration timesteps T (default 128).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip layers already saved to --output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate layer map and metadata without loading weights.",
    )
    parser.add_argument(
        "--results-out",
        type=Path,
        default=RESULTS_DIR / "spikingkiki-35b-convert.json",
        help="Path for the conversion metadata JSON.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.getLogger().setLevel(args.log_level)

    t_start = time.monotonic()

    log.info(
        "SpikingKiki-35B conversion — timesteps=%d resume=%s dry_run=%s",
        args.timesteps,
        args.resume,
        args.dry_run,
    )

    layer_map = build_layer_map()
    map_stats = layer_map_stats(layer_map)
    log.info(
        "layer map: %d total (%s)",
        map_stats["total"],
        ", ".join(f"{k}={v}" for k, v in map_stats.items() if k != "total"),
    )

    spike_stats: dict[str, Any] = {}

    # --- Dry-run path (no torch needed) ---
    if args.dry_run:
        log.info("dry-run: skipping weight loading")
        meta = build_metadata(
            args, layer_map, spike_stats,
            elapsed_s=time.monotonic() - t_start,
            status="dry_run",
        )
        meta["layer_map"] = layer_map
        args.results_out.parent.mkdir(parents=True, exist_ok=True)
        args.results_out.write_text(json.dumps(meta, indent=2) + "\n")
        log.info("metadata written to %s", args.results_out)
        return 0

    # --- Check model directory ---
    if not args.input.exists():
        log.error("model directory not found: %s", args.input)
        meta = build_metadata(
            args, layer_map, spike_stats,
            elapsed_s=time.monotonic() - t_start,
            status="no_weights",
        )
        args.results_out.parent.mkdir(parents=True, exist_ok=True)
        args.results_out.write_text(json.dumps(meta, indent=2) + "\n")
        return 2

    # --- Import torch (guarded) ---
    torch = _try_import_torch()
    if torch is None:
        log.error("torch not available — cannot load model weights")
        meta = build_metadata(
            args, layer_map, spike_stats,
            elapsed_s=time.monotonic() - t_start,
            status="torch_missing",
        )
        args.results_out.parent.mkdir(parents=True, exist_ok=True)
        args.results_out.write_text(json.dumps(meta, indent=2) + "\n")
        return 3

    AutoModelForCausalLM, AutoConfig = _try_import_transformers()
    if AutoModelForCausalLM is None:
        log.error("transformers not available")
        meta = build_metadata(
            args, layer_map, spike_stats,
            elapsed_s=time.monotonic() - t_start,
            status="transformers_missing",
        )
        args.results_out.parent.mkdir(parents=True, exist_ok=True)
        args.results_out.write_text(json.dumps(meta, indent=2) + "\n")
        return 3

    # --- Resume state ---
    converted_keys: set[str] = set()
    if args.resume:
        converted_keys = load_resume_state(args.output)
        log.info("resume: %d layers already converted", len(converted_keys))

    # --- Load model ---
    log.info("loading model from %s (bfloat16, cpu map) …", args.input)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(args.input),
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        model.eval()
    except Exception as exc:
        log.error("model load failed: %s: %s", type(exc).__name__, exc)
        meta = build_metadata(
            args, layer_map, spike_stats,
            elapsed_s=time.monotonic() - t_start,
            status="load_failed",
        )
        meta["error"] = str(exc)
        args.results_out.parent.mkdir(parents=True, exist_ok=True)
        args.results_out.write_text(json.dumps(meta, indent=2) + "\n")
        return 4

    # --- LASConverter ---
    from src.spiking.las_converter import LASConverter
    converter = LASConverter(timesteps=args.timesteps, max_rate=1.0)

    args.output.mkdir(parents=True, exist_ok=True)

    # --- Layer-by-layer conversion ---
    n_total = NUM_LAYERS
    for layer_idx in range(n_total):
        t_layer = time.monotonic()
        log.info(
            "[%d/%d] converting block %d …", layer_idx + 1, n_total, layer_idx
        )

        try:
            block_stats = convert_block_torch(
                model=model,
                layer_idx=layer_idx,
                converter=converter,
                spike_stats=spike_stats,
                converted_keys=converted_keys,
                output_dir=args.output,
                timesteps=args.timesteps,
            )
        except Exception as exc:
            log.error(
                "block %d conversion failed: %s: %s",
                layer_idx, type(exc).__name__, exc,
            )
            spike_stats[f"layer_{layer_idx}"] = {
                "status": "failed",
                "error": str(exc),
            }
            # Save resume state so we can skip completed layers on retry
            save_resume_state(args.output, converted_keys)
            continue

        spike_stats[f"layer_{layer_idx}"] = block_stats
        elapsed_block = time.monotonic() - t_layer
        log.info(
            "  block %d done in %.1fs (%d keys converted so far)",
            layer_idx,
            elapsed_block,
            len(converted_keys),
        )

        # Periodic resume state flush (every 5 blocks)
        if layer_idx % 5 == 0:
            save_resume_state(args.output, converted_keys)
            log.debug("resume state flushed (%d keys)", len(converted_keys))

    save_resume_state(args.output, converted_keys)

    # Mark converted layers in the layer map
    for entry in layer_map:
        entry["converted"] = entry["key_prefix"] in converted_keys

    elapsed_total = time.monotonic() - t_start
    log.info("conversion complete in %.1fs", elapsed_total)

    meta = build_metadata(
        args, layer_map, spike_stats,
        elapsed_s=elapsed_total,
        status="ok",
    )
    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    args.results_out.write_text(json.dumps(meta, indent=2) + "\n")
    log.info("metadata written to %s", args.results_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
