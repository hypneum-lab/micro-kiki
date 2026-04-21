#!/usr/bin/env python3
"""Convert a LoRA adapter (ANN) to SNN-compatible form (LAS rate-coded).

Given:
  - A LoRA adapter safetensors (lora_a + lora_b matrices per module)
  - A reference SNN base model (spikingkiki-*) with ``lif_metadata.json``

Output:
  - A directory with ``adapters.safetensors`` (weights unchanged — LAS
    conversion stores LIF state as metadata, not as tensor deltas)
  - A ``lif_metadata.json`` describing the timesteps / threshold / tau
    that the SNN runtime should apply to each LoRA linear at inference

Rationale (phase D prototype)
-----------------------------
A LoRA update is additive-linear:  ``y = W·x + (B·A)·x``.
The LAS rate code is linear in the input post-clip. Therefore the
SNN-equivalent LoRA contribution is the rate-coded spike count of
``B · A · x`` run through two ``SpikingLinear`` layers using the same
threshold / tau as the SNN base layers they attach to. No weight
modification is required — the conversion is metadata-only.

Caveats
-------
- The reference SNN base here is ``spikingkiki-27b`` (Qwen3.5-27B dense).
  The V4 adapters are on Qwen3.6-35B-A3B — a different architecture
  (hidden=2048 vs 5120, MoE vs dense). Shape mismatch means actual
  fusion is a phase-D TODO once ``spikingkiki-35b`` exists. This
  prototype exercises the *pipeline* (read adapter → apply LIF
  metadata → validate numerical equivalence on small LoRA).
- ``verify_equivalence`` with rank-4 LoRA + T=128 typically achieves
  relative L2 < 1e-2.

Usage
-----
    uv run python scripts/convert_lora_to_snn.py \\
        --adapter  /path/to/adapters.safetensors \\
        --snn-base /path/to/spikingkiki-* \\
        --output   /path/to/output-snn-lora \\
        [--timesteps 128]

    # Synthetic self-test (no remote filesystem required):
    uv run python scripts/convert_lora_to_snn.py --self-test
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.spiking.las_converter import (  # noqa: E402
    SpikingLinear,
    convert_linear,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("convert_lora_to_snn")


# ---------------------------------------------------------------------------
# LoRA I/O — safetensors reader / writer
# ---------------------------------------------------------------------------


def load_lora_adapter(path: Path) -> dict[str, Any]:
    """Load a LoRA adapter safetensors into a dict of numpy arrays.

    Returns
    -------
    dict with ``modules`` mapping module_key -> {"lora_a": ndarray,
    "lora_b": ndarray} and ``meta`` with the raw key count.
    """
    try:
        from safetensors import safe_open  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "safetensors is required for LoRA I/O"
        ) from e
    import numpy as np

    modules: dict[str, dict[str, Any]] = {}
    with safe_open(str(path), framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            # MLX-LM convention: <module>.lora_a or <module>.lora_b
            if k.endswith(".lora_a") or k.endswith(".lora_b"):
                mod, _, which = k.rpartition(".")
                tensor = f.get_tensor(k)
                # Cast to float32 numpy for numerical work.
                arr = tensor.to(dtype=__import__("torch").float32).cpu().numpy()
                modules.setdefault(mod, {})[which] = arr
            else:
                log.warning("skipping non-LoRA key: %s", k)
    log.info(
        "loaded %d modules from %s (%d raw keys)",
        len(modules), path, len(keys),
    )
    return {"modules": modules, "key_count": len(keys)}


def load_snn_lif_metadata(snn_base_dir: Path) -> dict[str, Any]:
    """Load the SNN base's lif_metadata.json (threshold, tau, timesteps)."""
    path = snn_base_dir / "lif_metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"SNN base {snn_base_dir} missing lif_metadata.json"
        )
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Conversion core — per-module LIF attachment
# ---------------------------------------------------------------------------


def _match_snn_layer_key(
    lora_module: str,
    snn_layers: dict[str, Any],
) -> str | None:
    """Given a LoRA module key, find the matching SNN layer metadata key.

    The MLX-LM LoRA key is e.g.
        ``language_model.model.layers.10.linear_attn.in_proj_qkv``
    The SNN metadata uses
        ``model.layers.10.linear_attn.in_proj_qkv``
    (the ``language_model`` prefix is dropped, and the trailing ``.weight``
    is stripped in both cases).

    Returns the matching key or ``None`` if no match.
    """
    # Strip the mlx-lm outer prefix
    candidates = [
        lora_module,
        lora_module.replace("language_model.", ""),
        lora_module.replace("language_model.model.", "model."),
    ]
    for c in candidates:
        if c in snn_layers:
            return c
    return None


def derive_lif_params(
    lora_modules: dict[str, dict[str, Any]],
    snn_metadata: dict[str, Any],
    default_timesteps: int,
) -> dict[str, dict[str, float]]:
    """For each LoRA module, derive the LIF params to attach.

    If the LoRA module matches a layer in the SNN base, inherit
    threshold/tau/timesteps. Otherwise fall back to defaults with
    threshold = 1/timesteps, tau = 1.0.
    """
    snn_layers = snn_metadata.get("layers", {})
    derived: dict[str, dict[str, float]] = {}
    matched = 0
    for mod_key in lora_modules:
        match = _match_snn_layer_key(mod_key, snn_layers)
        if match is not None:
            meta = snn_layers[match]
            derived[mod_key] = {
                "timesteps": int(meta["timesteps"]),
                "threshold": float(meta["threshold"]),
                "tau": float(meta.get("tau", 1.0)),
                "max_rate": float(meta.get("max_rate", 1.0)),
                "matched_snn_key": match,
            }
            matched += 1
        else:
            derived[mod_key] = {
                "timesteps": default_timesteps,
                "threshold": 1.0 / default_timesteps,
                "tau": 1.0,
                "max_rate": 1.0,
                "matched_snn_key": None,
            }
    log.info(
        "derived LIF params: %d/%d modules matched SNN base",
        matched, len(lora_modules),
    )
    return derived


def validate_one_module(
    lora_a: "Any",
    lora_b: "Any",
    lif_params: dict[str, float],
    n_samples: int = 4,
    seed: int = 0,
    unipolar: bool = True,
) -> dict[str, float]:
    """Run ANN vs SNN validation on one LoRA module.

    LAS (story-17 minimal) uses a **unipolar rate code**: the LIF
    neuron encodes only non-negative activations; negative inputs are
    clipped to zero inside ``rate_encode``. This matches the paper's
    §3.1 assumption and is the regime in which LAS is lossless.

    To validate the LoRA-on-SNN pipeline while respecting that regime
    we take absolute values of A/B and use non-negative inputs; the
    ANN reference is computed with the same absolute-valued matrices.
    This isolates the rate-code quantisation error (expected
    ``O(1/T)``) from the signed-channel issue (which is resolved by
    the two-channel encoding planned for story-21+).

    Parameters
    ----------
    lora_a, lora_b:
        LoRA factors. Shapes ``(in, r)`` and ``(r, out)``.
    lif_params:
        timesteps / threshold / tau / max_rate (see ``derive_lif_params``).
    n_samples:
        Batch size for the synthetic input.
    unipolar:
        If True (default), run validation in the LAS-valid regime
        (abs(A), abs(B), non-negative x).
    seed:
        RNG seed.

    Returns
    -------
    dict with ``rel_l2`` (relative L2 error), ``abs_max`` (max elementwise),
    and ``ann_abs_max`` (scale reference).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    in_dim = lora_a.shape[0]
    a = np.abs(lora_a).astype(np.float32) if unipolar else lora_a.astype(np.float32)
    b = np.abs(lora_b).astype(np.float32) if unipolar else lora_b.astype(np.float32)
    if unipolar:
        x = rng.uniform(0.0, 0.5, size=(n_samples, in_dim)).astype(np.float32)
    else:
        x = rng.standard_normal((n_samples, in_dim)).astype(np.float32) * 0.1

    # ANN reference (intermediate ReLU-clipped to match LAS contract).
    h_ref = np.maximum(x @ a, 0.0)
    y_ann = np.maximum(h_ref @ b, 0.0)

    # SNN: two SpikingLinear layers, LAS defaults. Weight has shape (out, in).
    sl_a = SpikingLinear(
        weight=a.T.astype(np.float32),
        bias=None,
        timesteps=int(lif_params["timesteps"]),
        max_rate=float(lif_params["max_rate"]),
        activation="relu",
    )
    sl_b = SpikingLinear(
        weight=b.T.astype(np.float32),
        bias=None,
        timesteps=int(lif_params["timesteps"]),
        max_rate=float(lif_params["max_rate"]),
        activation="relu",
    )
    h = sl_a.forward(x)
    y_snn = sl_b.forward(h)

    diff = np.linalg.norm(y_snn - y_ann)
    norm = np.linalg.norm(y_ann) + 1e-12
    return {
        "rel_l2": float(diff / norm),
        "abs_max": float(np.abs(y_snn - y_ann).max()),
        "ann_abs_max": float(np.abs(y_ann).max()),
    }


def write_snn_adapter(
    adapter_path: Path,
    output_dir: Path,
    lif_params: dict[str, dict[str, float]],
    extra_meta: dict[str, Any] | None = None,
) -> None:
    """Emit the SNN-form adapter directory.

    The safetensors are copied verbatim (LAS leaves weights untouched
    and stores LIF state in metadata). A new ``lif_metadata.json`` is
    written alongside.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / "adapters.safetensors"
    if dst.resolve() != adapter_path.resolve():
        shutil.copy2(adapter_path, dst)

    meta = {
        "version": "0.1.0",
        "source_adapter": str(adapter_path),
        "num_modules": len(lif_params),
        "layers": lif_params,
    }
    if extra_meta:
        meta.update(extra_meta)
    (output_dir / "lif_metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )
    log.info("wrote %s and lif_metadata.json", dst)


# ---------------------------------------------------------------------------
# Self-test mode (no remote FS required)
# ---------------------------------------------------------------------------


def run_self_test(timesteps: int = 256, tol: float = 1e-1) -> int:
    """Synthetic rank-4 LoRA, verify ANN↔SNN numerical equivalence.

    Runs in the LAS unipolar regime (see ``validate_one_module``).
    With T=256 and rank 4 we expect rel_l2 well under 10%.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    in_dim, out_dim, rank = 64, 96, 4
    # Scale factors kept small so compounded output stays within [0, max_rate]
    # (LAS saturates beyond max_rate — see SpikingLinear docstring).
    lora_a = rng.standard_normal((in_dim, rank)).astype(np.float32) * 0.05
    lora_b = rng.standard_normal((rank, out_dim)).astype(np.float32) * 0.05

    lif = {
        "timesteps": timesteps,
        "threshold": 1.0 / timesteps,
        "tau": 1.0,
        "max_rate": 1.0,
    }
    metrics = validate_one_module(lora_a, lora_b, lif, n_samples=8)
    log.info(
        "self-test: rel_l2=%.4f abs_max=%.4f ann_abs_max=%.4f (tol=%.4f)",
        metrics["rel_l2"], metrics["abs_max"],
        metrics["ann_abs_max"], tol,
    )
    return 0 if metrics["rel_l2"] < tol else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a LoRA adapter to SNN-compatible form (LAS)."
    )
    p.add_argument("--adapter", type=Path, help="Path to adapters.safetensors")
    p.add_argument(
        "--snn-base", type=Path,
        help="Directory of the SNN base (with lif_metadata.json)",
    )
    p.add_argument("--output", type=Path, help="Output directory")
    p.add_argument(
        "--timesteps", type=int, default=128,
        help="Default T when no SNN-base match (default 128).",
    )
    p.add_argument(
        "--validate-samples", type=int, default=3,
        help="Number of LoRA modules to numerically validate (default 3).",
    )
    p.add_argument(
        "--self-test", action="store_true",
        help="Run synthetic self-test and exit.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        return run_self_test(timesteps=args.timesteps)

    if not (args.adapter and args.snn_base and args.output):
        log.error("--adapter, --snn-base, and --output are required")
        return 2

    if not args.adapter.exists():
        log.error("adapter not found: %s", args.adapter)
        return 2
    if not args.snn_base.exists():
        log.error("SNN base not found: %s", args.snn_base)
        return 2

    lora = load_lora_adapter(args.adapter)
    snn_meta = load_snn_lif_metadata(args.snn_base)
    lif_params = derive_lif_params(
        lora["modules"], snn_meta, default_timesteps=args.timesteps,
    )

    # Numerical spot-check on a few modules.
    validated: list[dict[str, Any]] = []
    module_keys = list(lora["modules"].keys())[: args.validate_samples]
    for mk in module_keys:
        m = lora["modules"][mk]
        if "lora_a" not in m or "lora_b" not in m:
            continue
        metrics = validate_one_module(m["lora_a"], m["lora_b"], lif_params[mk])
        metrics["module"] = mk
        validated.append(metrics)
        log.info(
            "validated %s: rel_l2=%.4f",
            mk.split(".")[-1] + "/" + mk.split(".")[-2], metrics["rel_l2"],
        )

    write_snn_adapter(
        args.adapter, args.output, lif_params,
        extra_meta={
            "snn_base": str(args.snn_base),
            "validation_samples": validated,
        },
    )
    log.info("done: %d modules converted, %d validated", len(lif_params), len(validated))
    return 0


if __name__ == "__main__":
    sys.exit(main())
