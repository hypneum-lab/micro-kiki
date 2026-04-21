# SpikingKiki conversion runbook

Operator reference for the five-phase pipeline that turns
Qwen3.6-35B-A3B + V4 LoRA adapters into a rate-coded spiking model.
Phase A (prep / dry-run) is already landed in this commit. Phase C
(base conversion, ~40 h) and phase D (adapter conversion) are
launched manually after V4 training completes.

## 1. Purpose

Convert the dense / MoE ANN base `Qwen3.6-35B-A3B` into a Lossless
ANN→SNN ("LAS") rate-coded equivalent — `SpikingKiki-35B-A3B` —
and port all 35 V4 LoRA adapters onto the new base. Target outcomes:

- Functional parity with the ANN base on the standard eval set
  (<2% win-rate drop, <1% relative L2 on residual stream).
- ~99% CMOS / ~100% neuromorphic energy savings per
  `run_analysis` (stored at `results/spikingkiki-35b-analysis.json`).
- 35 adapters fused at inference time without re-training any
  of them.

## 2. Prerequisites

- **Machine**: Mac Studio M3 Ultra 512 GB. Nothing else has the RAM
  to materialise the 94-layer MoE state dict during conversion.
- **Venv**: `/Users/clems/KIKI-Mac_tunner/.venv` (MLX + torch +
  transformers + safetensors). `uv pip sync` if missing.
- **Base model**: `/Users/clems/KIKI-Mac_tunner/models/Qwen3.6-35B-A3B/`
  (HF format, bf16 safetensors, 12 shards).
- **V4 adapters**: `/Users/clems/KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b-v4-sota/`
  one subdir per domain, each containing `adapters.safetensors` +
  `adapter_config.json` (MLX-LM convention: `lora_a` / `lora_b` keys).
- **Reference SNN (prototype)**: `/Users/clems/models/spikingkiki-27b/`
  — older 27B dense conversion, used in phase 2 as a sanity check
  for the LoRA-on-SNN pipeline.
- **V4 training must be complete**: conversion loads the full base
  at `torch_dtype=bfloat16` on CPU (~70 GB). Do not launch while
  `PID 31745` or any MLX trainer is active.

## 3. Phase C — base conversion (40 h, overnight)

Exact SSH invocation from `kxkm-ai`:

```bash
ssh grosmac 'ssh studio tmux new-session -d -s spiking-c "\
  cd /Users/clems/KIKI-Mac_tunner && \
  .venv/bin/python /path/to/micro-kiki/scripts/convert_spikingkiki_35b.py \
    --input  /Users/clems/KIKI-Mac_tunner/models/Qwen3.6-35B-A3B \
    --output /Users/clems/models/SpikingKiki-35B-A3B \
    --config /Users/clems/KIKI-Mac_tunner/models/Qwen3.6-35B-A3B/config.json \
    --timesteps 128 \
    --resume \
    2>&1 | tee /Users/clems/logs/spikingkiki-35b-convert.log"'
```

- `--config` accepts either a flat (Qwen3.5) or nested `text_config`
  (Qwen3.6 multimodal) layout — see
  `scripts/convert_spikingkiki_35b.py::load_model_config`.
- `--resume` reads `.convert_state.json` from the output dir and
  skips already-converted layers. Safe to re-invoke after any SIGINT.
- `--timesteps 128` is the default; lower T trades accuracy for
  speed (`spikingkiki-27b` used T=16).
- Expected wall time: ~40 h on M3 Ultra 512 GB. The bottleneck is
  bf16 load + per-expert `_extract_linear_weights` over 30 720 FFN
  modules (= 40 layers × 256 experts × 3 projections).
- Dry-run (no weights needed):
  `.venv/bin/python scripts/convert_spikingkiki_35b.py --dry-run \
    --config <path>` prints the layer map counts and writes
  `results/spikingkiki-35b-convert.json`.

### Output layout

```
SpikingKiki-35B-A3B/
  model_layers_0_self_attn_q_proj.npz   # SpikingLinear weight dump
  model_layers_0_self_attn_k_proj.npz
  ...
  .convert_state.json                   # resume pointer
```

Each `.npz` stores `weight`, `bias`, `timesteps`, `max_rate`.

## 4. Phase D — adapter conversion (minutes each)

Use `scripts/convert_lora_to_snn.py`. Per-adapter wall time ~O(seconds)
because LAS is metadata-only — the adapter weights themselves are
untouched, only `lif_metadata.json` changes.

```bash
for domain in yaml-json platformio music-audio ...; do
  ssh grosmac 'ssh studio bash -s' <<EOF
.venv/bin/python /path/to/micro-kiki/scripts/convert_lora_to_snn.py \
  --adapter  /Users/clems/KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b-v4-sota/${domain}/adapters.safetensors \
  --snn-base /Users/clems/models/SpikingKiki-35B-A3B \
  --output   /Users/clems/models/spikingkiki-lora-v4/${domain} \
  --timesteps 128
EOF
done
```

The script copies the safetensors verbatim and writes a
`lif_metadata.json` that maps each LoRA linear to its inherited
threshold / tau / timesteps (keyed by matching against the SNN base's
`lif_metadata.json`).

**Validation**: the script runs `validate_one_module` on the first
`--validate-samples` (default 3) LoRA modules and logs
`rel_l2 = ||y_snn − y_ann|| / ||y_ann||`. Expected < 0.1 with T=128
in the LAS-valid unipolar regime.

## 5. Phase E — validation

After both base and at least one adapter are converted:

```bash
.venv/bin/python -m pytest tests/spiking/ -v
```

Plus an end-to-end generation compare. Gate thresholds (one adapter,
one held-out prompt set):

| Metric | Threshold | Rationale |
|---|---|---|
| `verify_equivalence` rel L2 | < 0.02 on attention | Rate-code quantisation bound `O(1/T)` |
| Win-rate vs ANN baseline | drop < 0.03 | Forgetting gate parity with the main pipeline |
| Perplexity lift | < 5 % | Tolerable for T=128; revisit if larger |

Failing any gate = rollback the conversion (delete output dir and
re-run with larger T or signed two-channel encoding).

## 6. Rollback / resume

- `--resume` is the primary recovery path. Re-invoking Phase C with
  the same `--output` will skip any layer whose `key_prefix` is
  present in `.convert_state.json`.
- Partial layer recovery: if a specific `.npz` is corrupt
  (e.g. OOM during write), delete **both** the `.npz` and its
  entry from `.convert_state.json["converted_keys"]`, then re-run
  with `--resume`.
- To start fresh: `rm -rf /Users/clems/models/SpikingKiki-35B-A3B`
  and re-launch without `--resume`.
- Phase D adapters are stateless — just delete the output dir and
  re-run.

## 7. Known gotchas

- **Qwen3.6 config is nested**: `config.json` wraps the LM config
  under `"text_config"`. `load_model_config` flattens it
  transparently; pass the raw config path via `--config`.
- **Hybrid attention**: Qwen3.6 alternates `linear_attention` and
  `full_attention` (3:1 ratio, `full_attention_interval=4`). The
  current `build_layer_map` only emits the standard
  `self_attn.{q,k,v,o}_proj` key prefixes — this is an issue for
  the linear-attention layers which use
  `linear_attn.in_proj_{qkv,z,b,a}` + `linear_attn.out_proj`.
  Phase C will need a second patch to emit the correct key prefixes
  per-layer based on `config["layer_types"]`. Tracked as
  **TODO-phase-C1**.
- **LAS is unipolar**: negative activations are clipped to zero
  inside `rate_encode` (see `src/spiking/lif_neuron.py`). For signed
  LoRA deltas the correct answer is two-channel encoding
  (story 21+). The phase-D prototype documents this but defers the
  fix — the numerical validation runs in the unipolar regime.
- **spikingkiki-27b mismatch**: that base is Qwen3.5-27B
  (hidden=5120, 64 layers, dense MLP). It does **not** share the
  architecture of Qwen3.6-35B-A3B (hidden=2048, 40 layers, 256
  experts). Running `convert_lora_to_snn.py` with the V4 adapters
  against `spikingkiki-27b` exercises the **pipeline** (read /
  validate / write) only; actual runtime fusion requires the
  phase-C output.
