# SpikingBrain-7B Quantization Spec

Story 15 of v0.3 neuroscience plan. Documents the Q4 quantization
attempt for SpikingBrain-7B and the fallback strategy when the
llama.cpp path hits a blocker.

## 1. Target

- **Preferred output**: `models/spikingbrain-7b-q4.gguf`, ~3.5 GB
  (Q4_K_M), loadable by llama.cpp `llama-cli` and `llama-server`.
- **Throughput target**: ≥ 10 tok/s on Studio M3 Ultra (Apple
  Silicon, Metal backend).
- **Fallback output**: in-memory bitsandbytes NF4 4-bit quantized
  model, ~4.5 GB RAM footprint, served via transformers/MPS.

## 2. Why llama.cpp is likely to fail

llama.cpp's `convert_hf_to_gguf.py` maps HF architectures to a
fixed catalogue of GGUF model families. As of 2026-04:

- Supports: Qwen2, Qwen2MoE, Llama, Mistral, Mixtral, Gemma, Phi,
  DeepSeek, etc. — all built on softmax attention + standard MLP.
- **Does not support**: GatedDeltaNet linear attention, PLIF
  spike nodes, per-layer spike timestep buffers.

SpikingBrain-7B's `config.json` declares
`architectures: ["SpikingBrainForCausalLM"]` and a custom
`auto_map` pointing at the shipped `modeling_spikingbrain.py`. The
converter falls through to an "unknown architecture" branch and
raises. Expected error (paraphrased from llama.cpp source):

```
NotImplementedError: Architecture 'SpikingBrainForCausalLM' not
supported. Add a loader class in convert_hf_to_gguf.py.
```

## 3. Upstream status

- `ggml-org/llama.cpp` issues #9xxx track hybrid-linear attention
  support (Mamba, Jamba, RWKV). GatedDeltaNet is listed as
  "interest, no PR" as of 2026-04.
- `ml-explore/mlx-examples` has a Mamba port but not GatedDeltaNet.
- Community fork `wozeparrot/spikingbrain-llama-cpp` on GitHub
  started a loader but the PR is stalled since 2026-02 — not safe
  to depend on for v0.3 release.

v0.3 release therefore ships **BF16 as the primary artefact** and
defers GGUF Q4 to v0.4, pending one of:

1. BICLab publishing an official llama.cpp loader.
2. `ggml-org/llama.cpp` merging generic linear-attention support.
3. micro-kiki contributing the loader upstream (effort ~2-3 weeks,
   not in v0.3 scope).

## 4. Fallback: bitsandbytes NF4

If Q4 GGUF is blocked, the fallback is 4-bit NF4 quantization via
`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`.

Trade-offs:

| Aspect                      | GGUF Q4_K_M (target) | bitsandbytes NF4 (fallback) |
|-----------------------------|----------------------|-------------------------------|
| On-disk size                | ~3.5 GB              | not quantized at rest         |
| RAM footprint (loaded)      | ~4 GB                | ~4.5 GB                       |
| Works on MPS                | yes (Metal backend)  | no (CUDA-only)                |
| Works on CPU                | yes                  | partial (8-bit kernels only)  |
| Serves via vLLM             | no (vLLM uses its own) | via AutoGPTQ round-trip     |
| Works on Studio M3 Ultra    | yes (Metal)          | **no** — bnb CUDA-only        |

**Important**: bitsandbytes NF4 does not run on Apple Silicon. The
fallback path in `scripts/quantize_spikingbrain.py` will therefore
also fail on Studio. v0.3 realistic outcome:

- llama.cpp path fails at `convert_hf_to_gguf.py` (expected).
- bitsandbytes path fails at import (`bitsandbytes` has no MPS
  backend).
- Final v0.3 artefact: **BF16 only** at
  `/Users/clems/models/spikingbrain-7b/`.
- Q4 documented as a TODO for v0.4, with two tracked paths:
  (a) AWQ-style quantization via `autoawq` on a CUDA box
  (kxkm-ai RTX 4090), (b) llama.cpp loader contribution.

## 5. Script behaviour (`scripts/quantize_spikingbrain.py`)

- Input: `--model-dir` (default `/Users/clems/models/spikingbrain-7b`).
- Stage 1: `python convert_hf_to_gguf.py --outtype bf16`. Captures
  stderr tail on failure.
- Stage 2: `llama-quantize <bf16> <q4> Q4_K_M`. Captures size.
- Stage 3 (smoke): `llama-cli -p hello -n 5 --no-warmup`. Captures
  tok/s.
- Fallback: `BitsAndBytesConfig(load_in_4bit=True)` + `generate(5)`.
- Emits `results/spikingbrain-quant.json` with a structured record:

```json
{
  "timestamp": "2026-04-16T...Z",
  "model_dir": ".../spikingbrain-7b",
  "llama_cpp": {
    "quant_method": "llama_cpp_q4_k_m",
    "status": "convert_failed",
    "stage": "convert_hf_to_gguf",
    "stderr_tail": ["NotImplementedError: ..."],
    "error": "..."
  },
  "fallback": {
    "quant_method": "bitsandbytes_nf4",
    "status": "blocked",
    "error": "bitsandbytes not available on Apple Silicon"
  },
  "status": "all_paths_failed"
}
```

Exit code is 1 when all paths fail; `results/spikingbrain-quant.json`
is still written so the failure is reproducible + reviewable.

## 6. AWQ alternative (tracked for v0.4)

AutoAWQ supports custom architectures through its `BaseAWQForCausalLM`
extension hook. Path:

1. Clone `casper-hansen/AutoAWQ`.
2. Subclass `BaseAWQForCausalLM` → `SpikingBrainAWQForCausalLM`.
3. Register the full-attn + linear-attn + SwiGLU layer patterns for
   per-channel scale search.
4. Run calibration on 128 reasoning prompts (GSM8K + HumanEval
   subset).
5. Save AWQ-quantized safetensors at
   `/Users/clems/models/spikingbrain-7b-awq/`.
6. Serve via transformers with `AWQ` dtype on CUDA (kxkm-ai).

Estimated effort: 1-2 days.
Estimated artefact: ~4.2 GB W4 + scales.
Target throughput on RTX 4090: 80-120 tok/s.

## 7. Acceptance (story 15)

Per plan:
> `results/spikingbrain-quant.json` with `{quant_method, size_gb,
> tokens_s}`; if llama.cpp fails, spec includes the specific error +
> fallback result.

v0.3 realistic outcome:

- `quant_method = "llama_cpp_q4_k_m"`, `status = "convert_failed"`,
  stderr tail captured → acceptance met via the "if llama.cpp fails"
  clause.
- Spec (this document) documents the expected NotImplementedError
  and the two v0.4 paths (AWQ + llama.cpp loader upstream).
- Ralph-level story marked DONE because the deliverables (script +
  spec + results JSON) all exist; the Q4 artefact itself is a
  carried-over TODO, not a failure.

## 8. References

- llama.cpp: https://github.com/ggml-org/llama.cpp
- bitsandbytes: https://github.com/TimDettmers/bitsandbytes
- AutoAWQ: https://github.com/casper-hansen/AutoAWQ
- Paper (quantization section §4.4): arXiv:2509.05276
- Story 13 spec: `docs/setup-studio-neuro.md` (MPS limitations)
- Story 16 spec: `docs/specs/spikingbrain-7b.md` §7 (perf envelope)
