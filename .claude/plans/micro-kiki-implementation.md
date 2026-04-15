# micro-kiki — Implementation Plan

**Scope**: Build 32 domain-expert MoE-LoRA stacks on Qwen3.5-4B base, with sigmoid meta-router, end-to-end distillation pipeline, and triple-device serving (RTX 4090 + Mac Studio + optional ANE).

**Derived from**: `docs/specs/2026-04-15-micro-kiki-design.md` + `docs/research/micro-kiki-moe-research.md`.

**Conventions**:
- Each step is ONE shippable unit of work — can be committed independently.
- Steps run sequentially unless marked `(parallel)`.
- Acceptance = what "done" looks like (tests / eval score / artifact exists).
- Dependencies = which prior step MUST be complete first.

---

## Implementation Steps

### Phase I — Foundations (bootstrap)

1. **Download + verify Qwen3.5-4B base**
   - Files to touch: `scripts/download_base.py`, `models/qwen3.5-4b/` (gitignored)
   - Download from HuggingFace `Qwen/Qwen3.5-4B` (or mirror), verify SHA256, save BF16 safetensors at `models/qwen3.5-4b/bf16/`.
   - Also produce Q4_K_M GGUF at `models/qwen3.5-4b-q4.gguf` via `llama.cpp/convert_hf_to_gguf.py` + `llama-quantize`.
   - Acceptance: both files exist, loader smoke test passes (prompt "hello" → non-empty response), file sizes within ±5% of expected (~8 GB BF16, ~2.5 GB Q4).
   - Dependencies: none.

2. **Write base model loader (`src/base/loader.py`)**
   - File to touch: `src/base/loader.py`, `src/base/__init__.py`, `tests/test_loader.py`
   - Class `BaseModelLoader` with methods `load_bf16()`, `load_q4()`, `enable_lora_switching()` (Unsloth + PEFT hooks).
   - Expose context manager `with_stack(adapter_name)` that hot-swaps the LoRA weights.
   - Acceptance: `pytest tests/test_loader.py::test_load_and_switch` passes.
   - Dependencies: step 1.

3. **Write teacher client (`src/distill/teacher_client.py`)**
   - File to touch: `src/distill/teacher_client.py`, `src/distill/__init__.py`, `tests/test_teacher_client.py`
   - OpenAI-compatible client wrapping: Mistral-Large-Opus (Studio mlx-lm server), Qwen3.5-122B-A10B Opus-v3 (Studio), Qwen3.5-35B-A3B Opus (kxkm-ai :8000), Devstral-v3/v4 (kxkm-ai).
   - Features: async HTTP via httpx, exponential backoff retry (3 attempts), on-disk response cache keyed by SHA256(prompt + model + params), Qwen3 thinking-mode toggle (`enable_thinking=False` for scoring).
   - Endpoints configurable via env vars `TEACHER_MISTRAL_URL`, `TEACHER_QWEN122_URL`, etc.
   - Acceptance: `pytest tests/test_teacher_client.py` — mocks HTTP, verifies cache hit, verifies retry on 500.
   - Dependencies: none.

4. **Smoke-test harness (`tests/conftest.py`)**
   - File to touch: `tests/conftest.py`, `tests/test_smoke.py`
   - Pytest fixtures: `tmp_model_dir`, `mock_teacher`, `sample_prompts`.
   - Smoke tests: 1 per module (`test_loader_smoke`, `test_teacher_smoke`).
   - Acceptance: `uv run pytest` runs cleanly, all smoke tests green.
   - Dependencies: steps 2, 3.

### Phase II — Data pipeline

5. **Distilled dataset generator (`src/distill/generator.py`)**
   - File to touch: `src/distill/generator.py`, `tests/test_generator.py`
   - Function `generate_examples(prompts: list[str], teacher: TeacherClient, n_per_prompt: int = 1) -> Dataset`.
   - Emits `jsonl` with `{prompt, completion, teacher_model, domain, hash}` rows.
   - Supports resume from checkpoint (scan existing jsonl, skip already-done hashes).
   - Acceptance: generates 10 examples from mock teacher, output jsonl valid, resume test passes.
   - Dependencies: step 3.

6. **Cross-domain dedup (`src/distill/dedup.py`)**
   - File to touch: `src/distill/dedup.py`, `tests/test_dedup.py`
   - MinHash + LSH across ALL domain jsonls, flag examples appearing in > 1 domain, assign to highest-affinity domain only.
   - CLI: `python -m src.distill.dedup --input data/raw/ --output data/dedup/`
   - Acceptance: on synthetic dup set (3 domains, 30% overlap), produces disjoint partition; unit tests pass.
   - Dependencies: step 5.

7. **Seed prompt lists for first 3 domains**
   - File to touch: `data/prompts/chat-fr.jsonl`, `data/prompts/python.jsonl`, `data/prompts/reasoning.jsonl`
   - 300-500 seed prompts per domain. Sources:
     - `chat-fr`: mix of MTBench-FR, OpenAssistant-FR, manual curation.
     - `python`: CodeAlpaca + StackOverflow Python top-tagged questions.
     - `reasoning`: GSM8K trainset + ARC-easy subset (as prompts, not with answers).
   - Acceptance: 3 files exist, valid jsonl, each > 300 rows, French ratio ≥ 95% for chat-fr.
   - Dependencies: none.

8. **Generate first distilled dataset (chat-fr, 2K examples)**
   - File to touch: `data/distilled/chat-fr.jsonl`, `scripts/distill_chat_fr.py`
   - Teacher: Mistral-Large-Opus (Studio). Target: 2000 completed examples.
   - Command: `uv run scripts/distill_chat_fr.py --teacher mistral-large-opus --n 2000 --out data/distilled/chat-fr.jsonl`
   - Acceptance: `wc -l data/distilled/chat-fr.jsonl == 2000`, random-sample manual quality check (5/5 OK).
   - Dependencies: steps 5, 7, Studio teacher reachable.

### Phase III — First stack (chat-fr, prove E2E)

9. **MoE-LoRA stack trainer (`src/stacks/trainer.py`)**
   - File to touch: `src/stacks/trainer.py`, `src/stacks/moe_lora.py`, `tests/test_trainer.py`
   - MoLoRA-style: 4 LoRA experts per attention projection (q, k, v, o), rank 16, top-2 routing per token, softmax gate.
   - Uses HuggingFace `peft` + custom `MoLoraConfig`. Training via `trl.SFTTrainer`.
   - Hyperparams from YAML config. Saves adapters to `outputs/stacks/stack-XX-<name>/`.
   - Acceptance: `pytest tests/test_trainer.py::test_forward_pass` — 1 step with dummy data, loss finite.
   - Dependencies: step 2.

10. **First stack config (`configs/stack-01-chat-fr.yaml`)**
    - File to touch: `configs/stack-01-chat-fr.yaml`
    - Fields: `base_model`, `num_experts: 4`, `lora_rank: 16`, `lora_alpha: 32`, `top_k: 2`, `learning_rate: 2e-4`, `batch_size: 4`, `grad_accum: 8`, `epochs: 3`, `seq_len: 4096`, `dataset: data/distilled/chat-fr.jsonl`.
    - Acceptance: file loads via `yaml.safe_load` in test, required keys present.
    - Dependencies: step 9.

11. **Train stack-01 (chat-fr)**
    - File to touch: `outputs/stacks/stack-01-chat-fr/` (gitignored, artifact-only)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-01-chat-fr.yaml`
    - Target: ~30 min on 4090 or ~15 min on Studio MLX.
    - Acceptance: final train loss < 1.5, eval loss (held-out 5% split) < 1.8, adapter files saved (< 200 MB).
    - Dependencies: steps 8, 9, 10.

12. **Per-stack eval harness (`src/eval/stack_eval.py`)**
    - File to touch: `src/eval/stack_eval.py`, `data/eval/chat-fr.jsonl` (held-out 100 prompts + judge criteria)
    - Loads base + stack adapter, generates responses, LLM-judges with Mistral-Large-Opus (pairwise vs base-only).
    - Outputs score JSON: `{stack, n_prompts, win_rate_vs_base, avg_judge_score, sample_responses}`.
    - Acceptance: run on stack-01 produces valid JSON, win_rate > 0.55 (stack better than base).
    - Dependencies: step 11.

13. **Evaluate + commit stack-01 baseline**
    - File to touch: `results/stack-01-baseline.json`, update `README.md` status section
    - Run eval, archive results, log tokens/sec at inference, commit adapter metadata (not weights) to git.
    - Acceptance: `results/` dir exists, JSON committed, README `- [x] First stack trained`.
    - Dependencies: step 12.

### Phase IV — Meta-router (v0, 3 stacks)

14. **Train stacks 02 (reasoning) + 03 (python)**
    - File to touch: `configs/stack-02-reasoning.yaml`, `configs/stack-03-python.yaml`, then same train command
    - Reuse distill generator for each domain (400 examples each as starter — full 2-5K later).
    - Acceptance: stack-02 win_rate_vs_base > 0.55 on reasoning eval (GSM8K 50-sample), stack-03 > 0.55 on HumanEval 50-subset.
    - Dependencies: steps 11-13 pattern.

15. **Meta-router module (`src/routing/router.py`)**
    - File to touch: `src/routing/router.py`, `src/routing/__init__.py`, `tests/test_router.py`
    - Architecture: extract base model last-hidden-state at layer N (N=16 for 4B), pool (mean + attention-query), feed to Linear(3072→512) → LayerNorm → GELU → Linear(512→32) → sigmoid.
    - ~2M params total. Training target: multi-label outcome discovery (binary per-domain, threshold 0.12, chat floor 0.20).
    - Acceptance: `test_router_forward` — dummy batch, outputs shape [B, 32], all in [0,1]; `test_router_init` — trainable params < 3 M.
    - Dependencies: step 2 (base access).

16. **Train router v0 on 3-stack mix**
    - File to touch: `scripts/train_router.py`, `outputs/router/v0/router.pt`
    - Dataset: concat 3 domain jsonls, label each example with its domain-id; also generate 1K "mixed-intent" examples (queries that span 2 domains) for multi-label training.
    - Loss: BCEWithLogitsLoss, train 3 epochs, lr 1e-3.
    - Acceptance: macro-F1 ≥ 0.85 on 3-domain eval; router file < 10 MB.
    - Dependencies: steps 14, 15.

17. **Runtime adapter switcher (`src/serving/switchable.py`)**
    - File to touch: `src/serving/switchable.py`, `tests/test_switchable.py`
    - Class `SwitchableModel`: holds base, registry of loaded adapters, method `apply_stacks(names: list[str])` — merges selected LoRAs (weighted sum or sequential application).
    - Cap active stacks at 4. Cache merged weights until stack-set changes.
    - Acceptance: switches between stack-01 ↔ stack-03 ↔ none in < 500 ms; generation before/after differs measurably.
    - Dependencies: steps 11, 14.

18. **End-to-end smoke test**
    - File to touch: `tests/test_e2e_3stacks.py`, `scripts/demo_e2e.py`
    - Prompt → router → top-2 sigmoid scores → SwitchableModel.apply_stacks → generate → print.
    - Test cases: "écris un haïku" → chat-fr dominant; "factor 231 into primes" → reasoning dominant; "python fastapi hello world" → python dominant.
    - Acceptance: all 3 test cases have expected domain in top-2 router picks; generation coherent (no garbled tokens).
    - Dependencies: steps 16, 17.

### Phase V — Curriculum expansion (stacks 4-16)

19. **Train stacks 4-16 (coding core + secondary)**
    - Files to touch: `configs/stack-04-*.yaml` … `configs/stack-16-*.yaml`, `data/distilled/<domain>.jsonl` per step
    - Domains (per design doc §curriculum): 04 js/ts, 05 rust, 06 go, 07 c-cpp, 08 java/kotlin, 09 swift, 10 sql, 11 shell-bash, 12 dockerfile, 13 yaml-config, 14 regex, 15 git-workflows, 16 testing-patterns.
    - 13 sequential sub-steps; each = distill ~1.5K examples + train + eval. Budget ~45 min each.
    - Acceptance per stack: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks (see step 21).
    - Dependencies: steps 11-13 pattern for each.

20. **Re-train router every 4 stacks**
    - File to touch: `outputs/router/v1/` (after stack 08), `v2/` (after 12), `v3/` (after 16)
    - Re-run step 16's script with new domain-id list.
    - Acceptance: macro-F1 on all-N-stacks eval ≥ 0.80; router still < 10 MB.
    - Dependencies: step 19.

21. **Forgetting check after each stack (`src/eval/forgetting.py`)**
    - File to touch: `src/eval/forgetting.py`, `scripts/run_forgetting.sh`
    - For each already-trained stack, run its eval set with the NEW stack loaded (not active) and with active-swap test.
    - Flag delta > 0.03 in win_rate_vs_base.
    - Acceptance: script runs automatically after each training; emits `results/forgetting-stack-XX.json`; no stack drops > 3%.
    - Dependencies: step 12.

### Phase VI — Technical domains (stacks 17-27)

22. **Train stacks 17-27 (embedded + hardware)**
    - Configs + data: 17 embedded-rtos, 18 stm32, 19 iot-protocols (MQTT/CoAP), 20 freecad, 21 platformio, 22 power-electronics, 23 emc-rules, 24 dsp, 25 spice-netlists, 26 analog-electronics, 27 kicad-pcb.
    - Pull seed prompts from existing `KIKI-models-tuning` datasets where available (kiki-embedded, kiki-stm32, etc.).
    - Acceptance: 11 sub-steps, each passing per-stack eval (win_rate > 0.55).
    - Dependencies: step 19 pattern + external data sources.

23. **Curate domain-specific datasets from kiki-* legacy data**
    - File to touch: `scripts/import_kiki_datasets.py`
    - Read existing LoRA training jsonls from `~/Documents/Projets/Factory 4 Life/` (if present) or GitHub `KIKI-models-tuning`, remap to micro-kiki format, run through dedup step 6.
    - Acceptance: for each of domains 17-27, ≥ 800 distilled examples in `data/distilled/`.
    - Dependencies: step 6.

24. **Group eval + forgetting check (every 4 stacks)**
    - File to touch: `results/group-eval-after-XX.json`
    - After stacks 20, 24, 27: re-run router eval + forgetting check across ALL stacks so far.
    - Acceptance: global macro-F1 ≥ 0.75; no stack-forgetting > 3%.
    - Dependencies: steps 20, 21, 22.

### Phase VII — Applications + complements (stacks 28-32)

25. **Train stacks 28-32 (apps + misc)**
    - Configs + data: 28 web-frontend (React/Vue/Svelte), 29 web-backend (FastAPI/Express/Django), 30 music-audio (DAW/DSP/MIDI), 31 devops (k8s/terraform/ansible), 32 llm-orch (LangChain/LlamaIndex/MCP).
    - Note: design doc §complements mentions math + security as candidates — if prioritized, swap into 31/32 slots.
    - Acceptance: 5 sub-steps, each passing per-stack eval.
    - Dependencies: step 22 pattern.

26. **Final meta-router retrain (v-final, 32 outputs)**
    - File to touch: `outputs/router/v-final/router.pt`, `results/router-final.json`
    - Full 32-domain training set; include 3K cross-domain multi-intent examples.
    - Acceptance: macro-F1 ≥ 0.82 on held-out 32-class eval; p95 latency < 8 ms on RTX 4090.
    - Dependencies: step 25.

27. **Full evaluation suite**
    - File to touch: `scripts/run_full_eval.sh`, `results/full-eval-v0.1.json`
    - Run: HumanEval (+ pass@1), GSM8K, MMLU-Pro sample, 32 custom domain evals, latency benchmarks.
    - Acceptance: consolidated JSON report; compare against base-Qwen3.5-4B numbers; each domain shows ≥ 0.55 win rate; aggregate score ≥ +15% over base.
    - Dependencies: step 26.

### Phase VIII — Serving deployment

28. **vLLM server with dynamic LoRA (`src/serving/vllm_server.py`)**
    - File to touch: `src/serving/vllm_server.py`, `docker/vllm.Dockerfile`
    - vLLM 0.6+ with `--enable-lora --max-loras 4 --max-lora-rank 16`; expose OpenAI-compatible endpoint on :8100.
    - Router runs as a FastAPI sidecar on :8101 that rewrites incoming request with `lora_request` field.
    - Acceptance: `curl :8100/v1/chat/completions` with "write python factorial" → correct response, logs show stacks 03 + 02 active.
    - Dependencies: step 26.

29. **mlx-lm server for Mac Studio (`src/serving/mlx_server.py`)**
    - File to touch: `src/serving/mlx_server.py`, `configs/mlx-server.json`
    - mlx-lm with adapter switching via subprocess restart (mlx doesn't support hot-swap cleanly — accept ~200ms penalty per switch) OR via `mlx_lm.server --adapter-path` per-request.
    - Acceptance: runs on Studio, serves 32 stacks (max 4 active), p95 first-token latency < 500 ms.
    - Dependencies: step 26.

30. **Persistent service unit**
    - File to touch: `deploy/launchd/cc.saillant.micro-kiki.plist` (Mac), `deploy/systemd/micro-kiki.service` (Linux)
    - Launchd plist for Studio, systemd for kxkm-ai. Autostart on boot, restart on crash.
    - Acceptance: `launchctl load` + `sudo systemctl enable --now micro-kiki` both work; reboot test passes.
    - Dependencies: steps 28, 29.

31. **Integration test (router HTTP)**
    - File to touch: `tests/test_integration_http.py`
    - Spawn router sidecar + vLLM (or mlx-server) in pytest fixture, send 20 queries spanning 10 domains, assert each hits correct top-2 stacks.
    - Acceptance: 20/20 queries routed correctly; test completes in < 60 s.
    - Dependencies: step 28 OR step 29.

### Phase IX — ANE triple pipeline (Mac-only, optional)

32. **ANE draft model (`src/serving/ane_draft.py`)**
    - File to touch: `src/serving/ane_draft.py`, `models/qwen3.5-0.8b.mlpackage`
    - Convert Qwen3.5-0.8B to CoreML (ANE-targeted), use as speculative decoder for the 4B base running on GPU (Metal Performance Shaders).
    - Target: 2-3× decode speedup on M3 Ultra.
    - Acceptance: benchmark before/after on 100-token generation, speedup ≥ 1.8×.
    - Dependencies: step 29.

33. **ANE scorer for GRPO (`src/serving/ane_scorer.py`)**
    - File to touch: `src/serving/ane_scorer.py`
    - CoreML-compiled scorer (reward model) running on ANE for RL fine-tuning pipelines (future GRPO stacks).
    - Acceptance: scores 100 prompt-completion pairs in < 2 s on ANE; scores match GPU-ref within ±0.02.
    - Dependencies: step 32 tooling.

34. **ANE-resident meta-router (`src/serving/ane_router.py`)**
    - File to touch: `src/serving/ane_router.py`
    - Compile the 2M-param router to CoreML, run on ANE (not CPU) to free CPU for decode orchestration.
    - Acceptance: latency drops from ~5 ms (CPU) to ~1 ms (ANE); identical outputs within fp16 tolerance.
    - Dependencies: step 26.

### Phase X — Release 0.1

35. **Freeze config + migration guide**
    - File to touch: `MIGRATION.md`, `VERSION`, tag `v0.1.0`
    - Document breaking changes, adapter format, router protocol. Freeze `configs/` (hash-lock).
    - Acceptance: `git tag v0.1.0` pushed; MIGRATION.md reviewed.
    - Dependencies: step 31.

36. **HuggingFace release**
    - File to touch: `.github/workflows/hf-release.yml`, HF repo `electron-rare/micro-kiki-v0.1`
    - Upload: base-model reference (link to Qwen), 32 adapter .safetensors, router.pt, model card, eval results.
    - Acceptance: `huggingface-cli download electron-rare/micro-kiki-v0.1` works from clean env; model card renders.
    - Dependencies: step 35.

37. **Model card + cookbook**
    - File to touch: `MODEL_CARD.md`, `COOKBOOK.md`, `examples/` dir
    - Document: intended use, limitations, training data, eval scores, 5+ runnable examples (chat-fr, code, SPICE, KiCad, reasoning).
    - Acceptance: all examples in COOKBOOK run green in CI; model card passes HF lint.
    - Dependencies: step 36.

---

## Success criteria (v0.1)

- 32 stacks trained, each ≥ 0.55 win-rate vs base on its domain.
- Meta-router macro-F1 ≥ 0.82 across 32 domains.
- Serving stack runs on RTX 4090 24 GB (max 4 active stacks) AND on Mac Studio.
- Aggregate eval score ≥ +15% over base Qwen3.5-4B.
- Released on HuggingFace with model card + cookbook.

## Risks + mitigations

- **Forgetting across stacks** → step 21 forgetting check after EACH stack; rollback if > 3%.
- **Router saturation at 32 domains** → step 20 periodic retraining; fallback to hierarchical router (6 phase-groups → intra-group classifier).
- **VRAM overrun** → hard cap 4 active stacks, Q4 base, KV cache eviction.
- **Teacher unavailability** → step 3 caches responses; fallback to Qwen3.5-35B (kxkm-ai) if Studio offline.
