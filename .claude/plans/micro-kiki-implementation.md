# micro-kiki — Implementation Plan

**Scope**: Build 32 domain-expert MoE-LoRA stacks on Qwen3.5-4B base, with sigmoid meta-router, end-to-end distillation pipeline, and triple-device serving (RTX 4090 + Mac Studio + optional ANE).

**Derived from**: `docs/specs/2026-04-15-micro-kiki-design.md` + `docs/research/micro-kiki-moe-research.md`.

**Conventions**:
- Each step is ONE shippable unit of work — can be committed independently.
- Steps run sequentially unless marked `(parallel)`.
- Acceptance = what "done" looks like (tests / eval score / artifact exists).
- Dependencies = which prior step MUST be complete first.
- **Domain order locked** per design-doc curriculum: 1 chat-fr, 2 reasoning, 3 python, 4 typescript, 5 cpp, 6 rust, 7 html-css, 8 shell, 9 sql, 10 yaml-json, 11 docker, 12 kicad-dsl, 13 spice, 14 lua-upy, 15 embedded, 16 stm32, 17 iot, 18 freecad, 19 platformio, 20 power, 21 emc, 22 dsp, 23 spice-sim, 24 electronics, 25 kicad-pcb, 26 web-frontend, 27 web-backend, 28 music-audio, 29 devops, 30 llm-orch, 31 math, 32 security.

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
   - Files to touch: `src/base/loader.py`, `src/base/__init__.py`, `tests/test_loader.py`
   - Class `BaseModelLoader` with methods `load_bf16()`, `load_q4()`, `enable_lora_switching()` (Unsloth + PEFT hooks).
   - Expose context manager `with_stack(adapter_name)` that hot-swaps the LoRA weights.
   - Acceptance: `pytest tests/test_loader.py::test_load_and_switch` passes.
   - Dependencies: step 1.

3. **Write teacher client (`src/distill/teacher_client.py`)**
   - Files to touch: `src/distill/teacher_client.py`, `src/distill/__init__.py`, `tests/test_teacher_client.py`
   - OpenAI-compatible client wrapping: Mistral-Large-Opus (Studio mlx-lm server), Qwen3.5-122B-A10B Opus-v3 (Studio), Qwen3.5-35B-A3B Opus (kxkm-ai :8000), Devstral-v3/v4 (kxkm-ai).
   - Features: async HTTP via httpx, exponential backoff retry (3 attempts), on-disk response cache keyed by SHA256(prompt + model + params), Qwen3 thinking-mode toggle (`enable_thinking=False` for scoring).
   - Endpoints configurable via env vars `TEACHER_MISTRAL_URL`, `TEACHER_QWEN122_URL`, etc.
   - Acceptance: `pytest tests/test_teacher_client.py` — mocks HTTP, verifies cache hit, verifies retry on 500.
   - Dependencies: none.

4. **Smoke-test harness (`tests/conftest.py`)**
   - Files to touch: `tests/conftest.py`, `tests/test_smoke.py`
   - Pytest fixtures: `tmp_model_dir`, `mock_teacher`, `sample_prompts`.
   - Smoke tests: 1 per module (`test_loader_smoke`, `test_teacher_smoke`).
   - Acceptance: `uv run pytest` runs cleanly, all smoke tests green.
   - Dependencies: steps 2, 3.

### Phase II — Data pipeline

5. **Distilled dataset generator (`src/distill/generator.py`)**
   - Files to touch: `src/distill/generator.py`, `tests/test_generator.py`
   - Function `generate_examples(prompts: list[str], teacher: TeacherClient, n_per_prompt: int = 1) -> Dataset`.
   - Emits `jsonl` with `{prompt, completion, teacher_model, domain, hash}` rows.
   - Supports resume from checkpoint (scan existing jsonl, skip already-done hashes).
   - Acceptance: generates 10 examples from mock teacher, output jsonl valid, resume test passes.
   - Dependencies: step 3.

6. **Cross-domain dedup (`src/distill/dedup.py`)**
   - Files to touch: `src/distill/dedup.py`, `tests/test_dedup.py`
   - MinHash + LSH across ALL domain jsonls, flag examples appearing in > 1 domain, assign to highest-affinity domain only.
   - CLI: `python -m src.distill.dedup --input data/raw/ --output data/dedup/`
   - Acceptance: on synthetic dup set (3 domains, 30% overlap), produces disjoint partition; unit tests pass.
   - Dependencies: step 5.

7. **Audit data sources — write `docs/data-sources.md` (no download)**
   - Files to touch: `docs/data-sources.md`
   - For each of the 32 domains (in design-doc order), add a table row listing: (a) candidate local datasets under `~/Documents/Projets/Factory 4 Life/` (especially `KIKI-models-tuning/`), (b) existing HuggingFace datasets worth considering, (c) an `availability` column checked via `ls` / `find` (CONFIRMED / TBD / GAP), (d) any notes on licensing or language.
   - NO download or fetch from HF. This is an inventory + planning step only.
   - Full curation of the `kiki-*` legacy datasets (actual import + dedup) is deferred to step 35.
   - Acceptance: `docs/data-sources.md` exists with a header row and 32 data rows (one per domain); ≥ 70% of domains (≥ 23) have a CONFIRMED source; each GAP row has a one-line mitigation note (synthetic via teacher, scraping, manual seed, etc.).
   - Dependencies: none.

8. **Generate first distilled dataset (chat-fr, 2K examples)**
   - Files to touch: `data/prompts/chat-fr.jsonl`, `data/distilled/chat-fr.jsonl`, `scripts/distill_chat_fr.py`
   - Seed prompts (300–500) curated from the sources identified in step 7 for domain 1 (MTBench-FR, OpenAssistant-FR, manual).
   - Teacher: Mistral-Large-Opus (Studio). Target: 2000 completed examples.
   - Command: `uv run scripts/distill_chat_fr.py --teacher mistral-large-opus --n 2000 --out data/distilled/chat-fr.jsonl`
   - Acceptance: `wc -l data/distilled/chat-fr.jsonl == 2000`, random-sample manual quality check (5/5 OK), French ratio ≥ 95%.
   - Dependencies: steps 5, 7, Studio teacher reachable.

### Phase III — First stack (stack-01 chat-fr, prove E2E)

9. **MoE-LoRA stack trainer (`src/stacks/trainer.py`)**
   - Files to touch: `src/stacks/trainer.py`, `src/stacks/moe_lora.py`, `tests/test_trainer.py`
   - MoLoRA-style: 4 LoRA experts per attention projection (q, k, v, o), rank 16, top-2 routing per token, softmax gate.
   - Uses HuggingFace `peft` + custom `MoLoraConfig`. Training via `trl.SFTTrainer`.
   - Hyperparams from YAML config. Saves adapters to `outputs/stacks/stack-XX-<name>/`.
   - Acceptance: `pytest tests/test_trainer.py::test_forward_pass` — 1 step with dummy data, loss finite.
   - Dependencies: step 2.

10. **First stack config (`configs/stack-01-chat-fr.yaml`)**
    - Files to touch: `configs/stack-01-chat-fr.yaml`
    - Fields: `base_model`, `num_experts: 4`, `lora_rank: 16`, `lora_alpha: 32`, `top_k: 2`, `learning_rate: 2e-4`, `batch_size: 4`, `grad_accum: 8`, `epochs: 3`, `seq_len: 4096`, `dataset: data/distilled/chat-fr.jsonl`.
    - Acceptance: file loads via `yaml.safe_load` in test, required keys present.
    - Dependencies: step 9.

11. **Train stack-01 (chat-fr)**
    - Files to touch: `outputs/stacks/stack-01-chat-fr/` (gitignored, artifact-only)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-01-chat-fr.yaml`
    - Target: ~30 min on 4090 or ~15 min on Studio MLX.
    - Acceptance: final train loss < 1.5, eval loss (held-out 5% split) < 1.8, adapter files saved (< 200 MB).
    - Dependencies: steps 8, 9, 10.

12. **Per-stack eval harness (`src/eval/stack_eval.py`)**
    - Files to touch: `src/eval/stack_eval.py`, `data/eval/chat-fr.jsonl` (held-out 100 prompts + judge criteria)
    - Loads base + stack adapter, generates responses, LLM-judges with Mistral-Large-Opus (pairwise vs base-only).
    - Outputs score JSON: `{stack, n_prompts, win_rate_vs_base, avg_judge_score, sample_responses}`.
    - Acceptance: run on stack-01 produces valid JSON, win_rate > 0.55 (stack better than base).
    - Dependencies: step 11.

13. **Evaluate + commit stack-01 baseline**
    - Files to touch: `results/stack-01-baseline.json`, update `README.md` status section
    - Run eval, archive results, log tokens/sec at inference, commit adapter metadata (not weights) to git.
    - Acceptance: `results/` dir exists, JSON committed, README `- [x] First stack trained`.
    - Dependencies: step 12.

### Phase IV — Meta-router v0 (3 stacks)

14. **Train stack-02 (reasoning)**
    - Files to touch: `configs/stack-02-reasoning.yaml`, `data/prompts/reasoning.jsonl`, `data/distilled/reasoning.jsonl`, `data/eval/reasoning.jsonl`, `outputs/stacks/stack-02-reasoning/` (gitignored)
    - Seed prompts from GSM8K trainset + ARC-easy (design-doc domain 2). Distill 2K examples via Qwen3.5-35B Opus (Qwen3 thinking-mode teacher).
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-02-reasoning.yaml`
    - Duration: ~45 min on 4090 (or ~20 min on Studio MLX).
    - Acceptance: win_rate_vs_base ≥ 0.55 on `data/eval/reasoning.jsonl` (50-sample GSM8K subset); no regression > 0.03 on stack-01.
    - Dependencies: steps 9, 12.

15. **Train stack-03 (python)**
    - Files to touch: `configs/stack-03-python.yaml`, `data/prompts/python.jsonl`, `data/distilled/python.jsonl`, `data/eval/python.jsonl`, `outputs/stacks/stack-03-python/` (gitignored)
    - Seed prompts from CodeAlpaca + StackOverflow Python top-tagged. Teacher: Devstral-v3/v4 (kxkm-ai).
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-03-python.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on HumanEval 50-subset; no regression > 0.03 on stacks 01–02.
    - Dependencies: steps 9, 12, 14.

16. **Meta-router module (`src/routing/router.py`)**
    - Files to touch: `src/routing/router.py`, `src/routing/__init__.py`, `tests/test_router.py`
    - Architecture: extract base model last-hidden-state at layer N (N=16 for 4B), pool (mean + attention-query), feed to Linear(3072→512) → LayerNorm → GELU → Linear(512→32) → sigmoid.
    - ~2M params total. Training target: multi-label outcome discovery (binary per-domain, threshold 0.12, chat floor 0.20).
    - Acceptance: `test_router_forward` — dummy batch, outputs shape [B, 32], all in [0,1]; `test_router_init` — trainable params < 3 M.
    - Dependencies: step 2.

17. **Train router v0 on 3-stack mix**
    - Files to touch: `scripts/train_router.py`, `outputs/router/v0/router.pt`
    - Dataset: concat 3 domain jsonls (chat-fr, reasoning, python), label each example with its domain-id; also generate 1K "mixed-intent" examples (queries that span 2 domains) for multi-label training.
    - Loss: BCEWithLogitsLoss, train 3 epochs, lr 1e-3.
    - Acceptance: macro-F1 ≥ 0.85 on 3-domain eval; router file < 10 MB.
    - Dependencies: steps 13, 15, 16.

18. **Runtime adapter switcher (`src/serving/switchable.py`)**
    - Files to touch: `src/serving/switchable.py`, `tests/test_switchable.py`
    - Class `SwitchableModel`: holds base, registry of loaded adapters, method `apply_stacks(names: list[str])` — merges selected LoRAs (weighted sum or sequential application).
    - Cap active stacks at 4. Cache merged weights until stack-set changes.
    - Acceptance: switches between stack-01 ↔ stack-03 ↔ none in < 500 ms; generation before/after differs measurably.
    - Dependencies: steps 11, 15.

19. **End-to-end smoke test (3 stacks)**
    - Files to touch: `tests/test_e2e_3stacks.py`, `scripts/demo_e2e.py`
    - Prompt → router → top-2 sigmoid scores → SwitchableModel.apply_stacks → generate → print.
    - Test cases: "écris un haïku" → chat-fr dominant; "factor 231 into primes" → reasoning dominant; "python fastapi hello world" → python dominant.
    - Acceptance: all 3 test cases have expected domain in top-2 router picks; generation coherent (no garbled tokens).
    - Dependencies: steps 17, 18.

### Phase V — Curriculum coding core + secondary (stacks 04–14)

20. **Forgetting check framework (`src/eval/forgetting.py`)**
    - Files to touch: `src/eval/forgetting.py`, `scripts/run_forgetting.sh`
    - For each already-trained stack, run its eval set with the NEW stack loaded (active-swap test). Flag delta > 0.03 in win_rate_vs_base.
    - Meant to run automatically after EACH training step from now on (wrap `src.stacks.trainer` CLI or emit a shell hook).
    - Acceptance: script runs after stack-02 and stack-03 and emits `results/forgetting-stack-02.json`, `results/forgetting-stack-03.json`; no stack drops > 3%.
    - Dependencies: step 12.

21. **Train stack-04 (typescript)**
    - Files to touch: `configs/stack-04-typescript.yaml`, `data/distilled/typescript.jsonl`, `data/eval/typescript.jsonl`, `outputs/stacks/stack-04-typescript/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-04-typescript.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on stacks 01–03 (forgetting check via step 20).
    - Dependencies: steps 9, 15, 20.

22. **Train stack-05 (cpp)**
    - Files to touch: `configs/stack-05-cpp.yaml`, `data/distilled/cpp.jsonl`, `data/eval/cpp.jsonl`, `outputs/stacks/stack-05-cpp/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-05-cpp.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 21, 20.

23. **Train stack-06 (rust)**
    - Files to touch: `configs/stack-06-rust.yaml`, `data/distilled/rust.jsonl`, `data/eval/rust.jsonl`, `outputs/stacks/stack-06-rust/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-06-rust.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 22, 20.

24. **Train stack-07 (html-css)**
    - Files to touch: `configs/stack-07-html-css.yaml`, `data/distilled/html-css.jsonl`, `data/eval/html-css.jsonl`, `outputs/stacks/stack-07-html-css/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-07-html-css.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 23, 20.

25. **Train stack-08 (shell)**
    - Files to touch: `configs/stack-08-shell.yaml`, `data/distilled/shell.jsonl`, `data/eval/shell.jsonl`, `outputs/stacks/stack-08-shell/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-08-shell.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 24, 20.

26. **Re-train router v1 after stack 08**
    - Files to touch: `outputs/router/v1/router.pt`, `results/router-v1.json`
    - Re-run the router training script (step 17) with the 8-domain set; generate fresh mixed-intent examples covering new pairings.
    - Acceptance: macro-F1 ≥ 0.80 on 8-domain held-out eval; router still < 10 MB.
    - Dependencies: step 25.

27. **Train stack-09 (sql)**
    - Files to touch: `configs/stack-09-sql.yaml`, `data/distilled/sql.jsonl`, `data/eval/sql.jsonl`, `outputs/stacks/stack-09-sql/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-09-sql.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 26, 20.

28. **Train stack-10 (yaml-json)**
    - Files to touch: `configs/stack-10-yaml-json.yaml`, `data/distilled/yaml-json.jsonl`, `data/eval/yaml-json.jsonl`, `outputs/stacks/stack-10-yaml-json/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-10-yaml-json.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 27, 20.

29. **Train stack-11 (docker)**
    - Files to touch: `configs/stack-11-docker.yaml`, `data/distilled/docker.jsonl`, `data/eval/docker.jsonl`, `outputs/stacks/stack-11-docker/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-11-docker.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 28, 20.

30. **Train stack-12 (kicad-dsl)**
    - Files to touch: `configs/stack-12-kicad-dsl.yaml`, `data/distilled/kicad-dsl.jsonl`, `data/eval/kicad-dsl.jsonl`, `outputs/stacks/stack-12-kicad-dsl/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-12-kicad-dsl.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 29, 20.

31. **Re-train router v2 after stack 12**
    - Files to touch: `outputs/router/v2/router.pt`, `results/router-v2.json`
    - 12-domain set + mixed-intent refresh.
    - Acceptance: macro-F1 ≥ 0.80 on 12-domain held-out eval; router still < 10 MB.
    - Dependencies: step 30.

32. **Train stack-13 (spice)**
    - Files to touch: `configs/stack-13-spice.yaml`, `data/distilled/spice.jsonl`, `data/eval/spice.jsonl`, `outputs/stacks/stack-13-spice/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-13-spice.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 31, 20.

33. **Train stack-14 (lua-upy)**
    - Files to touch: `configs/stack-14-lua-upy.yaml`, `data/distilled/lua-upy.jsonl`, `data/eval/lua-upy.jsonl`, `outputs/stacks/stack-14-lua-upy/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-14-lua-upy.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 32, 20.

34. **Re-train router v3 after stack 14**
    - Files to touch: `outputs/router/v3/router.pt`, `results/router-v3.json`
    - 14-domain set + mixed-intent refresh.
    - Acceptance: macro-F1 ≥ 0.80 on 14-domain held-out eval; router still < 10 MB.
    - Dependencies: step 33.

### Phase VI — Technical domains (stacks 15–25)

35. **Curate kiki-* legacy datasets for domains 15–25**
    - Files to touch: `scripts/import_kiki_datasets.py`, `data/distilled/embedded.jsonl`, `data/distilled/stm32.jsonl`, …, `data/distilled/kicad-pcb.jsonl`
    - Using the sources confirmed in step 7's `docs/data-sources.md`, import legacy LoRA training jsonls from `~/Documents/Projets/Factory 4 Life/KIKI-models-tuning/` (or the GitHub mirror), remap to micro-kiki format, run through dedup (step 6).
    - Acceptance: for each of domains 15–25, ≥ 800 distilled examples in `data/distilled/`; dedup report emitted; missing domains flagged for synthetic top-up via teachers.
    - Dependencies: steps 6, 7.

36. **Train stack-15 (embedded)**
    - Files to touch: `configs/stack-15-embedded.yaml`, `data/eval/embedded.jsonl`, `outputs/stacks/stack-15-embedded/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-15-embedded.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 34, 35, 20.

37. **Train stack-16 (stm32)**
    - Files to touch: `configs/stack-16-stm32.yaml`, `data/eval/stm32.jsonl`, `outputs/stacks/stack-16-stm32/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-16-stm32.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 36, 20.

38. **Train stack-17 (iot)**
    - Files to touch: `configs/stack-17-iot.yaml`, `data/eval/iot.jsonl`, `outputs/stacks/stack-17-iot/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-17-iot.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 37, 20.

39. **Train stack-18 (freecad)**
    - Files to touch: `configs/stack-18-freecad.yaml`, `data/eval/freecad.jsonl`, `outputs/stacks/stack-18-freecad/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-18-freecad.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 38, 20.

40. **Train stack-19 (platformio)**
    - Files to touch: `configs/stack-19-platformio.yaml`, `data/eval/platformio.jsonl`, `outputs/stacks/stack-19-platformio/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-19-platformio.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 39, 20.

41. **Train stack-20 (power)**
    - Files to touch: `configs/stack-20-power.yaml`, `data/eval/power.jsonl`, `outputs/stacks/stack-20-power/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-20-power.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 40, 20.

42. **Group eval after stack 20**
    - Files to touch: `results/group-eval-after-20.json`
    - Run router eval + forgetting check across ALL 20 stacks.
    - Acceptance: global macro-F1 ≥ 0.75; no stack-forgetting > 3%.
    - Dependencies: step 41.

43. **Train stack-21 (emc)**
    - Files to touch: `configs/stack-21-emc.yaml`, `data/eval/emc.jsonl`, `outputs/stacks/stack-21-emc/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-21-emc.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 42, 20.

44. **Train stack-22 (dsp)**
    - Files to touch: `configs/stack-22-dsp.yaml`, `data/eval/dsp.jsonl`, `outputs/stacks/stack-22-dsp/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-22-dsp.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 43, 20.

45. **Train stack-23 (spice-sim)**
    - Files to touch: `configs/stack-23-spice-sim.yaml`, `data/eval/spice-sim.jsonl`, `outputs/stacks/stack-23-spice-sim/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-23-spice-sim.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 44, 20.

46. **Train stack-24 (electronics)**
    - Files to touch: `configs/stack-24-electronics.yaml`, `data/eval/electronics.jsonl`, `outputs/stacks/stack-24-electronics/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-24-electronics.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 45, 20.

47. **Group eval after stack 24**
    - Files to touch: `results/group-eval-after-24.json`
    - Run router eval + forgetting check across ALL 24 stacks.
    - Acceptance: global macro-F1 ≥ 0.75; no stack-forgetting > 3%.
    - Dependencies: step 46.

48. **Train stack-25 (kicad-pcb)**
    - Files to touch: `configs/stack-25-kicad-pcb.yaml`, `data/eval/kicad-pcb.jsonl`, `outputs/stacks/stack-25-kicad-pcb/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-25-kicad-pcb.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 47, 20.

49. **Group eval after stack 25 (end of Phase VI)**
    - Files to touch: `results/group-eval-after-25.json`
    - Run router eval + forgetting check across ALL 25 stacks.
    - Acceptance: global macro-F1 ≥ 0.75; no stack-forgetting > 3%.
    - Dependencies: step 48.

### Phase VII — Applications + complements (stacks 26–32)

50. **Train stack-26 (web-frontend)**
    - Files to touch: `configs/stack-26-web-frontend.yaml`, `data/distilled/web-frontend.jsonl`, `data/eval/web-frontend.jsonl`, `outputs/stacks/stack-26-web-frontend/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-26-web-frontend.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 49, 20.

51. **Train stack-27 (web-backend)**
    - Files to touch: `configs/stack-27-web-backend.yaml`, `data/distilled/web-backend.jsonl`, `data/eval/web-backend.jsonl`, `outputs/stacks/stack-27-web-backend/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-27-web-backend.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 50, 20.

52. **Train stack-28 (music-audio)**
    - Files to touch: `configs/stack-28-music-audio.yaml`, `data/distilled/music-audio.jsonl`, `data/eval/music-audio.jsonl`, `outputs/stacks/stack-28-music-audio/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-28-music-audio.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 51, 20.

53. **Train stack-29 (devops)**
    - Files to touch: `configs/stack-29-devops.yaml`, `data/distilled/devops.jsonl`, `data/eval/devops.jsonl`, `outputs/stacks/stack-29-devops/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-29-devops.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 52, 20.

54. **Train stack-30 (llm-orch)**
    - Files to touch: `configs/stack-30-llm-orch.yaml`, `data/distilled/llm-orch.jsonl`, `data/eval/llm-orch.jsonl`, `outputs/stacks/stack-30-llm-orch/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-30-llm-orch.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 53, 20.

55. **Train stack-31 (math)**
    - Files to touch: `configs/stack-31-math.yaml`, `data/distilled/math.jsonl`, `data/eval/math.jsonl`, `outputs/stacks/stack-31-math/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-31-math.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 54, 20.

56. **Train stack-32 (security)**
    - Files to touch: `configs/stack-32-security.yaml`, `data/distilled/security.jsonl`, `data/eval/security.jsonl`, `outputs/stacks/stack-32-security/` (gitignored)
    - Command: `uv run python -m src.stacks.trainer --config configs/stack-32-security.yaml`
    - Duration: ~45 min on 4090.
    - Acceptance: win_rate_vs_base ≥ 0.55 on domain eval; no regression > 0.03 on prior stacks.
    - Dependencies: steps 9, 55, 20.

57. **Final meta-router retrain (v-final, 32 outputs)**
    - Files to touch: `outputs/router/v-final/router.pt`, `results/router-final.json`
    - Full 32-domain training set; include 3K cross-domain multi-intent examples.
    - Acceptance: macro-F1 ≥ 0.82 on held-out 32-class eval; p95 latency < 8 ms on RTX 4090.
    - Dependencies: step 56.

58. **Full evaluation suite**
    - Files to touch: `scripts/run_full_eval.sh`, `results/full-eval-v0.1.json`
    - Run: HumanEval (+ pass@1), GSM8K, MMLU-Pro sample, 32 custom domain evals, latency benchmarks.
    - Acceptance: consolidated JSON report; compare against base-Qwen3.5-4B numbers; each domain shows ≥ 0.55 win rate; aggregate score ≥ +15% over base.
    - Dependencies: step 57.

### Phase VIII — Serving deployment

59. **vLLM server with dynamic LoRA (`src/serving/vllm_server.py`)**
    - Files to touch: `src/serving/vllm_server.py`, `docker/vllm.Dockerfile`
    - vLLM 0.6+ with `--enable-lora --max-loras 4 --max-lora-rank 16`; expose OpenAI-compatible endpoint on :8100.
    - Router runs as a FastAPI sidecar on :8101 that rewrites incoming request with `lora_request` field.
    - Acceptance: `curl :8100/v1/chat/completions` with "write python factorial" → correct response, logs show stacks 03 + 02 active.
    - Dependencies: step 57.

60. **mlx-lm server for Mac Studio (`src/serving/mlx_server.py`)**
    - Files to touch: `src/serving/mlx_server.py`, `configs/mlx-server.json`
    - mlx-lm with adapter switching via subprocess restart (mlx doesn't support hot-swap cleanly — accept ~200ms penalty per switch) OR via `mlx_lm.server --adapter-path` per-request.
    - Acceptance: runs on Studio, serves 32 stacks (max 4 active), p95 first-token latency < 500 ms.
    - Dependencies: step 57.

61. **Persistent service unit**
    - Files to touch: `deploy/launchd/cc.saillant.micro-kiki.plist` (Mac), `deploy/systemd/micro-kiki.service` (Linux)
    - Launchd plist for Studio, systemd for kxkm-ai. Autostart on boot, restart on crash.
    - Acceptance: `launchctl load` + `sudo systemctl enable --now micro-kiki` both work; reboot test passes.
    - Dependencies: steps 59, 60.

62. **Integration test (router HTTP)**
    - Files to touch: `tests/test_integration_http.py`
    - Spawn router sidecar + vLLM (or mlx-server) in pytest fixture, send 20 queries spanning 10 domains, assert each hits correct top-2 stacks.
    - Acceptance: 20/20 queries routed correctly; test completes in < 60 s.
    - Dependencies: step 59 OR step 60.

### Phase IX — ANE triple pipeline (Mac-only)

63. **ANE draft model (`src/serving/ane_draft.py`)**
    - Files to touch: `src/serving/ane_draft.py`, `models/qwen3.5-0.8b.mlpackage`
    - Convert Qwen3.5-0.8B to CoreML (ANE-targeted), use as speculative decoder for the 4B base running on GPU (Metal Performance Shaders).
    - Target: 2-3× decode speedup on M3 Ultra.
    - Acceptance: benchmark before/after on 100-token generation, speedup ≥ 1.8×.
    - Dependencies: step 60.

64. **ANE scorer for GRPO (`src/serving/ane_scorer.py`)**
    - Files to touch: `src/serving/ane_scorer.py`
    - CoreML-compiled scorer (reward model) running on ANE for RL fine-tuning pipelines (future GRPO stacks).
    - Acceptance: scores 100 prompt-completion pairs in < 2 s on ANE; scores match GPU-ref within ±0.02.
    - Dependencies: step 63.

65. **ANE-resident meta-router (`src/serving/ane_router.py`)**
    - Files to touch: `src/serving/ane_router.py`
    - Compile the 2M-param router to CoreML, run on ANE (not CPU) to free CPU for decode orchestration.
    - Acceptance: latency drops from ~5 ms (CPU) to ~1 ms (ANE); identical outputs within fp16 tolerance.
    - Dependencies: step 57.

### Phase X — Release 0.1

66. **Freeze config + migration guide**
    - Files to touch: `MIGRATION.md`, `VERSION`, tag `v0.1.0`
    - Document breaking changes, adapter format, router protocol. Freeze `configs/` (hash-lock).
    - Acceptance: `git tag v0.1.0` pushed; MIGRATION.md reviewed.
    - Dependencies: step 62.

67. **HuggingFace release**
    - Files to touch: `.github/workflows/hf-release.yml`, HF repo `electron-rare/micro-kiki-v0.1`
    - Upload: base-model reference (link to Qwen), 32 adapter .safetensors, router.pt, model card, eval results.
    - Acceptance: `huggingface-cli download electron-rare/micro-kiki-v0.1` works from clean env; model card renders.
    - Dependencies: step 66.

68. **Model card + cookbook**
    - Files to touch: `MODEL_CARD.md`, `COOKBOOK.md`, `examples/` dir
    - Document: intended use, limitations, training data, eval scores, 5+ runnable examples (chat-fr, code, SPICE, KiCad, reasoning).
    - Acceptance: all examples in COOKBOOK run green in CI; model card passes HF lint.
    - Dependencies: step 67.

---

## Success criteria (v0.1)

- 32 stacks trained, each ≥ 0.55 win-rate vs base on its domain.
- Meta-router macro-F1 ≥ 0.82 across 32 domains.
- Serving stack runs on RTX 4090 24 GB (max 4 active stacks) AND on Mac Studio.
- Aggregate eval score ≥ +15% over base Qwen3.5-4B.
- Released on HuggingFace with model card + cookbook.

## Risks + mitigations

- **Forgetting across stacks** → step 20 forgetting check after EACH stack; rollback if > 3%.
- **Router saturation at 32 domains** → re-trains at stacks 08/12/14/final (steps 26, 31, 34, 57); fallback to hierarchical router (6 phase-groups → intra-group classifier).
- **VRAM overrun** → hard cap 4 active stacks, Q4 base, KV cache eviction.
- **Teacher unavailability** → step 3 caches responses; fallback to Qwen3.5-35B (kxkm-ai) if Studio offline.
- **Data gaps for niche domains (15–25)** → step 7 audit flags gaps early; step 35 tops up via legacy + synthetic teacher generation.
