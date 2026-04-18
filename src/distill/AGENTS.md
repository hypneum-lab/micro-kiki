<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# distill — Knowledge Distillation Pipeline

## Purpose
Produces per-domain training data by calling the local Qwen3-Coder-480B-A35B teacher (MLX 4bit on Mac Studio, 1.1 TB), hashing responses, and deduplicating across domains with MinHash+LSH so the 32 experts don't clobber each other on shared prompts. Feeds the BF16 LoRA stack trainer. No network dependency — the teacher is always local.

## Key Files
| File | Description |
|------|-------------|
| `teacher_client.py` | Async OpenAI-compatible client for the 480B teacher. Ships `DEFAULT_ENDPOINTS`, `QWEN3_THINKING_MODELS`, `GenerateParams`, `RetryPolicy`, `TeacherCache` (on-disk), `TeacherClient`, `TeacherError`, `cache_key()` helper. |
| `generator.py` | Dataset generator. `GeneratorConfig` + `TeacherProtocol` structural type; `generate_examples()` streams per-domain prompts through the teacher, writing JSONL. `hash_record()` / `load_existing_hashes()` implement resume-from-checkpoint for long jobs. |
| `dedup.py` | Cross-domain deduplication via MinHash + LSH. Prevents the same near-duplicate prompt leaking into multiple stacks (a common forgetting driver). |
| `__init__.py` | Re-exports the public API: `GeneratorConfig, TeacherProtocol, generate_examples, hash_record, load_existing_hashes, DEFAULT_ENDPOINTS, QWEN3_THINKING_MODELS, GenerateParams, RetryPolicy, TeacherCache, TeacherClient, TeacherError, cache_key`. |

## For AI Agents

### Working In This Directory
- **Teacher is LOCAL Qwen3-Coder-480B-A35B MLX 4bit** on Mac Studio — project `Do`: *Use Qwen3-Coder-480B as teacher for distillation (local, no network dependency)*. Do not add fallbacks to remote APIs.
- **Always run dedup before training**: cross-domain duplicates are the biggest driver of the forgetting check failing later. MinHash+LSH signatures live in a separate index per generator run.
- **On-disk cache is load-bearing**: the teacher is expensive; `TeacherCache` keys off `cache_key()` — never change the key function without a cache migration, otherwise every distillation job re-pays the teacher cost.
- **Resume via `load_existing_hashes()`**: `generate_examples()` skips records whose `hash_record()` is already present. Crashes mid-domain are expected; do not start from scratch.
- **Retry policy is per-request, not per-job**: `RetryPolicy` handles transient MLX server blips. Persistent failures should surface as `TeacherError` and stop the run.
- **Distilled data lives in** `~/KIKI-Mac_tunner/data/micro-kiki/` per project CLAUDE.md — already classified + deduped + split. Do not re-shard it here.

### Testing Requirements
- Mirror tests in `tests/distill/`. `test_teacher_client.py` uses a `TeacherProtocol` stub (no live 480B). `test_dedup.py` verifies LSH collision behaviour at the default MinHash perm count.
- Never run the full distillation in CI — it requires the Mac Studio MLX server. Integration tests mock `TeacherClient`.

### Common Patterns
- `TeacherProtocol` structural typing: callers depend on the protocol, not the concrete `TeacherClient` — makes the generator testable without MLX.
- Deterministic `hash_record()` (content-addressed) so cache hits and dedup both key off the same bytes.
- Async throughout `teacher_client.py`; `generator.py` orchestrates via `asyncio.Semaphore`-bounded fan-out (avoid saturating the single-node MLX teacher).

## Dependencies

### Internal
- Consumed by `scripts/distill_domain.py` and the stack trainer in `src/stacks/`. Older fast/niche variants live in `scripts/legacy/` (pre-pivot).
- Output feeds `src/eval/stack_eval.py` (held-out slices) and forgetting-check eval sets.

### External
- `httpx` (async OpenAI-compatible requests to MLX teacher).
- MinHash/LSH backend (typically `datasketch`) — imported lazily in `dedup.py`.
- Teacher server: Qwen3-Coder-480B-A35B MLX 4bit, OpenAI-compatible endpoint on Mac Studio.

<!-- MANUAL: -->
