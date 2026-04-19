# C3 PII Audit (2026-04-19)

Grep sweep of `data/corpus-real/` after build from mascarade-datasets:

| Pattern | Hits | Notes |
|---|---|---|
| email regex `[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}` | 0 | clean |
| IPv4 `\b(?:\d{1,3}\.){3}\d{1,3}\b` | 0 | clean |
| API keys `sk-`, `hf_`, `xai-`, `ghp_`, `gho_` | 0 | clean |

## Corpus composition

- Total: 500 lines (50 per domain × 10 domains)
- Source: mascarade-datasets `*_chat.jsonl`
- Filename-derived domain labels (via `--use-filename-domain`), matching:
  - `dsp_chat.jsonl` → `dsp`
  - `embedded_chat.jsonl` → `embedded`
  - `iot_chat.jsonl` → `electronics` (remap)
  - `kicad_chat.jsonl` → `kicad-dsl` (remap)
  - other 7 files → same-name domains
- Cluster-taxonomy overlap: **0.410** (vs chance 0.100) — 4× better than random, confirming real domain structure.

## Caveat (reiterated)

Source is **teacher-generated** from Qwen3-Coder-480B during KIKI stack-training, NOT user-collected real dialogues. Honest framing in Paper A §6:

> The "real corpus" used in C3 is teacher-generated, representative of the query distribution the model would encounter in its intended deployment, but not sampled from actual end-user interactions. Full user-collected corpus is future work.

## Sign-off

- Grep sweep: clean
- Manual random-sample review: 10 lines/domain inspected, no PII or secrets observed
- Teacher-generated nature means PII risk was already low a priori
- Audited 2026-04-19
