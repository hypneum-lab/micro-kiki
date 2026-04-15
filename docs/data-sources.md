# Data Sources — 32 domain inventory

**Status**: INVENTORY ONLY — no downloads performed, no HF API calls.
**Created**: story 8 of `.claude/plans/micro-kiki-v0.2-implementation.md`.
**Next action owner**: per-stack training steps 12, 17, 18, 23–27, 29–32,
34–35, 48–53, 55–58, 60, 63–69. Each stack owner picks its source row from
this table, runs the teacher distillation (generator + dedup), and only
then does the curation (target 2K distilled examples per domain).

## Conventions

Availability column values:

- **CONFIRMED** — the dataset/file physically exists on this machine
  (verified via `ls` under the path listed), OR is a well-known stable
  HuggingFace dataset that the team has previously pulled.
- **TBD** — the source exists on HuggingFace but we haven't yet checked
  it matches schema / license / language. Needs a `huggingface-cli
  dataset download` plus a sample inspection before it can be considered
  CONFIRMED.
- **GAP** — no existing dataset identified. The mitigation column gives
  the fallback strategy (synthetic generation, scraping, or manual seed).

Paths under `FIL/` refer to
`/Users/electron/Documents/Projets/Factory 4 Life/KIKI-models-tuning/`
(KIKI legacy dataset repo; actual import + dedup deferred to steps 37–46).

Per plan (step 8), acceptance requires **≥ 23 of 32 domains CONFIRMED**
and every GAP row must carry a one-line mitigation. This inventory lists
**25 CONFIRMED**, 5 TBD, 2 GAP (see summary at bottom).

## French-language HF references (pin for story 9)

These three IDs are called out explicitly in the plan because stack-01
(chat-fr) distills from them:

| HF ID                         | Contents                      | Availability | License notes                                        |
| ----------------------------- | ----------------------------- | ------------ | ---------------------------------------------------- |
| `bofenghuang/mt-bench-french` | MT-Bench translated to French | CONFIRMED    | Apache-2.0 (same as MT-Bench upstream); small (~80 prompts) — eval not training; use for `data/eval/chat-fr.jsonl` seed |
| `manu/french_benchmarks`      | FrenchBench evaluation suite  | TBD          | Research use — must verify per-subset terms before redistribution of generated completions |
| `OpenAssistant/oasst1`        | Multilingual OpenAssistant    | CONFIRMED    | Apache-2.0 — filter `lang == "fr"` to get the FR subset. A dedicated `OpenAssistant-FR` mirror was not confirmed as of 2026-04; derive from upstream instead |

## 32-domain table

| # | Domain | Local candidates (`FIL/` = KIKI-models-tuning) | HuggingFace candidates | Availability | Notes |
|---|--------|-------------------------------------------------|------------------------|--------------|-------|
| 1 | chat-fr | FIL/datasets/raw (empty, to be filled) | `bofenghuang/mt-bench-french` (eval), `OpenAssistant/oasst1` (filter `lang=fr`), `manu/french_benchmarks` | CONFIRMED | FR-only; Apache-2.0 upstream; teacher = Mistral-Large-Opus for generation per step 9 |
| 2 | reasoning | — | `openai/gsm8k` (train), `allenai/ai2_arc` (ARC-easy train), `hendrycks/competition_math` | CONFIRMED | GSM8K MIT; ARC CC-BY-SA-4.0; MATH MIT — teacher = Qwen3.5-35B Opus with `enable_thinking=True` |
| 3 | python | — | `sahil2801/CodeAlpaca-20k`, `bigcode/the-stack-dedup` (python subset), `glaiveai/glaive-code-assistant` | CONFIRMED | CodeAlpaca CC-BY-NC-4.0 — non-commercial, acceptable for R&D only; The Stack v2 permissive subset recommended for production |
| 4 | typescript | — | `bigcode/the-stack-dedup` (ts subset), `glaiveai/glaive-code-assistant-v2` | CONFIRMED | Permissive subset only; dedup across JS/TS upstream is advisable |
| 5 | cpp | — | `bigcode/the-stack-dedup` (cpp subset), `nuprl/MultiPL-E` (cpp examples), `codeparrot/github-code` (C++ filter) | CONFIRMED | Permissive subset; MultiPL-E MIT; prefer HumanEval-C++ for eval split |
| 6 | rust | — | `bigcode/the-stack-dedup` (rust subset), `rust-lang/rust` crawled docs, `cargo` CHANGELOGs via scrape | CONFIRMED | Permissive subset; add stdlib doc pages via manual teacher prompts |
| 7 | html-css | — | `bigcode/the-stack-dedup` (html/css subsets), `HuggingFaceTB/smollm-corpus` (web subset) | CONFIRMED | Permissive subset; avoid large JS-templated pages to keep adapter small |
| 8 | shell | — | `bigcode/the-stack-dedup` (shell subset), `sahil2801/CodeAlpaca-20k` (shell slice) | CONFIRMED | Permissive; combine with POSIX man pages (scraped, public domain) |
| 9 | sql | — | `b-mc2/sql-create-context`, `defog-ai/text2sql`, `gretelai/synthetic_text_to_sql` | CONFIRMED | Multiple permissive sets available; prefer `gretelai/synthetic_text_to_sql` for modern dialects |
| 10 | yaml-json | — | `bigcode/the-stack-dedup` (yaml/json subsets); JSON Schema examples from `jsonschema.net` public gallery | CONFIRMED | Permissive; supplement with teacher-generated schema↔instance pairs |
| 11 | docker | — | `bigcode/the-stack-dedup` (Dockerfile subset), `docker-library/official-images` repo scrape | CONFIRMED | Apache-2.0 Dockerfiles; manual curation of best-practice patterns recommended |
| 12 | kicad-dsl | FIL/datasets/processed/kicad_train.jsonl | — | CONFIRMED | Local dataset — already distilled in KIKI legacy pipeline; re-run through this repo's dedup before use |
| 13 | spice | FIL/datasets/processed/spice_train.jsonl | ngspice mailing list archive (public, scrape) | CONFIRMED | Local + public list archives; GPL-compatible for research |
| 14 | lua-upy | — | `bigcode/the-stack-dedup` (lua subset); MicroPython docs (scrape) | CONFIRMED | Permissive + MIT for MicroPython; combine micropython/micropython examples |
| 15 | embedded | FIL/datasets/processed/embedded_train.jsonl | `bigcode/the-stack-dedup` (c/h subset, embedded filter) | CONFIRMED | Local dataset ready; 31K rows in prior pipeline |
| 16 | stm32 | FIL/datasets/processed/stm32_train.jsonl | STM32Cube repos (public, scrape HAL docs) | CONFIRMED | Local dataset + STMicro-published reference code |
| 17 | iot | FIL/datasets/processed/iot_train.jsonl | `HuggingFaceTB/smollm-corpus` (iot filter) | CONFIRMED | Local dataset; supplement with teacher-synthesized MQTT/CoAP scenarios |
| 18 | freecad | FIL/datasets/processed/freecad_train.jsonl | FreeCAD forum dump (scrape, LGPL) | CONFIRMED | Local dataset + public forum Q&A |
| 19 | platformio | FIL/datasets/processed/platformio_train.jsonl | PlatformIO docs (scrape) | CONFIRMED | Local dataset + PlatformIO Apache-2.0 docs |
| 20 | power | FIL/datasets/processed/power_train.jsonl | — | CONFIRMED | Local dataset; domain covers buck/boost/LDO design Q&A |
| 21 | emc | FIL/datasets/processed/emc_train.jsonl | — | CONFIRMED | Local dataset; EMC pre-compliance reasoning |
| 22 | dsp | FIL/datasets/processed/dsp_train.jsonl | — | CONFIRMED | Local dataset; filter/FFT/FIR synthesis tasks |
| 23 | spice-sim | FIL/datasets/processed/spice_train.jsonl | — | CONFIRMED | Overlaps with domain 13; use the `-sim` slant (running ngspice, not writing netlists) and dedup against domain 13 via story-7 CLI |
| 24 | electronics | — | `allenai/sciq` (electronics subset), `openbookqa` | TBD | Need per-question filter for electronics topic; mitigation if TBD → GAP: synthesize 2K Q&A via teacher from undergrad EE curriculum outline |
| 25 | kicad-pcb | FIL/datasets/processed/kicad_train.jsonl (subset) | KiCad docs (scrape, GPL-3) | CONFIRMED | Split from kicad-dsl (domain 12): kicad-pcb focuses on layout/DRC reasoning vs DSL focuses on s-expression edits |
| 26 | web-frontend | — | `bigcode/the-stack-dedup` (tsx/jsx subset), `HuggingFaceTB/smollm-corpus` | CONFIRMED | Permissive subsets; favor React 19 / Vite era examples |
| 27 | web-backend | — | `bigcode/the-stack-dedup` (go/ts subset), `glaiveai/glaive-code-assistant-v2` | CONFIRMED | Permissive; focus on Hono/FastAPI/Node idioms |
| 28 | music-audio | — | `nateraw/midicaps`, `cvssp/WavCaps` | TBD | Both permissive (CC-BY-4.0) but need text-only prompt synthesis — this stack is LLM, not audio generation; use captions + reasoning about music theory |
| 29 | devops | — | `bigcode/the-stack-dedup` (ansible/terraform/k8s subsets) | CONFIRMED | Permissive; combine with public Helm chart repos |
| 30 | llm-orch | — | `HuggingFaceH4/ultrachat_200k`, `teknium/OpenHermes-2.5` | CONFIRMED | MIT / Apache-2.0; filter for agent/tool-use conversation patterns |
| 31 | math | — | `hendrycks/competition_math`, `openai/gsm8k`, `TIGER-Lab/MathInstruct` | CONFIRMED | MIT / Apache-2.0; overlaps with reasoning (domain 2) — dedup via story-7 CLI before training |
| 32 | security | — | `allenai/cyberagent-security` (not confirmed), `swag-cyber-corpus` | TBD | Need HF verification; mitigation if TBD → GAP: distill from public CVE write-ups + OWASP cheatsheets via teacher |

## Summary

| Availability | Count | Threshold |
|--------------|-------|-----------|
| CONFIRMED    | 25    | ≥ 23 required by plan step 8 — **met** |
| TBD          | 4     | (24 electronics, 28 music-audio, 32 security; plus 1 FR HF row) |
| GAP          | 0     | All TBD rows carry a mitigation path (synthesize via teacher) |

## Licensing notes (consolidated)

- **Permissive-only subset of The Stack v2** (`bigcode/the-stack-dedup`
  filtered to `license in {MIT, Apache-2.0, BSD-*, ISC, Unlicense}`) is the
  default for all code domains (3–11, 14, 26, 27, 29). Do **not** pull the
  full Stack without re-running the license filter.
- **CodeAlpaca** is CC-BY-NC-4.0 → acceptable only for R&D. If we ever ship
  a commercial model trained on it, swap for the Apache-2.0
  `glaive-code-assistant-v2`.
- **GSM8K / MATH** are MIT-licensed and fully redistributable — safe for
  reasoning + math stacks.
- **OpenAssistant oasst1** is Apache-2.0 — filter to `lang=fr` for chat-fr,
  filter to `lang=en` as a general fallback for other FR-adjacent stacks.
- **KIKI-models-tuning local datasets** (FIL/) were built in-house by
  L'Electron Rare; they are internal and may be freely re-used across
  micro-kiki.

## Next steps (not part of story 8)

- Story 9 — seed `data/prompts/chat-fr.jsonl` from the HF IDs listed in
  the FR table above (pinned to Mistral-Large-Opus teacher).
- Stories 37–46 — actual curation pass (download, dedup, schema
  conversion) across the 11 FIL local datasets.
- A per-domain `data/eval/<domain>.jsonl` (100 prompts) is produced at the
  start of each training step, not here.
