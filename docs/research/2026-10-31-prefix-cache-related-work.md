# Prefix cache on hybrid models: related work (2026-10-31)

Status: documentation only. No code change beyond a comment in
`src/serving/full_pipeline_server.py`.

## What PR #6 does

`_MLXRuntimeAdapter.generate` keeps a snapshot of the prompt cache holding
exactly `tokens[:-1]`. On a prefix hit it prefills only the new suffix and
generates from a `deepcopy` of the snapshot plus the last prompt token.
Trimming is not an option on Qwen3.x hybrid models: their linear-attention
layers use `ArraysCache`, which `mlx-lm` cannot trim
(`can_trim_prompt_cache` is False), so the recurrent state cannot be
rewound. Measured with `kv_bits=8` on mlx-lm 0.31.2: `prefill_step_size`
512 instead of 2048 cut peak Metal RAM from 4.35 to 2.62 GiB on a
30k-token prompt (Qwen3.5-0.8B) and from 18.07 to 16.23 GiB on a
9.4k-token prompt (Qwen3.8-27B-4bit). The `prefill_step_size` tip came
from Bobby Price (BLACKWEB AI), who also reported the upstream
`RotatingKVCache` masking bug (ml-explore/mlx-lm#1978); credited in PR #6.

## Published work describing the same or adjacent mechanisms

All ids resolved via the arXiv API on 2026-10-31.

| arXiv | Date | Title (first author) | Relation to micro-kiki |
|---|---|---|---|
| [2609.33477](https://arxiv.org/abs/2609.33477) | 2026-09-27 | Just Let Linear States Forget the Distant Past: Prefix Caching via Suffix Replay for Hybrid LLMs (Liu) | Same mechanism as PR #6, published one month earlier; cite when the serving layer is described. |
| [2609.34054](https://arxiv.org/abs/2609.34054) | 2026-09-28 | PReCache: Efficient KV Cache Sharing for Multi-LoRA Agents via Low-Rank Precomputation (Jeon) | Sharing one prefix cache across up to 4 active stacks; not implemented here. |
| [2609.17109](https://arxiv.org/abs/2609.17109) | 2026-09-15 | Shared-Prefix KV Reuse Across Standard LoRA Adapters: Quality and Serving Tradeoffs (Rajput) | Measures the quality loss of reusing a base-model prefix cache under a LoRA adapter; a control to run before sharing caches between stacks. |
| [2609.30884](https://arxiv.org/abs/2609.30884) | 2026-09-25 | CacheReforge: Bounded Recovery for Stale KV Caches under Evolving Adapters (Cao) | Cache staleness after an adapter update; relevant to post-consolidation serving. |
| [2608.05483](https://arxiv.org/abs/2608.05483) | 2026-08-06 | PLoRA: An NDP-Enhanced Pooled-Memory System for Cost-Efficient Multi-LoRA Serving (Yu) | Systems-level multi-LoRA serving; background. |

## Open items

- Decide whether the K Negotiator candidates may share one prefix cache
  across different stacks (2609.17109 gives the control protocol).
- The Juno project (Cassini-Research/Juno#128) ported the same fix for
  `can_trim_prompt_cache(cache) is False`; compare implementations.
