# dream-of-kiki — micro-kiki substrate integration spec

**Date**  : 2026-04-21
**Author** : Spike 1 research (Claude Code agent, kxkm-ai)
**Upstream** : `github.com/hypneum-lab/dream-of-kiki` @ C-v0.7.0+PARTIAL
**Canonical refs** (upstream paths, all under `/tmp/dream-of-kiki/` on kxkm-ai) :
- `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` (framework C)
- `kiki_oniric/core/primitives.py` (8 typed Protocols)
- `kiki_oniric/dream/episode.py` (DreamEpisode + BudgetCap)
- `kiki_oniric/substrates/esnn_thalamocortical.py` (reference substrate)
- `kiki_oniric/substrates/esnn_norse.py` (second reference substrate)

Purpose : map the framework C ABI onto micro-kiki state (mlx_lm model + LoRA
adapter + Aeon records) so we can build a `MicroKikiSubstrate` that satisfies
the DR-3 Conformance Criterion. This document is **read-only notes** — no
upstream changes in Spike 1 / 2.

---

## 1. DreamEpisode — the 5-tuple (upstream §4.1)

```python
@dataclass(frozen=True)
class DreamEpisode:
    trigger: EpisodeTrigger         # SCHEDULED | SATURATION | EXTERNAL
    input_slice: Mapping[str, Any]  # β snapshot + optional α / δ
    operation_set: tuple[Operation, ...]  # REPLAY | DOWNSCALE | RESTRUCTURE | RECOMBINE
    output_channels: tuple[OutputChannel, ...]  # WEIGHT_DELTA | LATENT_SAMPLE | HIERARCHY_CHG | ATTENTION_PRIOR
    budget: BudgetCap               # flops + wall_time_s + energy_j (all >= 0, finite)
    episode_id: str                 # DR-0 traceability
```

Invariants enforced by the dataclass itself:
- `frozen=True` + MappingProxyType on `input_slice` (immutable)
- `operation_set` non-empty
- `BudgetCap` negativity / NaN rejected in `__post_init__`

For the replay handler (used by Spike 2), the convention is that
`input_slice["beta_records"]` is a `list[dict]` where each record has at
minimum `{"x": list[float], "y": list[float]}` (see
`kiki_oniric/dream/operations/replay.py` `replay_handler_mlx`).

---

## 2. The 8 primitives (upstream `kiki_oniric/core/primitives.py`)

All are `runtime_checkable Protocol` types — conforming substrates must
implement each signature exactly. Split across two directions:

**Awake → Dream (inputs, 4)**

| Symbol | Protocol | Activation | Storage | Key methods |
|--------|----------|------------|---------|-------------|
| α | `AlphaStreamProtocol`  | P_max only     | ring mmap    | `append_trace(...)`, `iter_traces()` |
| β | `BetaBufferProtocol`   | all profiles   | SQLite       | `append_record(...)`, `fetch_unconsumed(limit)`, `mark_consumed(ids, de_id)` |
| γ | `GammaSnapshotProtocol`| fallback       | checkpoint   | `get_checkpoint_path()`, `get_checkpoint_sha256()` |
| δ | `DeltaLatentsProtocol` | P_equ, P_max   | ring N=256   | `snapshot(species_acts)`, `get_recent(n)` |

**Dream → Awake (output channels, 4)**

| Symbol | Protocol | Guard invariant | Key methods |
|--------|----------|-----------------|-------------|
| 1 | `WeightDeltaChannel`     | S1 + S2 | `apply(lora_delta, fisher_bump=None)` |
| 2 | `LatentSampleChannel`    | I3      | `enqueue(species, vec, provenance)`, `dequeue()` |
| 3 | `HierarchyChangeChannel` | S3      | `apply_diff(diff: list[tuple[str, dict]])` |
| 4 | `AttentionPriorChannel`  | S4      | `set_prior(prior)`, `get_prior()` |

For micro-kiki the natural mappings are:
- **β buffer** ← Aeon Trace `Episode` nodes with `saillance_score` derived
  from the Atlas retrieval score.
- **γ snapshot** ← the LoRA adapter NPZ safetensors + base-model path.
- **δ latents** ← `src/memory/aeon_predictor.py` latent outputs (per-stack).
- **Channel 1 (weight_delta)** ← LoRA delta applied via `mlx_lm.utils.save_adapters`.
- **Channel 3 (hierarchy_chg)** ← future `post_train_gate` decision to activate/deactivate a stack.
- **Channel 4 (attention_prior)** ← adapter-router logit bias (deferred).

---

## 3. Reference substrate shape — `EsnnSubstrate`

Read from `kiki_oniric/substrates/esnn_thalamocortical.py` (the cycle-2
canonical reference). The substrate is a `@dataclass` exposing 4 factory
methods that return **closures**, not bound methods:

```python
@dataclass
class EsnnSubstrate:
    backend: EsnnBackend = EsnnBackend.NORSE

    def replay_handler_factory(self)     -> Callable[[list[dict], int], NDArray]: ...
    def downscale_handler_factory(self)  -> Callable[[NDArray, float], NDArray]: ...
    def restructure_handler_factory(self)-> Callable[[NDArray, str, int, int], NDArray]: ...
    def recombine_handler_factory(self)  -> Callable[[NDArray, int, int], NDArray]: ...
```

Plus a module-level manifest function:

```python
def esnn_substrate_components() -> dict[str, str]:
    return {
        "primitives":  "kiki_oniric.core.primitives",
        "replay":      "kiki_oniric.substrates.esnn_thalamocortical",
        "downscale":   "...",
        "restructure": "...",
        "recombine":   "...",
        "finite":      "kiki_oniric.dream.guards.finite",
        "topology":    "kiki_oniric.dream.guards.topology",
        "runtime":     "kiki_oniric.dream.runtime",
        "swap":        "kiki_oniric.dream.swap",
        "p_min": "...", "p_equ": "...", "p_max": "...",
    }
```

Module-level constants : `ESNN_SUBSTRATE_NAME`, `ESNN_SUBSTRATE_VERSION`
(matching the framework DualVer of the repo).

Handlers returned by factories are **stateless closures** that consume
concrete NDArray tensors and write nothing side-effecting — state lives on
the substrate instance (MLX weights, LoRA scratch, Aeon episode store).

---

## 4. DR-0..DR-4 axioms (upstream §6.2, brief)

| ID  | Name | What it says |
|-----|------|--------------|
| **DR-0** | Accountability | Every dream-output channel emission must trace to a `DreamEpisode` with finite budget — "no ambient dreaming". Enforced by `DreamRuntime.execute()` logging an `EpisodeLogEntry` for **every** call regardless of handler success/failure. |
| **DR-1** | Episodic conservation (formalises I1) | Every β record created at time t must appear in `inputs(DE_{t'})` for some `t' ∈ [t, t + τ_max]` before being purged. Enforced by hourly cron checking `consumed_by IS NULL AND created_at < now() - τ_max`. |
| **DR-2** | Compositionality (weakened 2026-04-21) | For any permutation π of Op-monoid elements such that no RESTRUCTURE precedes REPLAY, π is composable, budget is additive, and effect chains. The precondition excludes 12 of 24 permutations (RESTRUCTURE reshapes tensors → REPLAY's fixed-shape forward fails). DR-2' (canonical order `replay < downscale < restructure < recombine`) is the stricter operational contract. |
| **DR-3** | Substrate-agnosticism (operational) | A substrate S **conforms** iff (1) typed Protocol signatures, (2) axiom property tests for DR-0/DR-1/DR-2 pass with ≥100 % BLOCKING coverage, (3) invariants S1/S2/S3/I1 are runtime-enforced with abort-on-violation + logging. Evidence is empirical, not a formal implication. |
| **DR-4** | Profile chain inclusion | `ops(P_min) ⊆ ops(P_equ) ⊆ ops(P_max)` and same for channel sets. Lemma DR-4.L : if P_min valid on S, best(P_equ) ≥ best(P_min) in expectation over capacity-monotone metrics. |

**BLOCKING invariants** referenced by DR-3 condition 3 :
- **I1** — episodic conservation until consolidation
- **S1** — retained-task non-regression (post-swap accuracy drop ≤ 2 %)
- **S2** — no NaN/Inf in W_scratch (magnitude ≤ `w_max`)
- **S3** — hierarchy guard (topology validation : connectivity, no rogue cycles, layer bounds)

---

## 5. The 3 profiles (upstream §3.1)

| Profile | Channels in | Channels out | Operations |
|---------|-------------|--------------|------------|
| **P_min** | `{β}`       | `{1}`         | `{replay, downscale}` |
| **P_equ** | `{β, δ}`    | `{1, 3, 4}`   | `{replay, downscale, restructure, recombine_light}` |
| **P_max** | `{α, β, δ}` | `{1, 2, 3, 4}`| `{replay, downscale, restructure, recombine_full}` |

Each profile is a `@dataclass` in `kiki_oniric/profiles/p_*.py` that owns a
`DreamRuntime` + 4 op-states, wires handlers in `__post_init__`. Only
`PMinProfile` currently exposes a `swap_now()` method (S9.4) that drives
`swap_atomic` with a retained-benchmark closure. P_equ / P_max are
"fully wired" but delegate swap to the caller.

**Profile ordering for micro-kiki phase 2** : start with P_min (replay +
downscale only, canal 1 weight-delta out) — matches our use case exactly
(we only need gradient updates from β=Aeon records). Move to P_equ once
the `post_train_gate` hook is re-wired as a `HierarchyChangeChannel`.

---

## 6. Conformance Criterion checklist for `MicroKikiSubstrate`

The DR-3 criterion is executable via `dream-harness conformance --substrate <S>`:

- [ ] **Cond. 1 — signature typing** : factory methods returning the 4
  handler closures with the exact `Callable[..., NDArray | None]` shape of
  `esnn_thalamocortical`.
- [ ] **Cond. 2 — axiom property tests** : run upstream's
  `tests/conformance/axioms/test_dr{0,1,2,4}_*.py` parametrised over our
  substrate. Spike 2 prototype does **not** pass these (deferred to phase 4).
- [ ] **Cond. 3 — invariant enforcement** :
  - I1 (Aeon `consumed_by_DE_id` flag on Trace episodes) — phase 3
  - S1 (retained-benchmark gate — already implemented as
    `scripts/check_forgetting.py`; expose via `swap_atomic`) — phase 3
  - S2 (finite check on LoRA delta before applying) — phase 2
  - S3 (stack-activation topology validation) — phase 3

---

## 7. Installation & test confirmation

Install succeeded on kxkm-ai (Linux x86_64, Python 3.12.11) despite MLX
being an Apple-Silicon-first dependency — MLX 0.31.1 installed as a CPU
build. A 14-test subset ran green :

```
$ cd /tmp/dream-of-kiki && pytest tests/unit/test_episode.py \
    tests/unit/test_esnn_substrate.py tests/unit/test_p_min.py -q --no-cov
14 passed in 0.34s
```

The full test-suite enforces `--cov-fail-under=90` and pulls in fixtures
touching MLX GPU ops that will not work on this host — conformance runs
must happen on the Mac Studio (or an Apple Silicon CI).

---

## 8. Integration hook design (future work, not in Spike 2)

1. **`scripts/check_forgetting.py` → dream opt-in** : add a
   `--consolidate-on-warning` flag. On S1 violation (angle < 30° AND
   win-rate drop > 0.03), instead of rollback, build a `DreamEpisode(
   trigger=SATURATION, operation_set=(REPLAY, DOWNSCALE), ...)` from the
   affected stack's β records and let `MicroKikiSubstrate.consolidate(ep)`
   produce a weight-delta that preserves the retained benchmark.

2. **v4 training loop between stacks** : after each stack finishes
   training, dump the Aeon Trace episodes touched during training as a
   `.jsonl` → feed to `MicroKikiSubstrate.ingest_beta_records()` → at
   curriculum-boundary time schedule `DreamEpisode(trigger=SCHEDULED,
   operation_set=(REPLAY, DOWNSCALE, RESTRUCTURE))`.

3. **Atlas / Trace contract** : Aeon `Episode.content` → β record
   `context` string, `Episode.metadata["outcome"]` → β record
   `outcome` string, retrieval-score → `saillance_score`.

---

## 9. Next steps (phase 3 / phase 4)

- **Phase 3 (½ d)** — open PR upstream adding
  `kiki_oniric/substrates/microkiki.py` (alias to `src.dream.substrate` in
  micro-kiki) and modify `scripts/check_forgetting.py` to expose
  `--consolidate-on-warning`.
- **Phase 4 (½ d)** — run `dream-harness conformance --substrate microkiki`
  on a Mac Studio worker ; iterate until DR-0/DR-1/DR-2'/DR-4 property
  tests + S1/S2/S3/I1 runtime guards all green. Promote DualVer EC axis
  from `+UNSTABLE` to `+PARTIAL`.

Both phases are deliberately out of scope for the current spike — the
purpose of Spike 1 / 2 is to lock the ABI contract and prove the shape.
