# OPLoRA restructure — dream-of-kiki upstream PR note

**Date**: 2026-04-21
**Upstream repo**: `hypneum-lab/dream-of-kiki`
**Branch**: `feat/micro-kiki-oplora-restructure` (stacked on PR #9)
**PR**: https://github.com/hypneum-lab/dream-of-kiki/pull/10 (draft)
**Status**: Draft PR opened, waiting PR #9 merge before marking
ready-for-review.

## Summary

Follow-up PR on dream-of-kiki wiring OPLoRA (Du et al., arXiv
2510.13003) into the `micro_kiki` substrate's `restructure_handler`.
PR #9 landed with the handler as an explicit `NotImplementedError`
stub; this PR replaces the stub with the real projection algebra,
so `micro_kiki` can participate in DR-2 compositionality tests in
the cycle-3 conformance harness.

The local micro-kiki training pipeline at `src/stacks/oplora.py`
keeps its torch-based implementation unchanged. The substrate-side
port lives upstream only; dream-of-kiki stays torch-free.

## Numpy port vs torch local impl

Local (`src/stacks/oplora.py`, torch, 23 lines):

```python
def orthogonal_projection(prior_subspace: torch.Tensor, dim: int):
    q, _ = torch.linalg.qr(prior_subspace.float())
    identity = torch.eye(dim, device=..., dtype=torch.float32)
    return (identity - q @ q.T).to(prior_subspace.dtype)
```

Upstream port (`kiki_oniric/substrates/micro_kiki.py`, numpy):

```python
def _oplora_projector(prior_deltas, rank_thresh=1e-4):
    stacked = np.concatenate(prior_deltas, axis=1)
    U, S, _Vt = np.linalg.svd(stacked, full_matrices=False)
    U_trim = U[:, S > rank_thresh]
    return (np.eye(out_dim) - U_trim @ U_trim.T).astype(np.float32)
```

Key differences:

- **SVD vs QR**: the numpy port uses SVD + singular-value filter
  (`rank_thresh=1e-4`) rather than QR. QR would include numerical-
  noise columns in `Q`, over-pruning the projected subspace; SVD
  lets us drop directions with `sigma < 1e-4`. This matches the
  OPLoRA paper recipe (§3.2 + §3.3 robustness study) more
  faithfully than the torch QR version.
- **Explicit shape guards**: shape mismatch across priors raises
  `ValueError`; empty priors raise at the helper level (the
  handler closure handles the no-op leg so DR-0 still credits
  empty-prior calls).
- **Rank-collapse fallback**: if *all* singular values fall below
  `rank_thresh`, the projector falls back to identity with a log
  warning. Guards against noise-prior inputs silently zeroing out
  a meaningful new adapter.
- **No torch dep**: the upstream substrate runs on the dream
  runtime which is numpy-only.

## Changes

- `kiki_oniric/substrates/micro_kiki.py` (+~150 lines)
  — `_oplora_projector` helper, `MicroKikiRestructureState`
  dataclass, real `restructure_handler_factory` closure, DR-0 /
  DR-1 state propagation.
- `tests/unit/test_micro_kiki_restructure.py` (+291 lines) — 9
  new unit tests for projector algebra and handler contract.
- `tests/unit/test_micro_kiki_substrate.py` — retires the phase-1
  `NotImplementedError` gate in favour of an OPLoRA-wired
  assertion.
- `CHANGELOG.md` — `[Unreleased]` entry under the existing
  micro-kiki block.

## Tests

All new tests pass locally (numpy-only, no MLX / torch dep, runs
on kxkm-ai). Full `tests/unit/` regression suite remains green.

## Next

1. Wait PR #9 to merge to `main` on dream-of-kiki.
2. Rebase `feat/micro-kiki-oplora-restructure` onto `main`.
3. Mark PR ready-for-review.
4. Follow-up PR for `recombine_handler_factory` (TIES-style LoRA
   merge) once the 32-expert curriculum stabilises (tracked in
   dream-of-kiki phase-2 roadmap).

## References

- Du et al., *OPLoRA: Orthogonal Projection for LoRA Continual
  Learning*, arXiv 2510.13003, §3.2–§3.3.
- dream-of-kiki PR #9 — `feat/micro-kiki-substrate` (base).
- Local torch impl — `src/stacks/oplora.py`.
