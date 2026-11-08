# Codex review of the GB300 code (2026-10-31)

Single pass by Codex CLI (`gpt-6.1-sol`, reasoning effort high, about
108 k tokens), limited to the added code `experiments/nh81_gb300/` and
`experiments/demo_35b_gb300/` with their tests (commit `961ba78`); the
frozen tree and the preregistrations were out of scope. 0 BLOQUANT,
9 MAJEUR, 0 MINEUR. Each finding was checked in the source.

| # | Finding (short) | Verdict | Action |
|---|---|---|---|
| 1 | `gpu_pids` fallback to `nvidia-smi` dropped graphics processes | confirmed | fixed: compute and graphics each queried (`_v3` then `_v2`); any failure fails closed |
| 2 | probe sampler ignored failed or decreasing reads and integrator faults | confirmed | fixed: faults and gaps recorded; P0 or P1 with a fault is void |
| 3 | P1 did not check CPU during acquisition; the demo P1 did not watch foreign GPU processes | confirmed | fixed: GPU and CPU watched through P1 in both; void, one retry, then not run |
| 4 | `window_ok_gb300` does not apply CPU / load rules to run windows | confirmed, kept | not changed: the frozen protocol (parent prereg, KXKM r2-r4) logs CPU during runs without rejecting; the GB300 prereg keeps that rule (section 1) |
| 5 | integrator could return a stale cumulative value; stall check skipped it | confirmed | fixed: staleness check at each read, loop exceptions counted as faults, stall check for both instruments |
| 6 | race between fault count and its acknowledgement | confirmed | fixed: one lock for the count and the acknowledgement |
| 7 | launchers lost the run's exit status | confirmed | fixed: status kept and returned after the summary / cleanup |
| 8 | deadlines did not bound the rest probe or the downloads | confirmed | fixed: remaining time applied to the probe and to each download, `--kill-after` added |
| 9 | demo runs under 15 s entered the medians | confirmed | fixed: short runs stay valid but are reported separately and kept out of the medians, as the demo prereg states |

Codex could not verify the GB300 driver ABI or aarch64 execution; these
remain covered by the smoke tests and the identity check of the
launchers.
