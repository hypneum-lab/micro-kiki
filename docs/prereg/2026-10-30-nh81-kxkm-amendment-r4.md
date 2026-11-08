# NH-81 KXKM replication, amendment r4: CPU check averaged over the guard window

**Level:** L1 (frozen in git: commit + annotated tag
`prereg-nh81-kxkm-20261030-r4`, pushed before any energy measurement
under this amendment).
**Date:** 2026-10-30.
**Issue:** hypneum-lab/micro-kiki#14 (NH-81).
**Amends:** `docs/prereg/2026-10-29-nh81-kxkm-replication-r3.md` (tag
`prereg-nh81-kxkm-20261029-r3`), which amends r2
(`docs/prereg/2026-10-28-nh81-kxkm-replication.md`).
**PI decision (2026-10-30):** this single change, then an automatic
relaunch after the E1-small confirmatory phase.

**L2 deposit (annotation added 2026-11-08):** restricted Zenodo record
[10.5281/zenodo.24053794](https://doi.org/10.5281/zenodo.24053794) (Hypneum Lab lot of
2026-11-08, file `prereg-nh81-kxkm-20261030-r4.zip`, SHA-256
`09efab1c216925883dc927024950f7c8917cb3820f1d74a1c69669cef843679b`) gives the freeze tag
`prereg-nh81-kxkm-20261030-r4` (commit `979771f0cf1f061ad906d8e420bad0664da9c5a0`,
`swh:1:rev:979771f0cf1f061ad906d8e420bad0664da9c5a0`) a verifiable deposit date, level
L2 of PROJECT-MANAGEMENT section 4. The deposit holds this file as it was at the tag;
anything dated after the tag in this live file is not in the deposit.
Restored on the default branch on 2026-11-08 from the tag; the document was
previously only on branch `exp/nh81-kxkm-nvml`.

## Single change

In the **start guard**, the CPU criterion "no other Python process above
20 % CPU" is computed as the **mean CPU over the whole 2-min guard
window** (CPU ticks from `/proc` at the start and at the end of the
window), instead of a 1-s sample taken every 5 s.

Everything else is unchanged from r3: the 90 W board-power threshold
(sampled every 5 s through the window, any sample >= 90 W fails the
check), the no-foreign-compute-process rule with a baseline frozen at
the start of each check, the N10-C rule, the 2-min window, the re-check
every 10 min for up to 3 h, the rest-window rules (board power, and CPU
already averaged over each 20-s window), the pause rule, replay,
statistics, primary criterion (W1 fused, withdraw if R-hat <= 1.1) and
the PI status rule (definitive only if KXKM confirms).

## Justification

Attempt 3 (commit `994de91`, `results/nh81-kxkm/attempt3-20261029-r3/`):
the first check at 2026-10-29 23:58 CET was refused because ComfyUI's
Python exceeded 20 % CPU in a 1-s sample, while the board drew 19.3 W
(P8) with no foreign compute process. A 1-s sample turns a transient
spike of an idle resident service into a refusal; a window average is
what the rule means (sustained foreign CPU load during the guard).

## Launch

A waiter script (read-only on other jobs) starts the unchanged r4
launcher once no E1-small or N10-C process remains, detected from the
process table and the GPU compute-process list, not from a fixed time.
If no window has opened by 2026-11-04 12:00 CET, the waiter stops and
the outcome is reported. If the start guard then refuses for 3 h, the
run stops and is reported, without relaunch.
