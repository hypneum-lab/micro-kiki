# NH-81 KXKM replication, revision r3: power-based rest guard

**Level:** L1 (frozen in git: commit + annotated tag
`prereg-nh81-kxkm-20261029-r3`, pushed before any energy measurement
under this revision).
**Date:** 2026-10-29.
**Issue:** hypneum-lab/micro-kiki#14 (NH-81).
**Amends:** `docs/prereg/2026-10-28-nh81-kxkm-replication.md` (tag
`prereg-nh81-kxkm-20261028-r2`). Everything in r2 holds unless replaced
below: same checkpoints and SHA-256, same code path (MLX 0.31.1 CUDA,
strict fp32 with `MLX_ENABLE_TF32=0`), same workloads, 12 valid pairs
per block, 20 s rest windows, calibration, replay, statistics, the
primary criterion (W1 generation, fused path: withdraw if R-hat <= 1.1)
and the PI's status rule (the macM1 verdict "withdraw, provisional
(IOReport)" becomes definitive only if the KXKM primary decision is
"withdraw"; divergence keeps it provisional and is reported).
**PI decision (2026-10-29):** replace the utilisation guard by a guard
based on board power, keep the foreign-process rule, keep the measured
and subtracted rest reference.

## 1. Why the utilisation guard is dropped

r2 inherited from macM1 "mean GPU utilisation <= 10 % in the rest
windows". On the RTX 4090 that is not a rest test:

* NVML utilisation is the fraction of time a kernel runs, not scaled by
  clock. At the lowest P-state (P8: SM 210 MHz, memory 405 MHz, board
  18-21 W) the resident, idle set (ComfyUI, Xorg, gnome-shell, login
  screen on `:0`) reads **36-44 %** GPU and 30-34 % memory utilisation;
  the same set reads 1-4 % at P2 (2565 MHz).
* Attempt 1 (2026-10-28 16:22-19:24): guard readings 16-21 % at every
  check (mixed with a crash-looping service, now disabled by the PI),
  then 39-68 % under N10-C jobs; 3 h, no pair.
* Attempt 2 (2026-10-28 20:18): no foreign compute process, idle probe
  39.9 % mean utilisation at 20.6 W; first guard check 38 %; stopped.

Utilisation is still logged for every window and reported, descriptive
only.

## 2. Board power on this card (measured before this revision)

Characterisation on 2026-10-29 06:10-06:15 CET with a synthetic load
(2048x2048 matmuls in MLX, no NH-81 model, 15 s), NVML power every
0.5 s (`/ai/saisail/nh81-kxkm/power_decay.py`):

| state | board power |
|---|---|
| P8, no CUDA context of ours (idle probes 2026-10-28/29) | 18-21 W |
| P2, our process holds a CUDA context, no kernel running | 67-71 W (76 W seen at 56 C) |
| synthetic load | 205 W mean, 216 W max |
| after the load, process alive | ~69 W for ~17 s, then P5 (39 W) and P8 (18-19 W) within ~5 s |
| N10-C pretrain of another session (attempt 1) | ~170 W |

So the 35 W suggested from the P8 figure would reject every rest window
taken while the measurement process is alive: the card stays in P2 at
~69 W for ~17 s after each run, i.e. for most of a 20 s rest window.

## 3. Rest guard (replaces the utilisation criterion everywhere)

* **Power threshold: 90 W.** Justification: above every idle state of
  the card with our context (P2 idle at most 76 W observed), well below
  any real foreign GPU job (N10-C ~170 W, synthetic load 205 W).
* **Rest windows.** A rest window is "at rest" if its **mean board
  power (NVML energy of the window / its duration, 20 s) is <= 90 W**,
  no other Python process exceeds 20 % CPU over the window (r2 rule),
  and **no compute process outside the baseline B** is seen in it (r2
  rule). A pair is rejected, kept in the log and replayed (r2 rules) if
  its rest window before or after is not at rest, or if a compute
  process outside B is seen in either run.
* **Start guard.** B is frozen at the start of each 2-min check; the
  check passes if, sampled every 5 s for 2 min, board power stays
  < 90 W, no compute process outside B appears, no other Python process
  exceeds 20 % CPU and no N10-C job is active. Re-checked every 10 min
  for up to 3 h (r2).
* **Pause rule.** Before every pair, the measurement pauses if an N10-C
  job is active, a compute process outside B is on the GPU, or board
  power is >= 90 W; it resumes after 2 min clean, within a cumulative
  3 h budget (r2).
* **Rest reference, unchanged in principle.** Net energy of a run =
  NVML energy over the run - (mean board power of the preceding rest
  window) x run duration, as on macM1 and in r2. Because the rest
  window after a run spans the P2-idle tail (~69 W) and part of the drop
  to P8, its mean power is reported for every pair, and two
  descriptive sensitivity readings are added: the gross ratio (no
  subtraction, already in r2) and the ratio with a fixed P2-idle
  reference equal to the median rest-window power of the run. Neither
  can change the decision.

## 4. Stop rule for this revision

If the start guard of r3 refuses (3 h without a passing check) or the
pause budget is exhausted, the run stops and is reported; it is not
relaunched without a new PI decision.

## 5. Disclosure

Between the r2 tag and this revision: attempts 1 and 2 (guard refusals,
no pair, logs in `results/nh81-kxkm/attempt*/`), and the synthetic
power characterisation of section 2. No NH-81 energy pair has been run
on KXKM.
