# NH-81 amendment: third replication on macM3 (Apple M3 Pro, IOReport)

**Level:** L1 (frozen in git: commit + annotated tag
`prereg-nh81-m3pro-20261030`, pushed before any energy measurement on
macM3).
**Date:** 2026-10-30.
**Issue:** hypneum-lab/micro-kiki#14 (NH-81).
**Amends:** the macM1 preregistration
`docs/prereg/2026-10-25-nh81-joules-per-token.md` (tag
`prereg-nh81-20261025`) and amendment 1; sits next to the KXKM NVML
replication (r2, r3, amendment r4), which it does not change.
**PI decision (2026-10-30 13:39):** replicate NH-81 on a third
platform, macM3 (M3 Pro), with IOReport, at any hour.

## 1. What is identical to the macM1 measurement

* **Models and data:** the two step-12 000 checkpoints and the held-out
  splits copied from macM1, SHA-256 checked on macM3 (dense
  `3c0891fb...afd7`, spiking `e9a3c929...a074`, test `ad59a2f1...fd2d`,
  val `bc703894...257a`, BPE `78bce0d3...0f4c`).
* **Code:** `experiments/nh81_joules/` at the tagged commit; the model
  module, workloads (W1 generation, W2 scoring; fused and event spiking
  paths), block order, calibration rule (>= 15 s on the dense model),
  12 valid pairs per block with replay of rejected pairs (at most 24
  attempts per block), 20 s rest windows, rest-power subtraction,
  statistics (geometric-mean R, bootstrap CI) and decision rule
  (W1 fused: withdraw if R-hat <= 1.1; reopen if R-hat >= 2 with CI lower
  bound > 1.1; inconclusive otherwise) are those of macM1. Prompts are
  tokenised on the host with the same BPE (tokenizers 0.22.2).
* **Instrument:** IOReport "Energy Model", **GPU rail** (`GPU Energy`),
  sudo-less, as on macM1 (amendment A1). CPU rail not used.
* **Framework and precision:** MLX 0.31.1 (Metal) from the repository
  lockfile (`uv sync --frozen --extra mlx`, uv 0.13.5, CPython 3.12.15;
  macM1 ran 3.12.13 with the same MLX/numpy/tokenizers versions), fp32.
  `MLX_ENABLE_TF32=0` is set as on KXKM (it has no effect on Metal).
* **Iso-quality:** val/test losses of both checkpoints recomputed on
  macM3 before the first pair; reproduction tolerance 0.005 nats against
  macM1 and the 0.02-nat iso-quality rule, as in the KXKM prereg r2
  section 3.

## 2. Host

macM3: Apple M3 Pro (6 performance + 6 efficiency cores, 18-core GPU),
18 GB, macOS 27.0.1, on AC power (laptop), rebooted on 2026-10-30
13:42 by the PI. Other users of the host are never touched: the M5
grid (4 Python workers, nice 19, `~/scratch/m5-macm3`), bouba_sens
extraction (unknown slots), other sessions' jobs. Runs use `nohup` and
`caffeinate -i` (unstable Wi-Fi).

## 3. Rest guard (no time window)

Idle probe after the reboot (2026-10-30 13:49-13:51 CET, 120 s, no model
loaded, `results/nh81-m3pro/idle_probe_prereg.json`): GPU rail
**18 mW** mean (10-s values 4.6-100 mW), GPU utilisation 0.55 % mean
(max 19 %), 1-min load average 2.6 mean (max 3.1, Spotlight still
indexing after boot), no python/ffmpeg/bouba process above 20 % CPU.

Fixed thresholds:

* **GPU-rail rest power <= 0.25 W.** About 14x the probe mean and 2.5x
  its largest 10-s value; below the 0.41 W read on macM1 while other
  sessions' GPU jobs ran at ~19 % utilisation (macM1 amendment A3).
* **1-min load average <= 6.0** (half the 12 cores). The measurement
  itself adds about one busy core; the M5 grid (4 workers) alone brings
  the load to ~8-9.
* **No other heavy process:** no process whose name matches `python`,
  `ffmpeg` or `bouba` above **20 % CPU averaged over the window**
  (cumulative CPU time from `ps`, as KXKM amendment r4).

Applied as follows:

* **Start guard:** a 2-min window where GPU-rail power (1-s reads every
  5 s) stays < 0.25 W, load average < 6.0, no N10-C activity
  (`run_campaign`, or a file under `~/scratch/n10c/runs/` modified in
  the last 10 min), and no heavy process above 20 % CPU averaged over
  the 2 min. Re-checked every 10 min for up to 3 h.
* **Rest windows (rejection rule):** a pair is rejected, kept in the
  log and replayed if its rest window before or after has mean GPU-rail
  power > 0.25 W (window energy / 20 s), load average > 6.0 at the end
  of the window, or a heavy process above 20 % CPU averaged over the
  window.
* **Pause rule** before each pair: N10-C active, GPU-rail power >= 0.25 W
  (1-s read) or load average >= 6.0; resume after 2 min clean;
  cumulative pause budget 3 h.
* GPU utilisation (ioreg) is logged, descriptive only.

## 4. Launch

A waiter checks every 10 min (instantaneous `ps %cpu` of heavy
processes, load average, M5 workers present) and starts **one** complete
measurement once the host looks free, in practice after the M5 grid.
The measurement's own start guard (section 3) then decides. The waiter
gives up on 2026-11-04 12:00 CET. If the start guard refuses for 3 h,
the run stops and is reported, without relaunch.

## 5. How the M3 Pro result enters the verdict (fixed in advance)

Current status: **"withdraw, provisional (IOReport)"**; the PI rule
makes it definitive only if the KXKM NVML replication confirms
(KXKM W1 fused R-hat <= 1.1).

Options considered: (a) M3 Pro descriptive only; (b) a joint criterion
with KXKM (either platform could make the verdict definitive).
**Choice: (a), the most cautious, with a veto on divergence.**

* The M3 Pro result **cannot make the verdict definitive** on its own,
  nor replace KXKM: it is a second IOReport measurement on Apple silicon,
  not an independent instrument.
* If the M3 Pro primary block gives **R-hat > 1.1** (decision
  "inconclusive" or "reopen"), the divergence is reported and the
  verdict **stays provisional even if KXKM confirms**.
* If the M3 Pro primary block gives "withdraw", it is reported as
  agreement; the status follows the KXKM rule unchanged.
* If M3 Pro is not measured (guard, deadline, reproduction check
  failed), nothing changes.

## 6. Artifacts

`experiments/nh81_joules/power_m3.py`, `idle_probe_m3.py`,
`m3_launcher.sh`, `m3_wait.sh`; `results/nh81-m3pro/` (probe,
iso-quality, raw log, summary); results note to follow.
