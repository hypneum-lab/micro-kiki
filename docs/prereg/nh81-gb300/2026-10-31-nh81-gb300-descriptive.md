# NH-81 on Blackwell (DGX Station GB300): descriptive third-hardware point

**Level:** L0 (draft on a branch, not frozen). It becomes L1 only after
every "TBD by PI" item below is closed, the code delta of section 6 is
committed, tested and reviewed, the resolved package set of
`hitl-packages.md` is reviewed, the PR is merged on green CI, and an
annotated tag plus a Software Heritage revision SWHID are recorded. No
energy measurement on the GB300 may start before that tag is pushed.
If a hardware qualification step of section 3 or 4 fails on the day,
the outcome is "not measured" with the reason; nothing is re-adjusted.
**Date:** 2026-10-31.
**Issue:** hypneum-lab/micro-kiki#14 (NH-81). Related PR: #19 (KXKM and
M3 Pro replications, branch `exp/nh81-kxkm-nvml`).
**PI decision (2026-10-31 21:20 CET):** prepare tonight the
pre-registrations for tomorrow's 24-h DGX Station GB300 slot.
**Status of this study: DESCRIPTIVE.** It produces no verdict, cannot
change the NH-81 status in any direction (no confirmation, no veto),
and contains no inferential comparison between hosts.

## TBD by PI (blocking the freeze)

1. **Rest-guard factor k** (section 4.2): threshold T = k x P_rest.
   Proposed: k = 1.25.
2. **Time budgets inside the slot** (section 4.4): start-guard window,
   cumulative pause budget and the global deadline after which a
   partial run stops (blocks already complete are reported, the others
   "not measured"). Proposed: 1 h, 1 h, and a hard stop 3 h after the
   first probe.
3. **Slot exclusivity** (section 4.4): the slot hour when NH-81 runs
   alone, the list of processes allowed on the host during the run
   (proposed: system services and the display stack only, identified by
   name in `host.json`), and the CPU rule (proposed: no process other
   than ours above 20 % CPU over a window, Python or not, host CPU
   time outside our process <= 1 core-equivalent per window, and 1-min
   load average <= 4.0).
4. **CUDA-12 fallback** (section 3): if the MLX CUDA-12 smoke test
   fails on sm_103, may `mlx-cuda-13==0.31.1` replace
   `mlx-cuda-12==0.31.1`? Proposed: yes, pre-declared here, recorded
   in the run log; otherwise "not measured".
5. **Energy-instrument rules** (section 4.1): accept the
   power-integration fallback if the energy counter is unsupported or
   fails its checks (proposed: yes, labelled), and the counter/power
   consistency tolerance (proposed: 10 % over the 60-s check).
6. **Code delta** (section 6): who writes it, who reviews it, in which
   PR (proposed: this PR, before the freeze).
7. **Checkpoint transfer source** (section 2): copy of the checkpoints,
   splits and `selection.json` taken from which host, and when (not
   tonight; KXKM, macM1 and macM3 are not touched tonight). SHA-256
   checked on arrival.
8. **Python, uv and CUDA headers**: exact CPython 3.12.x build, uv
   version and the source of the CUDA headers for `CUDA_HOME`
   (`hitl-packages.md`), and the second review of the resolved set.
9. **Exploratory precision blocks** (parent r2 section 2): run TF32 /
   fp16 after the primary run, or not. Proposed: not run.
10. **Non-discriminative guard** (section 4.2): if P_load <= T, proceed
    (each pair flagged) or stop. Proposed: proceed, flagged, reported.
11. **Power-cap and MIG policy** (sections 4.1, 4.3): proposed: MIG
    must be disabled (else "not measured"); caps are logged, not
    changed; a pair during which the GB300 enforced cap or the
    sloshing mode changes is rejected and replayed.
12. **Thermal stabilisation** (section 4.2): duration of the
    unmeasured dense load before P_rest. Proposed: 300 s.
13. **Non-positive net energies** (section 5): rule as written in
    section 5 (proposed) or another.
14. **Descriptive summary schema** (section 6 item 6): field list of
    the GB300 summary without decision or catalogue fields.

## 1. What is identical to the KXKM / M3 Pro protocol

Everything below is inherited unchanged from the parent preregistration
`docs/prereg/2026-10-25-nh81-joules-per-token.md` (tag
`prereg-nh81-20261025`), the KXKM NVML replication r2/r3/r4
(`docs/prereg/2026-10-28-nh81-kxkm-replication.md`,
`2026-10-29-nh81-kxkm-replication-r3.md`,
`2026-10-30-nh81-kxkm-amendment-r4.md`, tags
`prereg-nh81-kxkm-20261028-r2`, `-20261029-r3`, `-20261030-r4`) and the
M3 Pro replication (`2026-10-30-nh81-m3pro-replication.md`, tag
`prereg-nh81-m3pro-20261030`), all on branch `exp/nh81-kxkm-nvml`,
unless section 4 or 5 replaces it for this host.

* **Models and data:** dense `ann1` and spiking `spkt`, step 12 000,
  SHA-256 `3c0891fbb528f122b5ac8179b180fab879d38415e385c29b7acb7be7ef12afd7`
  and `e9a3c9294c4737eddf019fa2ace51f2fe967c942d89039fcbd2870aa26eaa074`;
  `test.i32` `ad59a2f13afab4f47983561f6f12dd269e31076e4a411a8e0dd071c1021cfd2d`,
  `val.i32` `bc7038948319eaed68c7ee45e61b950bda706cd6bd4d2e3109fd16687d94257a`;
  macM1 checkpoint selection `results/nh81/selection.json` (on `main`),
  SHA-256 `d51c35fda2f75330b2d200a51e94f55b631b4592de1fcd2ce01bfaf170d8f52d`,
  required by `isoquality.py --macm1` and `analyze.py stats --selection`.
  No training on the GB300.
* **Inputs:** `experiments/nh81_joules/prompts_ids.json` (token ids,
  no `tokenizers` needed), as on KXKM.
* **Code:** `experiments/nh81_joules/` at tag
  `prereg-nh81-m3pro-20261030` (commit `4b4dafd`, git tree of the
  directory `d1b67e6ea1f1d6c7064ac8a9bea59152de00101c`), plus only the
  GB300 delta of section 6. Model module, `mx.compile` forward,
  workloads (W1 generation, W2 scoring), spiking paths (fused primary,
  event secondary), block order (W1 fused, W2 fused, W1 event,
  W2 event), unmeasured warm-up, calibration and statistics
  (geometric-mean R, percentile bootstrap, 10 000 resamples) are
  unchanged.
* **Order of operations in `measure.py` (unchanged):** start guard
  first, then the energy meter, model loading, unmeasured warm-up and
  calibration. The guard therefore runs with no model of ours loaded.
* **Calibration (as coded):** one timed dense W1 pass and one timed
  run of 8 dense W2 batches; K = ceil(15 s / pass time) (at least 1)
  and the W2 batch count = ceil(15 s / batch time) (at least 8), used
  for both models. This extrapolates a single timing; it is what ran on
  macM1, KXKM and M3 Pro.
* **Pairs:** 12 valid interleaved pairs per block, 4 blocks = 48 valid
  pairs; 20 s rest window before each pair and one after the last;
  order alternates D-S / S-D following the count of valid pairs;
  rejected pairs are logged, excluded and replayed, at most 24 attempts
  per block; a block with < 10 valid pairs is "fewer than 10 valid
  pairs" and is reported as such.
* **Seeds:** the measurement has no stochastic step (greedy decoding,
  fixed windows). Seeds are those of the frozen code: held-out loss
  windows seeds 1 and 2 (`train.EVAL_SEED`), W2 scoring windows seed 3
  (`measure.Workload`), bootstrap seed 0. No new seed namespace.
* **Precision:** strict fp32, `MLX_ENABLE_TF32=0`; the TF32 check of
  KXKM r2 section 2 (512x1536 @ 1536x384 product, relative error with
  and without the flag) is repeated on the GB300 before the first pair
  and logged.
* **Iso-quality and reproduction check:** val/test losses of both
  checkpoints recomputed on the GB300 (`isoquality.py`); each within
  0.005 nats of its macM1 value, else "not replicable as specified" and
  no energy pair is run; the 0.02-nat iso-quality rule is reported.
* **Memory cap:** 6 GB (`mx.set_memory_limit`).
* **Net energy:** NVML energy over the run minus (mean power of the
  preceding rest window) x run duration, divided by tokens; gross ratio
  and fixed-reference ratio (median rest-window power) reported as in
  KXKM r3 section 3.
* **CPU during runs:** as in the frozen code, CPU load during the runs
  themselves is logged but does not reject a pair; the rest windows and
  the start guard carry the CPU rule (section 4.4).

## 2. Host

DGX Station GB300 (one slot of 24 h on 2026-11-01): one Blackwell
Ultra GPU (compute capability 10.3, up to 252 GB HBM3e), one Grace CPU
(72 Neoverse V2 cores), 496 GB LPDDR5X, Ubuntu 24.04 aarch64 with a
64 KB page size, CUDA 13.1 / 13.2 validated stack, a 1 600 W system
budget shared by dynamic power sloshing between the GB300 module
(GB110) and the additional RTX display card, which has priority.
Sources: nvidia.com DGX Station product page and the DGX Station
Development Guide (overview, software requirements, page size, GPU
selection, dynamic power sloshing), opened 2026-10-31.

Recorded at the start of the slot (no assumption): `nvidia-smi -q`,
driver and maximum CUDA version, `uname -r`, `getconf PAGESIZE`, the
NVML device list (index, name, UUID, PCI bus ID, MIG mode), the
enforced power limits of both GPUs, the GB110 cap and sloshing mode as
reported by the platform, CPU model, `/proc/loadavg`, and the full
process list. No service is stopped, killed or reconfigured; clocks
and power limits are not changed.

## 3. Software

A dedicated virtual environment, outside the repository lockfile,
created only from the reviewed resolved package set
(`hitl-packages.md`): CPython 3.12 (uv-managed, item 8), `mlx==0.31.1`,
`mlx-cuda-12==0.31.1` with its NVIDIA runtime dependencies, and
`numpy==2.2.6` (the repository lock version). No `pynvml`: NVML is
called through `ctypes` on the driver's `libnvidia-ml.so.1`, as on KXKM.
`CUDA_HOME` points to the CUDA headers needed by MLX's JIT kernels
(same requirement as KXKM; source under item 8).

Smoke test before anything else (not an energy measurement):
`import mlx.core as mx; mx.eval(mx.ones(4) @ mx.ones(4))` on the
selected GPU. `mlx-cuda-12` 0.31.1 pins `nvidia-cuda-nvrtc-cu12==12.9.*`;
CUDA 12.9 is the first toolkit with `sm_103` targets. If the smoke test
fails, the pre-declared fallback is `mlx-cuda-13==0.31.1` (item 4); the
change is logged. Any other version change is a deviation and needs a
written amendment before the first pair.

## 4. Instrument, GPU selection and guards (GB300-specific)

### 4.1 Instrument

* **Primary:** `nvmlDeviceGetTotalEnergyConsumption` (mJ, cumulative)
  on the GB300 handle, read right before and right after each window,
  as on KXKM.
* **Counter qualification** (before the probes, read-only, 60 s): the
  call returns success at every read; the value never decreases; and
  the counter-derived mean power over the 60 s agrees with the mean of
  `nvmlDeviceGetPowerUsage` sampled every 0.5 s within the tolerance of
  item 5. Its scope (GPU only or module) is not documented for this
  product: it is recorded as "NVML total energy, scope as reported by
  the driver"; if NVML power field values with a GPU / module scope
  exist, both are logged (descriptive). Whether Grace is included is
  neither assumed nor excluded.
* **Pre-declared fallback (item 5):** if the counter is unsupported or
  fails qualification, every energy figure of the run (probes, rest
  windows, runs) is the trapezoidal integral of `nvmlDeviceGetPowerUsage`
  (mW; documented as a 1-s average since Ampere), sampled every 0.1 s
  by a background thread, with the device synchronised (`mx.eval`) at
  window boundaries. The run is labelled "power-integrated"; since each
  reading is a 1-s average, sub-second resolution is not claimed and
  windows stay >= 15 s for runs and 20 s for rest.
* **During the run:** a counter read that fails, decreases or stalls
  (no increase over a whole run) invalidates that window (fail closed);
  under the fallback, a failed power read or a gap of more than 1 s
  between successful samples invalidates the window likewise. Three
  such events in the run end it ("instrument failure").
* **Power sharing:** both GPUs' power and enforced caps, and the GB110
  cap / sloshing mode, are logged per window; the item-11 policy
  applies to changes. Grace CPU or module power, if exposed (hwmon,
  power capping), is logged, descriptive only.

### 4.2 Rest guard recomputed for the GB300 (no fixed watt figure)

The 90 W threshold of KXKM r3 was specific to the RTX 4090 (P2-idle
with our context at most 76 W). No watt figure is assumed for the
GB300; the threshold is derived on the day by a fixed rule, in this
order:

1. **Exclusivity check first:** the start-guard conditions of
   section 4.4 that do not depend on T (no compute process on the
   GB300 outside the allowed list, CPU rule, MIG off) must hold for
   2 min before any probe; otherwise probes are not taken.
2. **Probe P0** (read-only, no CUDA context of ours): 120 s, power
   every 0.5 s and energy with the instrument of 4.1. Descriptive.
3. **Probe P1** (separate unmeasured process): loads both checkpoints,
   runs the unmeasured warm-up, then dense W1 passes continuously for
   the stabilisation time of item 12, then stays alive with the models
   loaded and nothing running for 60 s, then exits. P_rest = the
   largest mean power over any 20-s sliding window (1-s steps) of
   those 60 s; P_load = mean power over the last 60 s of the dense
   load. Energy from the instrument of 4.1. The exclusivity conditions
   are checked throughout P1; a violation voids P1 and it is retried
   once after a clean 2 min, else "not measured". P1 numbers are never
   used in the analysis.
4. **Threshold:** T = k x P_rest (item 1; the 4090 ratio was
   90 / 76 = 1.18), written to `rest_threshold.json` and to the run log
   before the start guard, fixed for the whole run, never re-adjusted.
5. **Discriminability:** if P_load <= T the guard cannot separate a
   foreign load of the size of ours from rest (item 10). Even with
   P_load > T, a foreign load smaller than P_load - P_rest may go
   undetected by power; the process rules of 4.4 are the main
   protection, and this limit is stated in the note.
6. **Drift (descriptive):** mean rest-window power is reported per
   block, and a 60-s read-only probe is repeated after the last pair.

T replaces the 90 W figure everywhere r3 used it, **in acquisition and
in analysis**: rest windows (window mean power <= T), start guard
(every 5-s sample < T), pause rule (instantaneous power >= T pauses).
Because the guard of `measure.py` runs before model loading, the
start guard compares a no-model state with a T derived from a
context-held state; this makes the start guard looser, not stricter,
and is accepted.

### 4.3 GPU selection by PCI bus ID

The display card changes the CUDA order relative to NVML (DGX Station
Development Guide, GPU selection). The GB300 is never assumed to be
device 0:

* The operator writes the GB300 PCI bus ID (from `nvidia-smi -q`) in
  `NH81_GPU_PCI_BUS_ID`. The launcher resolves its UUID through NVML
  and exports `CUDA_VISIBLE_DEVICES=<UUID>` before any Python process
  initialises CUDA; every process of the study inherits it.
* In each process, before the first window: the NVML handle is opened
  with `nvmlDeviceGetHandleByPciBusId_v2`; MLX's default device must be
  the GPU; the CUDA driver API (`libcuda.so.1` via `ctypes`:
  `cuDeviceGetCount`, `cuDeviceGetPCIBusId`, `cuDeviceGetUuid`) must
  show exactly one device whose PCI bus ID and UUID equal the NVML
  ones. Any mismatch aborts ("GPU selection mismatch").
* MIG must be disabled on the GB300 (item 11), else "not measured".
* Compute processes, power and energy are read on that handle only.
  The `nvidia-smi` fallback of `gpu_compute_pids` is restricted to that
  GPU (`--id=<UUID>`); a failed process query fails closed (the window
  is rejected).

### 4.4 Exclusivity in the slot

Nothing else runs on the GB300 GPU and no other heavy job runs on the
Grace CPU during the NH-81 window (shared power budget, shared memory
system). Replacing the KXKM baseline rule (r2 "baseline B"):

* **Allowed GPU processes:** only those of the item-3 allowed list,
  identified by name and PID in `host.json` at the start; any other
  compute or graphics process on the GB300 rejects the window. B is no
  longer "whatever is present when the guard passes".
* **CPU rule (item 3):** no process other than ours above 20 % CPU
  averaged over a window (all processes, not only Python), and the
  1-min load average under the item-3 bound, at the start guard and in
  every rest window. Per-process CPU is computed from `/proc` ticks at
  both ends of the window, a process that appears during the window
  counting from zero ticks; in addition, host CPU time outside our
  process (from `/proc/stat`) must stay at or below the equivalent of
  one core over the window (item 3). A process that both starts and
  ends inside a window is caught only by this host-level figure; this
  limit is stated.
* **Pause rule:** r3 pause rule with the slot's other studies in place
  of the N10-C rule; the start guard is re-checked every 10 min within
  the item-2 window; cumulative pause within the item-2 budget.
  Exceeding either, or the global deadline, ends the run; complete
  blocks are reported, the run is not relaunched without a PI decision.

## 5. Status and reporting (descriptive only)

* Reported for the four blocks: valid pairs, dense and spiking mJ/token,
  R-hat with its 95 % bootstrap CI, paired difference, tokens/s, mean
  run and rest power, T, P0, P_rest, P_load, the instrument label, the
  selected GPU identity, all rejected pairs and their reasons.
* The bootstrap CI describes the spread of the pairs of this run on
  this machine; it is not a generalisation to other machines, and no
  block is singled out after seeing the results (all four are reported
  in the fixed order).
* **Non-positive net energies:** every valid pair is kept with its
  gross and net values. If any valid pair of a block has a net energy
  <= 0 for either model, the block's net R-hat and CI are reported as
  "undefined (non-positive net energy)"; the gross ratio of the block is
  reported as usual. No pair is dropped for this reason.
* **No verdict.** The GB300 summary is written by a descriptive
  summary function (section 6 item 6) without the `decision` and
  `catalogue` fields of `analyze.py stats`; those words do not appear
  in the note, the ERRATA or issue #14 for this run.
* **No inferential comparison between hosts.** A table places the
  GB300 R-hat next to the macM1, M3 Pro and (if available) KXKM values,
  with each instrument named; no test, no pooled estimate, no
  cross-host criterion is computed. Any such analysis would need its
  own preregistration.
* The NH-81 status ("withdraw, provisional (IOReport)", PI rule of
  KXKM r2 section 5) is unchanged by any GB300 outcome.

## 6. Code delta required before the freeze (not written tonight)

Additions only; models, workloads, calibration and statistics do not
change. Each item gets a unit test that runs without a GPU, including
boundary tests of T (window power equal to T passes, above fails).

1. `NH81_HOST=gb300` in `measure.py`, reusing `power_nvml` with the
   PCI-bus-ID handle, the in-process CUDA/NVML identity check, MIG
   check and the restricted `nvidia-smi` fallback of 4.3.
2. T read from `rest_threshold.json`, stored in every rest-window
   record (`rest_max_w`, `rest_rule: "gb300"`), and used by a
   `rest_ok` branch of `analyze.py` that reads it from the record, as
   the existing M3 Pro branch does, so acquisition and re-analysis use
   the same T (instead of the constant `POWER_REST_MAX_W = 90.0`).
3. `rest_probe_gb300.py`: exclusivity check, P0, P1 of 4.2, writes
   `rest_threshold.json` (P0, P_rest, P_load, k, T, instrument).
4. Counter qualification, failure handling and the power-integration
   fallback of 4.1; per-window logging of both GPUs' power and caps.
5. Exclusivity of 4.4 (allowed list, all-process CPU rule, load
   average), item-2 budgets and deadline; `gb300_launcher.sh` with the
   SHA-256 checks of section 1, environment (`MLX_ENABLE_TF32=0`,
   `NH81_HOST=gb300`, `CUDA_VISIBLE_DEVICES`, `NH81_PROMPT_IDS`,
   `CUDA_HOME`), smoke test, TF32 check, `isoquality.py`, rest probe,
   measurement, final idle probe; refuses to overwrite an existing
   `measure.jsonl`.
6. A descriptive summary writer (`analyze.py describe`) applying
   section 5 (no `decision` / `catalogue`, the non-positive-net rule).

## 7. Artifacts

`experiments/nh81_joules/` (delta of section 6),
`results/nh81-gb300/` (`host.json`, `rest_threshold.json`,
`isoquality.json`, raw `measure.jsonl`, `summary.json`), note
`docs/results/2026-11-01-nh81-gb300.md`. All artefacts are copied off
the GB300 before the slot ends.

## 8. Scope

One pair of 26.5 M models, one Blackwell Ultra GPU, dense CUDA kernels
via MLX, fp32. Nothing about neuromorphic silicon or sparse/event
kernels. The point is instrumentation (a third hardware family for the
same protocol), not a test of NH-81.

## Annex: Codex review (2026-10-31)

Blind review by Codex CLI (`gpt-6.1-sol`, reasoning effort high) of
commit `195c7df`. 2 BLOQUANT and 8 MAJEUR found, each checked in the
source and confirmed: T not propagated to `analyze.py` (90 W constant);
baseline B accepting any present process and an unfiltered
`nvidia-smi` fallback; probes exposed to contamination and no thermal
stabilisation; counter failure modes; power sloshing involves the RTX
card; selection check not specified in-process; text/code gaps (seed 3,
calibration, guard before loading, CPU during runs); `decision` and
`catalogue` fields; non-positive nets; second package review. All are
addressed above; one MINEUR (macM1 `selection.json`) also fixed.
A targeted second pass on `1e3633d` marked 8 of 10 resolved and 2
partial (transient CPU processes, fallback read failures); both are
now specified in sections 4.4 and 4.1.
