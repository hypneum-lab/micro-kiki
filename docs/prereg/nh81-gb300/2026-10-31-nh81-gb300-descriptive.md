# NH-81 on Blackwell (DGX Station GB300): descriptive third-hardware point

**Level:** L0 (draft on a branch, not frozen). It becomes L1 only after
every "TBD by PI" item below is closed, the code delta of section 6 is
committed and reviewed, the package list of `hitl-packages.md` is
reviewed, the PR is merged on green CI, and an annotated tag plus a
Software Heritage revision SWHID are recorded. No energy measurement
on the GB300 may start before that tag is pushed.
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
2. **Time budgets inside the slot** (section 4.4): start-guard window
   and cumulative pause budget. Proposed: 1 h each (3 h on KXKM/M3).
3. **Slot exclusivity list** (section 4.4): process-name patterns of
   the slot's other jobs that make the measurement pause (proposed:
   `run_campaign`, `run.sh`, `compare_nh01`, `procgen`, any
   `python` above 20 % CPU) and the slot hour when NH-81 runs alone.
4. **CUDA-12 fallback** (section 3): if the MLX CUDA-12 smoke test
   fails on sm_103, may `mlx-cuda-13==0.31.1` replace
   `mlx-cuda-12==0.31.1`? Proposed: yes, pre-declared here, recorded
   in the run log; otherwise "not measured".
5. **Energy-counter fallback** (section 4.1): if
   `nvmlDeviceGetTotalEnergyConsumption` is unsupported, may energy be
   integrated from `nvmlDeviceGetPowerUsage`? Proposed: yes, labelled
   "power-integrated"; otherwise "not measured".
6. **Code delta** (section 6): who writes it, who reviews it, in which
   PR (proposed: this PR, before the freeze).
7. **Checkpoint transfer source** (section 2): copy of the checkpoints
   and splits taken from which host, and when (not tonight; KXKM,
   macM1 and macM3 are not touched tonight). SHA-256 checked on arrival.
8. **Python interpreter**: exact CPython 3.12.x build (uv-managed) and
   uv version used on the GB300 (section 3, `hitl-packages.md`).
9. **Exploratory precision blocks** (parent r2 section 2): run TF32 /
   fp16 after the primary run, or not. Proposed: not run.
10. **Non-discriminative guard** (section 4.2): if the dense calibration
    power does not exceed T, proceed (each pair flagged) or stop.
    Proposed: proceed, flagged, reported.

## 1. What is identical to the KXKM / M3 Pro protocol

Everything below is inherited unchanged from the parent preregistration
`docs/prereg/2026-10-25-nh81-joules-per-token.md` (tag
`prereg-nh81-20261025`), the KXKM NVML replication r2/r3/r4
(`docs/prereg/2026-10-28-nh81-kxkm-replication.md`,
`2026-10-29-nh81-kxkm-replication-r3.md`,
`2026-10-30-nh81-kxkm-amendment-r4.md`, tags
`prereg-nh81-kxkm-20261028-r2`, `-20261029-r3`, `-20261030-r4`) and the
M3 Pro replication (`2026-10-30-nh81-m3pro-replication.md`, tag
`prereg-nh81-m3pro-20261030`), all on branch `exp/nh81-kxkm-nvml`.

* **Models and data:** dense `ann1` and spiking `spkt`, step 12 000,
  SHA-256 `3c0891fbb528f122b5ac8179b180fab879d38415e385c29b7acb7be7ef12afd7`
  and `e9a3c9294c4737eddf019fa2ace51f2fe967c942d89039fcbd2870aa26eaa074`;
  `test.i32` `ad59a2f13afab4f47983561f6f12dd269e31076e4a411a8e0dd071c1021cfd2d`,
  `val.i32` `bc7038948319eaed68c7ee45e61b950bda706cd6bd4d2e3109fd16687d94257a`.
  No training on the GB300.
* **Inputs:** `experiments/nh81_joules/prompts_ids.json` (token ids,
  no `tokenizers` needed), as on KXKM.
* **Code:** `experiments/nh81_joules/` at tag
  `prereg-nh81-m3pro-20261030` (commit `4b4dafd`, git tree of the
  directory `d1b67e6ea1f1d6c7064ac8a9bea59152de00101c`), plus only the
  GB300 host delta of section 6. Model module, `mx.compile` forward,
  workloads (W1 generation, W2 scoring), spiking paths (fused primary,
  event secondary), block order (W1 fused, W2 fused, W1 event,
  W2 event), unmeasured warm-up, calibration (smallest K / batch count
  giving >= 15 s on the dense model, measured on the GB300, same for
  both models), statistics (geometric-mean R, percentile bootstrap,
  10 000 resamples) are unchanged.
* **Pairs:** 12 valid interleaved pairs per block, 4 blocks = 48 valid
  pairs; 20 s rest window before each pair and one after the last;
  order alternates D-S / S-D following the count of valid pairs;
  rejected pairs are logged, excluded and replayed, at most 24 attempts
  per block; a block with < 10 valid pairs is "fewer than 10 valid
  pairs" and is reported as such.
* **Seeds:** the measurement has no stochastic step (greedy decoding,
  fixed windows). Seeds are those of the frozen code: held-out loss
  windows seeds 1 and 2 (`train.EVAL_SEED`), bootstrap seed 0. No new
  seed namespace is introduced.
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
* **Foreign-process and CPU rules:** baseline B of GPU compute
  processes frozen at the start of each guard check; any compute
  process outside B on the selected GPU rejects the pair; no other
  Python process above 20 % CPU averaged over each window (r4).

## 2. Host

DGX Station GB300 (one slot of 24 h on 2026-11-01): one Blackwell
Ultra GPU (compute capability 10.3, up to 252 GB HBM3e), one Grace CPU
(72 Neoverse V2 cores), 496 GB LPDDR5X, Ubuntu 24.04 aarch64 with a
64 KB page size, CUDA 13.1 / 13.2 validated stack, 1 600 W shared
between CPU and GPU (dynamic power sloshing). Sources:
nvidia.com DGX Station product page and the DGX Station Development
Guide (overview, software requirements, page size, GPU selection,
dynamic power sloshing), opened 2026-10-31.

Recorded at the start of the slot (no assumption): `nvidia-smi -q`,
driver and maximum CUDA version, `uname -r`, `getconf PAGESIZE`, the
NVML device list (index, name, UUID, PCI bus ID), the enforced power
limit and the GB300 power cap reported by the platform, CPU model,
`/proc/loadavg`, and the full process list. No service is stopped,
killed or reconfigured; clocks and power limits are not changed.

## 3. Software

A dedicated virtual environment, outside the repository lockfile,
created only from the reviewed package list (`hitl-packages.md`):
CPython 3.12 (uv-managed, exact build TBD by PI item 8), `mlx==0.31.1`,
`mlx-cuda-12==0.31.1` with its NVIDIA runtime dependencies, and
`numpy==2.2.6` (the repository lock version). No `pynvml`: NVML is
called through `ctypes` on the driver's `libnvidia-ml.so.1`, as on KXKM.
`CUDA_HOME` points to the CUDA headers needed by MLX's JIT kernels
(same requirement as KXKM).

Smoke test before anything else (not an energy measurement):
`import mlx.core as mx; mx.eval(mx.ones(4) @ mx.ones(4))` on the
selected GPU. `mlx-cuda-12` 0.31.1 pins `nvidia-cuda-nvrtc-cu12==12.9.*`;
CUDA 12.9 is the first toolkit with `sm_103` targets. If the smoke test
fails, the pre-declared fallback is `mlx-cuda-13==0.31.1` (TBD by PI
item 4); the change is logged. Any other version change is a deviation
and needs a written amendment before the first pair.

## 4. Instrument, GPU selection and guards (GB300-specific)

### 4.1 Instrument

* **Primary:** `nvmlDeviceGetTotalEnergyConsumption` (mJ, cumulative)
  on the GB300 handle, read right before and right after each window,
  as on KXKM. At the start of the slot, the counter is checked: the
  call returns success and the value increases over a 10-s read-only
  interval. Its scope (GPU only, or module) is not documented for this
  product; it is recorded as "NVML total energy, scope as reported by
  the driver", and, if the driver exposes NVML power field values with
  a GPU / module scope, both are logged (descriptive only).
* **Pre-declared fallback (TBD by PI item 5):** if the counter is not
  supported, window energy = trapezoidal integral of
  `nvmlDeviceGetPowerUsage` (mW; a 1-s average on this generation),
  sampled every 0.1 s by a background thread. The run is then labelled
  "power-integrated"; windows are >= 15 s, i.e. >= 15 averaging
  periods.
* Grace CPU power is not part of the NVML reading. If the platform
  exposes CPU or module power (hwmon / power-capping interfaces), it is
  logged per window, descriptive only. Because the 1 600 W budget is
  shared, the enforced GPU power cap and any change of it during the
  run are logged.

### 4.2 Rest guard recomputed for the GB300 (no fixed watt figure)

The 90 W threshold of KXKM r3 was specific to the RTX 4090 (P2-idle
with our context at most 76 W). No watt figure is assumed for the
GB300; the threshold is derived on the day, before the start guard, by
a fixed rule:

1. **Probe P0** (read-only, no CUDA context of ours, nothing loaded):
   120 s, NVML power every 0.5 s and energy counter. Reports P0_mean,
   P0_max (1-s values), utilisation, compute processes. Descriptive.
2. **Probe P1** (context-held rest): a separate unmeasured process
   loads both checkpoints, runs the unmeasured warm-up, one dense W1
   pass and 8 dense W2 batches, then stays alive with the models loaded
   and nothing running for 60 s. P_rest = the largest mean power over
   any 20-s sliding window of those 60 s (energy-counter differences,
   1-s steps); P_load = mean power over the dense W1 pass. Its numbers
   are never used in the analysis.
3. **Threshold:** T = k x P_rest, with k fixed by the PI (item 1,
   proposed 1.25; the 4090 ratio was 90 / 76 = 1.18). T is written in
   the run log before the start guard and stays fixed for the run.
4. **Discriminability:** if P_load <= T, the guard cannot separate a
   foreign load of the size of ours from rest; the run proceeds or stops
   as decided under item 10, and every pair is flagged.

T then replaces the 90 W figure everywhere r3 used it: rest windows
(window mean power <= T), start guard (every 5-s sample < T), pause
rule (instantaneous power >= T pauses).

### 4.3 GPU selection by PCI bus ID

The display card, if present, changes the CUDA order relative to NVML
(DGX Station Development Guide, GPU selection). The GB300 is never
assumed to be device 0:

* The operator writes the GB300 PCI bus ID (from `nvidia-smi -q`) in
  `NH81_GPU_PCI_BUS_ID`. The NVML handle is opened with
  `nvmlDeviceGetHandleByPciBusId_v2`.
* `CUDA_VISIBLE_DEVICES` is set to the **UUID** of that NVML device, so
  MLX sees one device only.
* Before the first pair, the run checks that the NVML device name is a
  GB300 / Blackwell part and that the CUDA device MLX uses has the same
  UUID; any mismatch aborts the run ("GPU selection mismatch").
* Compute processes, power and energy are read on that handle only.

### 4.4 Exclusivity in the slot

Nothing else runs on the GB300 GPU and no other heavy job runs on the
Grace CPU during the NH-81 window (shared 1 600 W budget, shared memory
system). Enforced by: the operator schedules NH-81 alone (TBD by PI
item 3); the pause rule of r3 with the slot's process patterns in place
of the N10-C rule; the foreign-compute-process rule; the CPU rule. The
start guard is re-checked every 10 min within the window of item 2; the
cumulative pause budget is that of item 2. Exceeding either ends the
run, which is reported, not relaunched without a PI decision.

## 5. Status and reporting (descriptive only)

* Reported for the four blocks: valid pairs, dense and spiking mJ/token,
  R-hat with its 95 % bootstrap CI, paired difference, tokens/s, mean
  run and rest power, T, P0, P_rest, P_load, the instrument label
  (energy counter or power-integrated), and the selected GPU identity.
* **No verdict.** The parent decision rule is not applied as a verdict;
  the `decision` field that the unchanged `analyze.py` writes into
  `summary.json` is kept for code identity and is not quoted in the
  note, the ERRATA or issue #14 as a status.
* **No inferential comparison between hosts.** A table places the
  GB300 R-hat next to the macM1, M3 Pro and (if available) KXKM values,
  with each instrument named; no test, no pooled estimate, no
  cross-host criterion is computed. Any such analysis would need its
  own preregistration.
* The NH-81 status ("withdraw, provisional (IOReport)", PI rule of
  KXKM r2 section 5) is unchanged by any GB300 outcome.

## 6. Code delta required before the freeze (not written tonight)

Additions only; nothing in the model, workloads, statistics or analysis
changes. Each item gets a unit test that runs without a GPU.

1. `NH81_HOST=gb300` in `measure.py`, reusing `power_nvml` with:
   handle by PCI bus ID (`NH81_GPU_PCI_BUS_ID`), UUID check of
   section 4.3, threshold T read from the rest-probe output instead of
   the constant `POWER_REST_MAX_W = 90.0`.
2. `rest_probe_gb300.py`: probes P0 and P1 of section 4.2, writes
   `rest_threshold.json` (P0, P_rest, P_load, k, T).
3. Energy-counter capability check and the pre-declared
   power-integration fallback of section 4.1.
4. Pause/exclusivity patterns of section 4.4 instead of the N10-C
   directory rule; time budgets of item 2.
5. `gb300_launcher.sh`: SHA-256 checks of section 1, environment
   (`MLX_ENABLE_TF32=0`, `NH81_HOST=gb300`, `CUDA_VISIBLE_DEVICES`,
   `NH81_PROMPT_IDS`, `CUDA_HOME`), smoke test, TF32 check,
   `isoquality.py`, rest probe, measurement; refuses to overwrite an
   existing `measure.jsonl`.

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
