# NH-81 replication preregistration: NVML on the KXKM RTX 4090

**Level:** L1 (frozen in git: commit + annotated tag
`prereg-nh81-kxkm-20261028`, pushed before any energy measurement on
KXKM).
**Date:** 2026-10-28.
**Issue:** hypneum-lab/micro-kiki#14 (NH-81).
**Parent:** `docs/prereg/2026-10-25-nh81-joules-per-token.md` (tag
`prereg-nh81-20261025`), section 8 ("A confirmatory, independent
measurement with NVML energy counters on the KXKM RTX 4090 ... with the
same code, models, workloads and criteria"), and amendment 1
(`2026-10-28-nh81-amendment-1.md`, A1: "the independent confirmation
remains the NVML measurement on the KXKM RTX 4090").
**PI decision:** go for the KXKM replication, 2026-10-28.
**macM1 result being replicated:** W1 fused R-hat = 0.91 [0.83, 1.01],
decision "withdraw" (`docs/results/2026-10-25-nh81-joules.md`).

Everything not stated below is inherited unchanged from the parent
preregistration: hypothesis and estimand (section 1), iso-quality rule
(section 3), workloads W1/W2 and spiking paths fused/event (section 4),
statistics (section 6), decision rule and catalogue reading (section 7).

## 1. What is identical

* **Models.** The two checkpoints measured on macM1, copied from macM1 and
  checked by SHA-256 before use:
  * dense `ann1` step 12 000:
    `3c0891fbb528f122b5ac8179b180fab879d38415e385c29b7acb7be7ef12afd7`
  * spiking `spkt` step 12 000:
    `e9a3c9294c4737eddf019fa2ace51f2fe967c942d89039fcbd2870aa26eaa074`
  * test split `test.i32` (W2 windows, iso-quality):
    `ad59a2f13afab4f47983561f6f12dd269e31076e4a411a8e0dd071c1021cfd2d`,
    val split `val.i32`:
    `bc7038948319eaed68c7ee45e61b950bda706cd6bd4d2e3109fd16687d94257a`.
  No training happens on KXKM.
* **Code.** `experiments/nh81_joules/model_mlx.py` unchanged (same module,
  same `mx.compile` forward); `measure.py` / `analyze.py` keep the same
  protocol, calibration, block order (W1 fused, W2 fused, W1 event,
  W2 event), statistics and decision rule. The only additions are a host
  module for Linux/NVML and the replay/pause logic of section 4 below.
* **Framework.** MLX **0.31.1** (same version as macM1) with its CUDA
  backend `mlx-cuda-12==0.31.1`, from the existing pinned venv
  `/ai/saisail/n10c-run/.venv` (lock `n10c-bench/requirements.lock`,
  Python 3.10.18), used read-only. No new dependency is installed (repo
  supply-chain policy). NVML is called through `ctypes` on the driver's
  `libnvidia-ml.so.1` (driver 595.84), so no `pynvml` either.
  `CUDA_HOME=/ai/saisail/n10c-bench/cuda_home` (headers for MLX's JIT
  kernels, as for N10-C).
* **Inputs.** `tokenizers` is not in that venv, so the three prompts of
  `prompts.json` are tokenised once on macM1 with the same BPE
  (`bsk_bpe.json`, SHA-256 `78bce0d3...0f4c`, tokenizers 0.22.2) and
  committed as `prompts_ids.json`; `measure.py` reads the ids instead of
  encoding. The token ids, hence the workload, are identical.
* **Pairs.** 12 interleaved pairs per block, 4 blocks = **48 valid pairs**,
  20 s rest window before each pair and one after the last, order
  alternating D-S / S-D; >= 10 valid pairs per block required.
* **Calibration.** Same rule: K (W1 passes) and the W2 batch count are the
  smallest values giving >= 15 s on the dense model, measured on KXKM,
  and used for both models.
* **Memory cap.** 6 GB (`mx.set_memory_limit`).

## 2. Precision

* **Primary: fp32**, as on macM1. On this CUDA backend MLX runs fp32
  matmuls in **TF32** by default (checked 2026-10-28 before this prereg:
  a 512x1536 @ 1536x384 product has relative error 3e-4 by default,
  4e-7 with `MLX_ENABLE_TF32=0`). The primary run therefore sets
  **`MLX_ENABLE_TF32=0`** (strict IEEE fp32, the macM1 arithmetic).
* Exploratory only (never part of the verdict, separate log, run only
  after the primary run if the GPU is still free): TF32 and/or fp16
  weights and activations, same protocol. If not run, the note says so.

## 3. Iso-quality on the GPU

Before the first pair, the val and test losses of both checkpoints are
recomputed on the RTX 4090 with the training evaluation code
(`train.held_out_loss`: 512 fixed windows, seeds 1 and 2, fp32).

* **Reproduction check:** each loss within **0.005 nats** of its macM1
  value (spiking test 3.6445, dense test 3.6372; val 3.5882 / 3.5884).
  A larger deviation means the CUDA arithmetic differs from macM1 enough
  to question "same models"; the measurement is then not started and the
  outcome is "not replicable as specified".
* **Iso-quality:** the parent rule (|L_test(D*) - L_test(S)| <= 0.02
  nats, D* = dense checkpoint closest in val loss; only the step-12 000
  checkpoints are available, as selected on macM1) is re-applied to the
  GPU losses, with the same restrictions on the verdict if it fails.
* The test firing rate (fraction of neuron-ticks emitting) is reported.

## 4. Energy instrument and guards

Host: KXKM-AI (RTX 4090 24 GB, 64 GB RAM, Ubuntu). No service is stopped,
killed or reconfigured: ComfyUI stays resident (about 650 MiB VRAM, idle
at ~1 % utilisation), the gpu-swap orchestrator (:18890) stays up and may
load a backend on demand; VoiceClone, docling and the display server stay
as they are. GPU clocks are not locked (that would reconfigure the GPU);
power, SM clock, temperature and P-state are logged instead.

* **Instrument:** `nvmlDeviceGetTotalEnergyConsumption` (cumulative board
  energy since driver load, mJ). Energy of a window = difference of two
  readings taken right before and right after the workload (same place
  where macM1 read IOReport). Whole-board energy (GPU + memory + board),
  so it is the GPU-rail analogue of the macM1 primary metric.
* **Net energy and J/token:** unchanged: energy over the run minus (mean
  board power in the preceding rest window) x run duration, divided by
  tokens. The idle board power of the 4090 (tens of watts) is much larger
  than the M1 rail at rest, so the gross (not rest-subtracted) ratio is
  also reported, descriptive only.
* **Logged with every window:** NVML GPU utilisation (every 0.5 s), the
  set of compute processes on the GPU (NVML), GPU memory used, board
  power, SM clock, temperature, P-state; other Python processes above
  20 % CPU measured over the window from `/proc` (on Linux `ps %cpu` is a
  lifetime average); N10-C status.
* **"GPU at rest" guard (rejection rule).** Baseline B = the compute
  processes on the GPU when the start guard passes (expected: ComfyUI
  only). A pair is **rejected** if, in the rest window before it or the
  rest window after it, mean GPU utilisation exceeds 10 % or another
  Python process exceeds 20 % CPU (parent rule), **or** if any compute
  process outside B (and other than the measurement process) is seen on
  the GPU in any window of the pair (rest before, both runs, rest after).
* **Replay.** A rejected pair is kept in the log, excluded from the
  statistics and **replayed**: the script continues until 12 valid pairs
  per block, with at most 24 attempts per block. The D-S / S-D order
  follows the count of valid pairs, so the valid pairs stay balanced.
* **Pause rule** (replaces the macM1 N10-C stop rule). Before every pair,
  if an N10-C job runs (`pgrep -f run_campaign`, or a file under
  `/ai/saisail/n10c-run/campaign/` modified in the last 10 min), or a
  compute process outside B is on the GPU, or GPU utilisation is >= 10 %,
  the measurement **pauses**: re-check every 60 s, resume after 2 min
  clean. More than 3 h of cumulative pause ends the run ("KXKM busy");
  the blocks with >= 10 valid pairs are analysed, the others are
  "inconclusive: fewer than 10 valid pairs".
* **Start guard:** as on macM1: 2 min with GPU utilisation < 10 %, no
  other Python process > 20 % CPU and no N10-C activity, re-checked every
  10 min for up to 3 h.
* **Idle reference:** besides the 52 rest windows of the run (plus those of replayed pairs), a 60 s
  read-only probe (no model loaded) is recorded right before the run.

## 5. Decision and replication reading

* The parent decision rule (section 7) is applied to the KXKM primary
  block (W1, fused path, NVML board energy) with no change: withdraw if
  R-hat <= 1.1; reopen if R-hat >= 2 and CI lower bound > 1.1;
  inconclusive otherwise or with < 10 valid pairs.
* **Replication reading:** the macM1 verdict ("withdraw") is
  **replicated** if the KXKM primary decision is also "withdraw";
  **contradicted** if it is "reopen"; **not replicated (inconclusive)**
  otherwise. R-hat of KXKM and of macM1 are reported side by side for the
  four blocks, with the ratio R_KXKM / R_macM1; this comparison is
  descriptive (different hardware, different instrument).
* A "contradicted" or "not replicated" outcome does not overwrite the
  macM1 result; both are reported and the ERRATA entry states the scope
  of each.

## 6. Scope

One pair of 26.5 M models, one GPU (RTX 4090, Ada), dense CUDA kernels
via MLX, fp32. Nothing is said about neuromorphic silicon or about
sparse/event kernels that would exploit spike sparsity.

## 7. Artifacts

`experiments/nh81_joules/` (code, `power_nvml.py`, `prompts_ids.json`),
`results/nh81-kxkm/` (`isoquality.json`, raw log `measure.jsonl`,
`idle_probe.json`, `summary.json`), note
`docs/results/2026-10-28-nh81-kxkm-nvml.md`.
