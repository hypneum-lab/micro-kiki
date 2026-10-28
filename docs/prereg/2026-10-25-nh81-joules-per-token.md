# NH-81 preregistration: measured joules per token, spiking vs dense

**Level:** L1 (frozen in git: commit + annotated tag `prereg-nh81-20261025`,
pushed before any energy measurement).
**Date:** 2026-10-25.
**Issue:** hypneum-lab/micro-kiki#14 (NH-81). **PI decision:** "go NH-81" on
the macM1 GPU, 2026-10-25 19:00; protocol correction 19:04 (GPU rail as
primary metric, interleaved pairs, rejection rule, NVML follow-up).
**Context:** `docs/ERRATA.md` E-2026-10-25-01 withdrew "0.032 mJ/token" and
"91.6x", which came from an analytical operation count
(`scripts/eval_spikingkiki_35b.py`, `results/spikingkiki-35b-eval.json`,
`"status": "theoretical"`). This study replaces counting by measurement.

## 1. Hypothesis

Catalogue entry NH-81 (hypneum-lab/.github, review of 2026-10-24,
`synthese/02-catalogue-nouvelles-hypotheses.md`): *firing rate is not energy:
the measured J/token of a spiking LM is >= 0.9-1.1x that of its dense twin at
iso-quality on GPU/M1.* The analytical AC/MAC count predicts a >= 4x advantage
for the spiking model; NH-81 predicts <= 1.1x.

Estimand: **R = (J/token of the dense model) / (J/token of the spiking model)**
on the macM1 GPU rail. R > 1 means the spiking model uses less energy.

## 2. Model pair

Both models use the BSK architecture (hypneum-lab/bsk
`training/remote_train_lm.py`, commit 8cc7087), ported to MLX in
`experiments/nh81_joules/model_mlx.py` (one code path for training and
measurement, no weight conversion):

| | dense `ann1` | spiking `spkt` |
|---|---|---|
| Backbone | D=384, L=6, 6 heads, SEQ=128, LayerNorm, learned positions, untied head, vocab 16 000 | identical |
| FFN | `down(silu(gate h) * up h)`, width 4D | same, with signed rate-LIF on the FFN activation: T=6 ticks, binary spike at threshold 0.5, reset by 0.5, global scale mean abs, straight-through surrogate sigma(4x) |
| Training penalty | none | firing-rate L1, weight 0.2 (BSK `dev_marathon.sh` default `HL_SPIKE_L1=0.2`, the setting behind the 1-3 % firing claim) |

Parameter count is identical (about 27 M). No 35B model is involved.

**Data.** 64 French Project Gutenberg books (gutendex, `languages=fr`,
`sort=popular`, fetched 2026-10-25), ids and SHA-256 frozen in
`experiments/nh81_joules/corpus_manifest.json`; Gutenberg header/footer
stripped; BSK BPE `tokenizer/bsk_bpe.json` (16 000 tokens, SHA-256
`78bce0d3...0f4c`). Token stream cut into blocks of 4096 tokens: block i is
val if i mod 20 = 18, test if i mod 20 = 19, train otherwise
(11.80 M / 0.66 M / 0.66 M tokens; split SHA-256 in `split_info.json`).

**Training.** MLX, fp32, AdamW (lr 5e-4 constant, betas 0.9/0.999, no weight
decay), batch 16 x 128 tokens, 12 000 steps, seed 0, identical batch
sequence for both arms; checkpoint and val/test loss every 250 steps
(512 fixed windows per split, seeds 1 and 2).

Deviation from the brief's preference (training on electron-server CPU):
electron-server was saturated at the time (load average ~20 on 8 cores,
~30 000 zombie tasks; a 50-step timing smoke did not finish in 10 min), so
both arms are trained on the macM1 GPU under MLX (nice 19, memory cap 6 GB,
peak observed in a smoke run 2.3 GB). Training is not an energy measurement
and is not subject to the idle-GPU guard.

## 3. Iso-quality

Quality = held-out cross-entropy in nats/token.

* Spiking model S = final `spkt` checkpoint (step 12 000).
* Dense comparator D* = the `ann1` checkpoint (same architecture, same
  data stream) whose **val** loss is closest to S's val loss.
* Iso-quality holds if **|L_test(D*) - L_test(S)| <= 0.02 nats**
  (perplexity within about 2 %, the catalogue's PPL +/- 2 %).

Energy of a fixed architecture does not depend on which checkpoint is
loaded except through second-order data-dependent switching, so choosing D*
among same-size checkpoints matches quality without changing the cost
structure. The dense final checkpoint and its loss gap to S are reported as
well (the 2610.15146 reference puts the spiking quality cost at 0.23-0.34
nats at equal FLOPs).

If iso-quality fails:
* S worse than D* by more than 0.02 nats: the comparison favours the
  spiking model (a smaller dense model would match it); a "withdraw" verdict
  stays valid, a "reopen" verdict is not allowed (-> inconclusive).
* S better than D* by more than 0.02 nats: the comparison favours the dense
  model; "reopen" stays valid, "withdraw" is not allowed (-> inconclusive).

## 4. Workloads

All runs use MLX on the Metal GPU, fp32, `mx.compile` for every forward,
after an unmeasured warm-up that visits every input shape.

* **W1 (primary): generation.** Batch 1, greedy, three fixed French prompts
  (`prompts.json`), 256 new tokens each, sliding window of the last <= 128
  tokens recomputed per step (the BSK generation procedure). One *run* =
  K passes over the three prompts; K is fixed by a calibration run of the
  dense model before the first pair (smallest K giving >= 15 s), and the same
  K is used for both models.
* **W2 (secondary): batched scoring.** Forward pass over test windows,
  batch 32 x 128, fixed number of batches per run (calibrated the same way).
* Spiking inference path: **fused** (BSK reference: `down` applied to the
  spike rate) is primary because it is the cheapest exact implementation on
  this hardware, i.e. the most favourable to the spiking model. **Event**
  (`down` applied to each tick's spike vector, T results averaged; equal
  output) is secondary.

## 5. Energy measurement

Host: macM1 (M1 Max, 32 GB). The OrbStack VM `srv` (production, about three
CPU cores in continuous use) is never stopped; it loads the CPU cores and
makes CPU/package power unreliable, so:

* **Primary metric: GPU rail energy** (`powermetrics --samplers gpu_power`;
  ANE rail recorded and added if non-zero; the models do not use the ANE).
  CPU and package power are descriptive only.
* Instrument. `powermetrics` needs sudo, which is not available to the
  agent (`sudo -n true` fails). The PI starts the logger:
  `sudo powermetrics --samplers gpu_power,cpu_power,ane_power -i 100 -f plist -o ~/scratch/nh81/pm.plist`
  and the measurement script records the wall-clock bounds of every window.
  Alternate instrument, always recorded: the IOReport "Energy Model"
  counters read sudo-less (`power.py`; the same hardware counters
  `powermetrics` reads; channel `GPU Energy`, cumulative, nJ). If the
  powermetrics log is absent, the analysis uses IOReport and labels the
  verdict **provisional (IOReport)** until a powermetrics replication by the
  PI. If both are present, powermetrics is primary and the IOReport
  agreement is reported.
* Net energy of a run = GPU energy over the run - (mean GPU power in the
  preceding rest window) x run duration. J/token = net energy / tokens
  generated (W1) or scored (W2). Tokens/s are recorded.
* **Interleaved pairs.** One pair = 20 s rest window (measured, models
  loaded, nothing running) + one dense run + one spiking run; the order
  alternates between pairs (D-S, S-D, D-S, ...: interleaved A/B with order
  counterbalanced). At least **10 valid pairs** per workload; the script
  targets 12.
* Logged with every window: GPU utilisation from `ioreg` ("Device
  Utilization %", sampled every 0.5 s), the OrbStack VM's %CPU, the other
  Python processes above 20 % CPU, and the N10-C status.
* **Rejection rule.** A pair is rejected if, in the rest window before it
  or the rest window after it, the mean background GPU utilisation exceeds
  10 %, or any other Python process exceeds 20 % CPU. Rejected pairs are kept
  in the log and excluded from the statistics.
* **Start guard.** Measurement starts only after 2 min with GPU utilisation
  < 10 % and no other Python process > 20 % CPU, re-checked every 10 min for
  up to 3 h; otherwise the outcome is "macM1 busy, not measured".
* **N10-C stop rule.** Before every pair: `pgrep -f run_campaign` must be
  empty and no file under `~/scratch/n10c-run/campaign/` may have been
  modified in the last 10 min; otherwise the measurement stops at once.
* Memory: the measurement process caps MLX memory at 6 GB.

## 6. Statistics

Per valid pair i: r_i = (J/token dense)_i / (J/token spiking)_i.
R-hat = geometric mean of r_i; 95 % CI by percentile bootstrap over pairs
(10 000 resamples, seed 0). Also reported: mean paired difference of J/token
with its bootstrap CI, tokens/s, spike rate on the test set (fraction of
neuron-ticks emitting), and an analytical memory-traffic estimate (bytes of
weights and LIF state touched per token; labelled analytical).

## 7. Decision rule (primary: W1, fused path, GPU rail)

* **Withdraw the energy argument definitively** (for GPU/M1-class hardware)
  if R-hat <= 1.1.
* **Reopen** if R-hat >= 2 and the 95 % CI lower bound > 1.1.
* **Inconclusive** otherwise (1.1 < R-hat < 2, or R-hat >= 2 with the CI
  reaching 1.1), or with fewer than 10 valid pairs, or under the
  iso-quality restrictions of section 3.
* Catalogue reading: NH-81 is supported if R-hat <= 1.1 and refuted if the
  spiking model uses < 0.8x the dense energy (R-hat >= 1.25 with CI lower
  bound > 1.25).

W2 and the event path are secondary and cannot change the verdict; they
are reported with the same statistics.

## 8. Scope and follow-up

The verdict covers one pair (27 M parameters, BSK recipe) on the M1 Max GPU
with dense kernels. It says nothing about neuromorphic silicon, where
spike sparsity can be exploited. A confirmatory, independent measurement
with NVML energy counters on the KXKM RTX 4090 is planned once N10-C frees
the GPU, with the same code, models, workloads and criteria (pairs,
rejection rule, decision rule); it is outside the present run.

## 9. Artifacts

`experiments/nh81_joules/` (code, `corpus_manifest.json`, `prompts.json`),
`results/nh81/` (training logs, checkpoint selection, raw measurement log
`measure.jsonl`, summary `summary.json`), note
`docs/results/2026-10-25-nh81-joules.md`.
