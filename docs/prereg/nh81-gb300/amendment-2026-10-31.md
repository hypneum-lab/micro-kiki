# NH-81 GB300 amendment, 2026-10-31 (before any data)

**Amends:** `docs/prereg/nh81-gb300/2026-10-31-nh81-gb300-descriptive.md`,
frozen by tag `nh81-gb300-freeze-20261031` (merge commit `0c99299`,
SWHID `swh:1:rev:0c9929906e7ce54bc0803dc49d8b96c7483e55d0`). That tag
is kept; this amendment is frozen by the new tag
`nh81-gb300-freeze-20261031b`.
**PI decision (2026-10-31, 23:02 CET):** amend and re-tag tonight.
**Data:** none. No probe, measurement or smoke test has run on the
GB300; the slot starts on 2026-11-01. This is a pre-data amendment.

**L2 deposit (annotation added 2026-11-08):** restricted Zenodo record
[10.5281/zenodo.24053794](https://doi.org/10.5281/zenodo.24053794) (Hypneum Lab lot of
2026-11-08, file `nh81-gb300-freeze-20261031b.zip`, SHA-256
`095353244acaa8e70d33c3132cbc8740d67c979df4a27e8bf8b2e76a68427724`) gives the freeze tag
`nh81-gb300-freeze-20261031b` (commit `185c5d44b50de76a73c9a2a45ef56c6e28fbcedc`,
`swh:1:rev:185c5d44b50de76a73c9a2a45ef56c6e28fbcedc`) a verifiable deposit date, level
L2 of PROJECT-MANAGEMENT section 4. The deposit holds this file as it was at the tag;
anything dated after the tag in this live file is not in the deposit.

## Reason

A Codex check of the code fixes (effort medium, after the first
freeze) marked four earlier findings as partial and found one
regression in the GB300 code (`experiments/nh81_gb300/`, shared with
the demo):

* integrator faults between two sampler reads were not seen by the
  rest-threshold probe (finding 2);
* a watcher thread that raised could leave a probe or run accepted
  without its checks (regression of finding 3);
* the final idle probe had no timeout (finding 8).

## Changes (code only; protocol text unchanged)

* `rest_probe_gb300.py`, `Sampler.run`: any fault counted since the
  previous read marks the interval faulty, so P0 / P1 with such a fault
  are void (prereg 4.1-4.2, as intended).
* `gb300_rules.py`, `window_ok_gb300`: a window whose watcher recorded
  an exception (`watch_errors`) is rejected.
* `gb300_launcher.sh`: the final probe runs under `timeout` (120 s,
  kill after 60 s more).
* Tests: one light test per change in `tests/test_nh81_gb300.py`.

Finding 4 (CPU during runs not rejecting) is unchanged: the frozen
protocol logs CPU during runs without rejecting.
