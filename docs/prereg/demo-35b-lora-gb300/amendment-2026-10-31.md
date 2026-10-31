# 35B demo GB300 amendment, 2026-10-31 (before any data)

Demo, not a scientific claim.

**Amends:** `docs/prereg/demo-35b-lora-gb300/2026-10-31-demo-35b-lora-gb300.md`,
frozen by tag `demo-35b-gb300-freeze-20261031` (merge commit `4b17d76`,
SWHID `swh:1:rev:4b17d76284aa5ba0fca166de021d77d2843e80bb`). That tag
is kept; this amendment is frozen by the new tag
`demo-35b-gb300-freeze-20261031b`. It goes with the NH-81 GB300
amendment of the same date (shared instrument code).
**PI decision (2026-10-31, 23:02 CET):** amend and re-tag tonight.
**Data:** none. Nothing has been downloaded or run on the GB300; the
slot starts on 2026-11-01. This is a pre-data amendment.

## Reason

A Codex check of the code fixes (effort medium, after the first
freeze) marked findings 2, 3, 8 and 9 as partial and found one
regression (an exception in the P1 watcher thread could leave P1
accepted without surveillance).

## Changes

* `run_demo.py`, `probe_before_model`: P0 calls `sampler.clean()` and
  the demo stops ("not run") on an instrument fault (finding 2).
* `run_demo.py`, P1 and `window` watchers: an exception is recorded
  and treated as contamination; P1 is voided (one retry) and a run is
  rejected, never accepted (finding 3 and the regression).
* `demo_launcher.sh`, `fetch`: each `curl`, retries included, runs under
  `timeout` with the time left to the global 3-h deadline (finding 8).
* **Protocol change (finding 9):** a run shorter than 15 s is logged
  with `short=True` but is **not valid**: it does not count among the
  3 valid runs of its (domain, arm) and the attempt cap of 6 still
  applies. This replaces "kept, flagged `short` and reported
  separately" in section 4 of the demo prereg; `summarize` counts short
  runs separately, outside valid runs and medians.
* Tests: one light test per change in `tests/test_demo_35b_gb300.py`.
