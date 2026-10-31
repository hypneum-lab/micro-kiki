# NH-81 GB300 freeze manifest (prepared 2026-10-31, NOT a freeze)

Prepared for the L1 freeze of `docs/prereg/nh81-gb300/`. **No tag, no
Software Heritage save yet:** the freeze waits for the PI review of the
resolved requirements. At freeze time: merge on green CI, recompute
this list on the merge commit, record the SHA-256 of this file in the
annotated tag message `nh81-gb300-freeze-YYYYMMDD`, then save to
Software Heritage and record the revision SWHID.

Frozen measurement code (unchanged, exported by the launcher):
`experiments/nh81_joules` at tag `prereg-nh81-m3pro-20261030`
(commit `4b4dafd`), git tree `d1b67e6ea1f1d6c7064ac8a9bea59152de00101c`.

Data (checked by the launcher): see the prereg, section 1.

## Files (SHA-256)

```
f0cbe25bf4ed1275f8701d539f4384b3f982dcbbb91430a2a018ee8a2efe381e  docs/prereg/nh81-gb300/2026-10-31-nh81-gb300-descriptive.md
3f64aa56b6fcaeb347e2d80d1ba199fd0f50ac3afe9f5cdf7932a88722e6edef  docs/prereg/nh81-gb300/hitl-packages.md
12c63efd5a255eb850f3e2822715e3e2ce6e7f165927df21590a4feeaecc5b59  docs/prereg/nh81-gb300/requirements-gb300-nh81-cuda13.in
a19e33bedd51357cddbbd67ee726cc2047e1ca0f8e4c0391dc745fdd829d2f17  docs/prereg/nh81-gb300/requirements-gb300-nh81-cuda13.txt
8a3e785b45303da590fdbe66e7b74ab42a991eca84f930b6b85b72f3a0e92c30  docs/prereg/nh81-gb300/requirements-gb300-nh81.in
1c45a829706116dc8b69c8657292f562dd2c2cc8b44f7e362d8f46431708480d  docs/prereg/nh81-gb300/requirements-gb300-nh81.sha256
45a261d231176a1fb5e22e66b4c3ad7566b61f3a9ed634f0060305d421557100  docs/prereg/nh81-gb300/requirements-gb300-nh81.txt
8edc34fff750cf36dcb59f400dc60d44ed93c88c25a3cc6dcdfe210490f4dde3  docs/prereg/nh81-gb300/requirements-review.md
4dbcb8214e77e644aef142c83a819d49805ce19f3b87ae817e24022f840b8f77  experiments/nh81_gb300/describe.py
075f51ccaa7ea4b644b6ddfd3971b782de42b7aa777949b5d60f2ccace946183  experiments/nh81_gb300/gb300_launcher.sh
02fc49e2d40555d9eb6d0a528914eb50c88f2c5a855f27c5b188354a60606cf9  experiments/nh81_gb300/gb300_rules.py
e867161debb881796d8734c26a98064b0ee9dbf4083830872c2ea6a3b89bbffc  experiments/nh81_gb300/measure_gb300.py
2d5fac190c894a7a817e4923004b6c4f402c64a78880dbc988cdfd9d71bff650  experiments/nh81_gb300/power_gb300.py
54d4d8cc2c7848f9cc8965ce5adcd5c16da9d9e7159e383bbcee9f034d7fd3e8  experiments/nh81_gb300/rest_probe_gb300.py
81056e24fe78258698f9e9c4ab3edf1fa40d99351a9516ec7161b31b5523ad2f  experiments/nh81_gb300/tf32_check.py
e712ea751c6ed835d07bf9aa8c728d1c1ceb3bbacf29d8890787171bd085fb77  tests/test_nh81_gb300.py
```
