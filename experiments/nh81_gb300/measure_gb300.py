"""Run the frozen NH-81 ``measure.py`` on the GB300 (prereg section 6).

The frozen tree (``experiments/nh81_joules`` at tag
``prereg-nh81-m3pro-20261030``, directory tree
``d1b67e6ea1f1d6c7064ac8a9bea59152de00101c``) is used unchanged from
``NH81_CODE_DIR``. This wrapper only:

1. checks the GPU identity in this process (PCI bus ID, UUID, MIG);
2. makes ``power_gb300`` the host module (``NH81_HOST=kxkm`` path with
   ``power_nvml`` replaced in ``sys.modules``);
3. replaces ``analyze.rest_ok`` / ``analyze.window_ok`` by the GB300
   rules, so acquisition uses T from ``rest_threshold.json``;
4. sets the start-guard and pause budgets of the PI decisions;
5. writes a first ``gb300_start`` record (the frozen code logs
   ``host=kxkm``, which on this run means "NVML path").

Usage: python measure_gb300.py <measure.py arguments>
"""

from __future__ import annotations

import functools
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.environ["NH81_CODE_DIR"]
sys.path[:0] = [HERE, CODE]

import gb300_rules as R  # noqa: E402
import power_gb300  # noqa: E402

ident = power_gb300.check_identity()
sys.modules["power_nvml"] = power_gb300
os.environ["NH81_HOST"] = "kxkm"

import analyze  # noqa: E402

analyze.rest_ok = R.rest_ok_gb300
analyze.window_ok = R.window_ok_gb300

import mlx.core as mx  # noqa: E402
import measure  # noqa: E402

if mx.default_device() != mx.gpu:
    raise SystemExit("MLX default device is not the GPU")
measure.MAX_PAUSE_S = R.PAUSE_BUDGET_S
measure.start_guard = functools.partial(measure.start_guard, max_wait_s=R.GUARD_BUDGET_S)


def _first_record(out: str) -> None:
    rec = {"kind": "gb300_start", "logged": time.time(), "host": "gb300", "gpu": ident,
           "rest_threshold": power_gb300.cfg(), "mlx": mx.__version__,
           "tf32": os.environ.get("MLX_ENABLE_TF32", "default"),
           "note": "frozen measure.py runs its NVML path (host=kxkm) with power_gb300"}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    out = sys.argv[sys.argv.index("--out") + 1]
    _first_record(out)
    sys.argv[0] = os.path.join(CODE, "measure.py")
    raise SystemExit(measure.main())
