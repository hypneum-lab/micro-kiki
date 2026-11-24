"""Smoke test for scripts/diag_aeon_centering_query_order.py (small stream)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_diag_aeon_centering_query_order_smoke(tmp_path):
    out = tmp_path / "diag.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "diag_aeon_centering_query_order.py"),
            "--dim", "16",
            "--n-turns", "60",
            "--n-queries", "10",
            "--cold-start", "4",
            "--shuffle-seeds", "1,2",
            "--out", str(out),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text())
    assert set(payload["conditions"]) == {"C", "D", "F1", "F2"}
    assert payload["n_queries"] == 10

    # Without centering there is no evaluation-time state: every variant of
    # condition C must give the same scores and the same per-query ranks.
    c = payload["conditions"]["C"]["variants"]
    flat = [c["as_run"], c["null_first"], c["reversed"], c["frozen"], *c["shuffled"].values()]
    for key in ("predictive_per_query", "null_per_query"):
        assert all(v[key] == flat[0][key] for v in flat), key
    # Means are summed in scoring order, so compare them up to rounding.
    for key in ("predictive_mrr", "null_stack_mrr"):
        assert all(abs(v[key] - flat[0][key]) < 1e-9 for v in flat), key

    for cond in payload["conditions"].values():
        assert cond["weights_all_finite"]
        for v in (cond["variants"]["as_run"], cond["variants"]["frozen"]):
            assert v["outputs_all_finite"]
            assert len(v["predictive_per_query"]["gold_ranks"]) == 10
            assert sum(v["predictive_per_query"]["second_place"].values()) == 10
    assert set(payload["conditions"]["D"]["no_mlp_ema"]) == {
        "init_post_train_mean_updates_1",
        "init_post_train_mean_updates_2",
        "init_zero_updates_1",
        "init_zero_updates_2",
    }
