"""Smoke test the latent-dim ablation script on a tiny synthetic corpus."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pennylane")
pytest.importorskip("torch")


def test_ablation_runs(tmp_path: Path) -> None:
    # Tiny 2-domain synthetic corpus.
    data = tmp_path / "data" / "final"
    for dom in ["dsp", "electronics"]:
        (data / dom).mkdir(parents=True)
        lines = [
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"{dom} Q {i}"},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            )
            for i in range(16)
        ]
        (data / dom / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_json = tmp_path / "ablation.json"
    ckpt_dir = tmp_path / "ckpts"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/ablate_text_jepa_dim.py",
            "--data-dir", str(data),
            "--domains", "dsp,electronics",
            "--max-per-domain", "16",
            "--latent-dims", "4,8",
            "--train-epochs", "1",
            "--vqc-epochs", "1",
            "--batch-size", "4",
            "--output", str(out_json),
            "--backbone", "random",
            "--seq-len", "4",
            "--input-dim", "16",
            "--hidden-dim", "16",
            "--checkpoints-dir", str(ckpt_dir),
        ],
        capture_output=True,
        text=True,
        cwd="/Users/electron/Documents/Projets/micro-kiki-poc-textjepa",
    )
    assert result.returncode == 0, f"stderr={result.stderr}\nstdout={result.stdout}"

    data_json = json.loads(out_json.read_text())
    assert "runs" in data_json
    assert "baseline" in data_json
    assert len(data_json["runs"]) == 2
    for entry in data_json["runs"]:
        assert "latent_dim" in entry
        assert "accuracy" in entry
        assert "collapsed" in entry
        assert "student_params" in entry
        assert "compression_ratio" in entry
        assert 0.0 <= entry["accuracy"] <= 1.0

    # Checkpoints were persisted
    assert (ckpt_dir / "student_dim4.pt").exists()
    assert (ckpt_dir / "student_dim8.pt").exists()

    # Baseline has expected fields
    assert data_json["baseline"]["latent_dim"] == 16
    assert 0.0 <= data_json["baseline"]["accuracy"] <= 1.0
