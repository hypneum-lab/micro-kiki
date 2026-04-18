"""Smoke test for the training CLI — runs on synthetic data, 2 epochs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_train_script_runs_on_tiny_corpus(tmp_path: Path) -> None:
    # Create a tiny 2-domain corpus
    data = tmp_path / "data" / "final"
    for dom in ["dsp", "electronics"]:
        (data / dom).mkdir(parents=True)
        lines = [
            json.dumps({"messages": [{"role": "user", "content": f"{dom} question {i}"}, {"role": "assistant", "content": "a"}]})
            for i in range(8)
        ]
        (data / dom / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ckpt = tmp_path / "student.pt"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_text_jepa.py",
            "--data-dir", str(data),
            "--domains", "dsp,electronics",
            "--max-per-domain", "8",
            "--epochs", "2",
            "--batch-size", "2",
            "--output", str(ckpt),
            "--backbone", "random",
            "--seq-len", "8",
            "--input-dim", "32",
            "--latent-dim", "8",
            "--hidden-dim", "16",
        ],
        capture_output=True,
        text=True,
        cwd="/Users/electron/Documents/Projets/micro-kiki-poc-textjepa",
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert ckpt.exists()
