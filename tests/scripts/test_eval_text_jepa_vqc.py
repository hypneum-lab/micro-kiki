"""Smoke test the eval script on a tiny synthetic corpus + untrained student."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pennylane = pytest.importorskip("pennylane")
torch = pytest.importorskip("torch")


def test_eval_script_produces_json(tmp_path: Path) -> None:
    from src.routing.text_jepa.encoder import StudentEncoder

    # Tiny corpus (use only 2 domains — faster and matches the eval script's n_classes)
    data = tmp_path / "data" / "final"
    for dom in ["dsp", "electronics"]:
        (data / dom).mkdir(parents=True)
        lines = []
        for i in range(20):
            text = f"{dom} unique question number {i}"
            lines.append(json.dumps({"messages": [{"role": "user", "content": text}, {"role": "assistant", "content": "a"}]}))
        (data / dom / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Fake trained student checkpoint
    student = StudentEncoder(input_dim=16, hidden_dim=16, output_dim=8)
    ckpt = tmp_path / "student.pt"
    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "predictor_state_dict": {},
            "config": {
                "input_dim": 16,
                "latent_dim": 8,
                "hidden_dim": 16,
                "seq_len": 4,
                "backbone": "random",
            },
        },
        ckpt,
    )

    out_json = tmp_path / "results.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_text_jepa_vqc.py",
            "--data-dir", str(data),
            "--domains", "dsp,electronics",
            "--max-per-domain", "20",
            "--epochs", "2",
            "--checkpoint", str(ckpt),
            "--output", str(out_json),
            "--backbone", "random",
            "--seq-len", "4",
            "--input-dim", "16",
        ],
        capture_output=True,
        text=True,
        cwd="/Users/electron/Documents/Projets/micro-kiki-poc-textjepa",
    )
    assert result.returncode == 0, f"stderr={result.stderr}\nstdout={result.stdout}"
    data_json = json.loads(out_json.read_text())
    assert "baseline" in data_json
    assert "text_jepa" in data_json
    assert "accuracy" in data_json["baseline"]
    assert "accuracy" in data_json["text_jepa"]
    assert 0.0 <= data_json["baseline"]["accuracy"] <= 1.0
    assert 0.0 <= data_json["text_jepa"]["accuracy"] <= 1.0
