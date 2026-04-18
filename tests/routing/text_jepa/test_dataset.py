"""Tests for the 10-domain corpus loader."""
from __future__ import annotations

import json
from pathlib import Path


def test_load_domain_corpus_reads_jsonl(tmp_path: Path) -> None:
    from src.routing.text_jepa.dataset import load_domain_corpus

    d = tmp_path / "final"
    (d / "dsp").mkdir(parents=True)
    (d / "electronics").mkdir(parents=True)

    (d / "dsp" / "train.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "What is FFT?"}, {"role": "assistant", "content": "x"}]}) + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "IIR filter"}, {"role": "assistant", "content": "x"}]}) + "\n",
        encoding="utf-8",
    )
    (d / "electronics" / "train.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "Ohm law"}, {"role": "assistant", "content": "x"}]}) + "\n",
        encoding="utf-8",
    )

    samples = load_domain_corpus(d, domains=["dsp", "electronics"], max_per_domain=10)
    assert len(samples) == 3
    texts = {s.text for s in samples}
    assert "What is FFT?" in texts
    assert "Ohm law" in texts
    labels = {s.domain for s in samples}
    assert labels == {"dsp", "electronics"}


def test_load_domain_corpus_respects_max_per_domain(tmp_path: Path) -> None:
    from src.routing.text_jepa.dataset import load_domain_corpus

    d = tmp_path / "final" / "dsp"
    d.mkdir(parents=True)
    lines = [
        json.dumps({"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": "a"}]})
        for i in range(20)
    ]
    (d / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    samples = load_domain_corpus(d.parent, domains=["dsp"], max_per_domain=5)
    assert len(samples) == 5


def test_load_domain_corpus_skips_missing_dir(tmp_path: Path) -> None:
    from src.routing.text_jepa.dataset import load_domain_corpus

    (tmp_path / "final").mkdir()
    samples = load_domain_corpus(tmp_path / "final", domains=["missing"], max_per_domain=10)
    assert samples == []
