"""Light tests for experiments/demo_35b_gb300 (no GPU, no MLX)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "demo_35b_gb300"))

import demo_rules as D  # noqa: E402

PROMPTS = ROOT / "docs" / "prereg" / "demo-35b-lora-gb300" / "prompts.json"


def test_frozen_prompts_load_and_match_domains():
    by = D.load_prompts(str(PROMPTS))
    assert list(by) == D.DOMAINS
    assert all(len(v) == 3 for v in by.values())


def test_calibrate_k():
    assert D.calibrate_k(4.0) == 4
    assert D.calibrate_k(20.0) == 1
    with pytest.raises(ValueError):
        D.calibrate_k(0.0)


def test_check_adapter_exact_missing_extra_shape_dtype():
    want = {"a.lora_a": ((2048, 16), "float32"), "a.lora_b": ((16, 512), "float32")}
    assert D.check_adapter(dict(want), want) == []
    got = {"a.lora_a": ((2048, 8), "float32"), "x.lora_b": ((16, 512), "bfloat16")}
    probs = D.check_adapter(got, want)
    assert "missing a.lora_b" in probs and "extra x.lora_b" in probs
    assert any(p.startswith("shape a.lora_a") for p in probs)


def test_jpt_and_zero_tokens():
    rest = {"gpu_j": 200.0, "dur_s": 20.0}
    assert D.jpt({"gpu_j": 400.0, "dur_s": 20.0, "generated_tokens": 100}, rest) == pytest.approx(2.0)
    assert D.jpt({"gpu_j": 400.0, "dur_s": 20.0, "generated_tokens": 0}, rest) is None


def _run(d, arm, j, n=100, dur=20.0, valid=True):
    return {"domain": d, "arm": arm, "valid": valid, "j_per_token": j, "gpu_j": 400.0,
            "generated_tokens": n, "dur_s": dur}


def test_summarize_medians_ratio_and_exclusions():
    runs = [_run("python", "base", x) for x in (1.0, 2.0, 3.0)]
    runs += [_run("python", "adapter", x) for x in (2.0, 4.0, -1.0)]
    runs += [_run("python", "adapter", 9.0, valid=False), _run("cpp", "base", 1.0, dur=10.0)]
    s = D.summarize(runs)
    c = s["cells"]
    assert c["python|base"]["j_per_token_median"] == 2.0
    assert c["python|adapter"]["excluded_nonpositive"] == 1
    assert c["python|adapter"]["j_per_token_median"] == 3.0
    assert s["ratio_of_medians_adapter_over_base"]["python"] == pytest.approx(1.5)
    assert c["cpp|base"]["short_runs"] == 1
    assert s["status"] == "demo, not a scientific claim"
