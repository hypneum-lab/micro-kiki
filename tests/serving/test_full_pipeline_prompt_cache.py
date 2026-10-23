"""Prompt-cache reuse in ``_MLXRuntimeAdapter.generate``.

mlx / mlx_lm are stubbed: a fake cache records the token ids that went
through the model, so the tests can assert that each token is
prefilled exactly once and that ``mlx_generate`` only receives the
tokens not already in the cache.
"""
from __future__ import annotations

import sys
import types

import pytest


class _FakeCache:
    def __init__(self) -> None:
        self.tokens: list[int] = []

    @property
    def state(self) -> list[int]:
        return self.tokens


class _FakeArray:
    def __init__(self, toks: list[int]) -> None:
        self.toks = list(toks)

    def __getitem__(self, _idx: object) -> list[int]:
        return self.toks


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def __call__(self, toks: list[int], cache: list[_FakeCache]) -> None:
        self.calls.append(list(toks))
        for c in cache:
            c.tokens.extend(toks)


class _FakeTokenizer:
    bos_token = None

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(ch) for ch in text]


@pytest.fixture()
def adapter(monkeypatch: pytest.MonkeyPatch):
    from src.serving import full_pipeline_server as fps

    gen_calls: list[dict] = []

    def fake_generate(model, tokenizer, *, prompt, prompt_cache, **kw):
        # Record what the cache held when generation started, then
        # mutate it like mlx_lm does (prompt + generated tokens).
        gen_calls.append(
            {
                "prompt": list(prompt),
                "cached": list(prompt_cache[0].tokens),
                "kw": kw,
            }
        )
        for c in prompt_cache:
            c.tokens.extend(list(prompt) + [0, 0, 0])
        return "out"

    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.generate = fake_generate
    models = types.ModuleType("mlx_lm.models")
    cache_mod = types.ModuleType("mlx_lm.models.cache")
    cache_mod.make_prompt_cache = lambda model: [_FakeCache(), _FakeCache()]
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda temp: None
    gen_mod = types.ModuleType("mlx_lm.generate")
    quantize_calls: list[dict] = []
    gen_mod.maybe_quantize_kv_cache = lambda cache, **kw: quantize_calls.append(kw)
    mx = types.ModuleType("mlx.core")
    mx.array = _FakeArray
    mx.eval = lambda *_a: None
    mx.clear_cache = lambda: None
    mlx = types.ModuleType("mlx")
    mlx.core = mx
    for name, mod in {
        "mlx_lm": mlx_lm,
        "mlx_lm.models": models,
        "mlx_lm.models.cache": cache_mod,
        "mlx_lm.sample_utils": sample_utils,
        "mlx_lm.generate": gen_mod,
        "mlx": mlx,
        "mlx.core": mx,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    a = object.__new__(fps._MLXRuntimeAdapter)
    a._model = _FakeModel()
    a._tokenizer = _FakeTokenizer()
    a._cached_prompt_tokens = None
    a._prompt_cache = None
    a.gen_calls = gen_calls
    a.quantize_calls = quantize_calls
    return a


def _ids(text: str) -> list[int]:
    return [ord(ch) for ch in text]


def test_cold_request_prefills_all_but_last_token(adapter) -> None:
    assert adapter.generate("abcd") == "out"
    call = adapter.gen_calls[0]
    assert call["cached"] == _ids("abc")
    assert call["prompt"] == _ids("d")
    assert adapter._cached_prompt_tokens == _ids("abc")


def test_identical_prompt_skips_prefill(adapter) -> None:
    adapter.generate("abcd")
    model_calls = len(adapter._model.calls)
    adapter.generate("abcd")
    assert len(adapter._model.calls) == model_calls  # no new prefill
    second = adapter.gen_calls[1]
    # Generated tokens of the first request must not leak into the
    # cache handed to the second one.
    assert second["cached"] == _ids("abc")
    assert second["prompt"] == _ids("d")


def test_extended_prompt_prefills_only_suffix(adapter) -> None:
    adapter.generate("abcd")
    adapter._model.calls.clear()
    adapter.generate("abcdefg")
    assert adapter._model.calls == [_ids("def")]
    call = adapter.gen_calls[1]
    assert call["cached"] == _ids("abcdef")
    assert call["prompt"] == _ids("g")


def test_diverging_prompt_rebuilds_cache(adapter) -> None:
    adapter.generate("abcd")
    adapter._model.calls.clear()
    adapter.generate("xbcd")
    assert adapter._model.calls == [_ids("xbc")]
    assert adapter.gen_calls[1]["cached"] == _ids("xbc")


def test_prefill_is_chunked_and_quantized(adapter, monkeypatch) -> None:
    from src.serving import full_pipeline_server as fps

    monkeypatch.setattr(fps, "_PREFILL_STEP_SIZE", 2)
    adapter.generate("abcdef")
    assert adapter._model.calls == [_ids("ab"), _ids("cd"), _ids("e")]
    assert len(adapter.quantize_calls) == 3
    assert adapter.quantize_calls[0] == {
        "quantized_kv_start": 0,
        "kv_group_size": fps._KV_GROUP_SIZE,
        "kv_bits": fps._KV_BITS,
    }
    kw = adapter.gen_calls[0]["kw"]
    assert kw["prefill_step_size"] == 2
    assert kw["kv_bits"] == fps._KV_BITS
