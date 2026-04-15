"""Tests for :class:`src.base.loader.BaseModelLoader`.

The default test set is fully mocked — no real weights are loaded and
no ML dependencies (``torch``, ``transformers``, ``peft``,
``llama_cpp``) need to be installed. A single ``@pytest.mark.integration``
test is provided as a placeholder for an end-to-end run on kxkm-ai;
it is skipped by default.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.base.loader import BaseModelLoader


# ---------------------------------------------------------------------------
# Construction + trivial accessors
# ---------------------------------------------------------------------------


def test_loader_init_stores_path(tmp_path: Path) -> None:
    loader = BaseModelLoader(tmp_path / "qwen")
    assert loader.model_path == tmp_path / "qwen"
    assert loader.model is None
    assert loader.tokenizer is None
    assert loader.backend is None
    assert loader.active_adapter is None
    assert loader.adapters == {}


def test_loader_accepts_string_path() -> None:
    loader = BaseModelLoader("/fake/path/qwen")
    assert loader.model_path == Path("/fake/path/qwen")


# ---------------------------------------------------------------------------
# load_bf16 — verify transformers calls without real weights
# ---------------------------------------------------------------------------


def _install_stub_modules() -> dict[str, ModuleType]:
    """Install stub ``torch`` + ``transformers`` modules in sys.modules.

    Returns the mapping of created modules so the caller can introspect
    the calls made on them. Pre-existing modules are preserved and
    restored via the monkeypatch.setitem in the caller.
    """
    torch_mod = ModuleType("torch")
    torch_mod.bfloat16 = "bfloat16-sentinel"  # type: ignore[attr-defined]

    transformers_mod = ModuleType("transformers")
    transformers_mod.AutoModelForCausalLM = MagicMock()  # type: ignore[attr-defined]
    transformers_mod.AutoTokenizer = MagicMock()  # type: ignore[attr-defined]
    transformers_mod.AutoModelForCausalLM.from_pretrained.return_value = MagicMock(  # type: ignore[attr-defined]
        name="bf16-model"
    )
    transformers_mod.AutoTokenizer.from_pretrained.return_value = MagicMock(  # type: ignore[attr-defined]
        name="tokenizer"
    )

    return {"torch": torch_mod, "transformers": transformers_mod}


def test_load_bf16_invokes_transformers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stubs = _install_stub_modules()
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)

    loader = BaseModelLoader(tmp_path).load_bf16()

    assert loader.backend == "bf16"
    assert loader.model is not None
    assert loader.tokenizer is not None
    stubs["transformers"].AutoModelForCausalLM.from_pretrained.assert_called_once()  # type: ignore[attr-defined]
    kwargs = stubs["transformers"].AutoModelForCausalLM.from_pretrained.call_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["torch_dtype"] == "bfloat16-sentinel"
    assert kwargs["device_map"] == "auto"
    assert kwargs["trust_remote_code"] is True


# ---------------------------------------------------------------------------
# load_q4 — lazy llama_cpp import with a graceful error
# ---------------------------------------------------------------------------


def test_load_q4_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gguf = tmp_path / "qwen.Q4_K_M.gguf"
    gguf.write_bytes(b"")

    fake_llama = MagicMock(name="Llama-instance")
    fake_llama_cls = MagicMock(return_value=fake_llama)
    stub = ModuleType("llama_cpp")
    stub.Llama = fake_llama_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", stub)

    loader = BaseModelLoader(tmp_path).load_q4(gguf)

    assert loader.backend == "q4"
    assert loader.model is fake_llama
    fake_llama_cls.assert_called_once_with(model_path=str(gguf), n_ctx=4096)


def test_load_q4_missing_llama_cpp_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force the import to fail by replacing the module with None.
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    loader = BaseModelLoader(tmp_path)
    with pytest.raises(RuntimeError, match="llama-cpp-python"):
        loader.load_q4(tmp_path / "missing.gguf")


# ---------------------------------------------------------------------------
# enable_lora_switching + register_adapter guardrails
# ---------------------------------------------------------------------------


def test_enable_lora_switching_requires_bf16(tmp_path: Path) -> None:
    loader = BaseModelLoader(tmp_path)
    # No backend loaded → must refuse.
    with pytest.raises(RuntimeError, match="load_bf16"):
        loader.enable_lora_switching()


def test_enable_lora_switching_rejects_q4_backend(tmp_path: Path) -> None:
    loader = BaseModelLoader(tmp_path)
    loader._backend = "q4"  # simulate a Q4-loaded state
    loader._model = MagicMock()
    with pytest.raises(RuntimeError, match="Q4 GGUF"):
        loader.enable_lora_switching()


def test_register_adapter_requires_enable_first(tmp_path: Path) -> None:
    loader = BaseModelLoader(tmp_path)
    with pytest.raises(RuntimeError, match="enable_lora_switching"):
        loader.register_adapter("foo", "/fake/adapter")


def test_register_adapter_after_enable_records_path(tmp_path: Path) -> None:
    loader = BaseModelLoader(tmp_path)
    loader._backend = "bf16"
    loader._model = MagicMock()
    loader.enable_lora_switching()
    loader.register_adapter("stack-01", tmp_path / "adapters" / "chat-fr")
    assert loader.adapters["stack-01"].endswith("chat-fr")


# ---------------------------------------------------------------------------
# with_stack — the core hot-swap acceptance test
# ---------------------------------------------------------------------------


class _FakePeftModel:
    """Minimal stand-in for :class:`peft.PeftModel` used by the tests.

    Tracks the active adapter name and records the full call sequence
    so we can assert swap/restore ordering.
    """

    def __init__(self, base: Any, path: str, adapter_name: str):
        self.base = base
        self.peft_config: dict[str, str] = {adapter_name: path}
        self._active = adapter_name
        self.events: list[tuple[str, str]] = [("from_pretrained", adapter_name)]

    def load_adapter(self, path: str, adapter_name: str) -> None:
        self.peft_config[adapter_name] = path
        self.events.append(("load_adapter", adapter_name))

    def set_adapter(self, name: str) -> None:
        self._active = name
        self.events.append(("set_adapter", name))

    def disable_adapter(self) -> None:
        self._active = None  # type: ignore[assignment]
        self.events.append(("disable_adapter", ""))


@pytest.fixture
def fake_peft(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Install a fake ``peft`` module exposing :class:`_FakePeftModel`."""
    captured: dict[str, Any] = {"instance": None}

    def _from_pretrained(
        base: Any, path: str, adapter_name: str
    ) -> _FakePeftModel:
        inst = _FakePeftModel(base, path, adapter_name)
        captured["instance"] = inst
        return inst

    fake = ModuleType("peft")
    fake.PeftModel = SimpleNamespace(from_pretrained=_from_pretrained)  # type: ignore[attr-defined]
    # isinstance(self._model, PeftModel) check — use _FakePeftModel as
    # the runtime type so the loader's wrap-once logic works.
    fake.PeftModel = _FakePeftModel  # type: ignore[attr-defined]
    fake.PeftModel.from_pretrained = staticmethod(_from_pretrained)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "peft", fake)
    fake._captured = captured  # type: ignore[attr-defined]
    return fake


def test_with_stack_rejects_unknown_adapter(tmp_path: Path) -> None:
    loader = BaseModelLoader(tmp_path)
    loader._backend = "bf16"
    loader._model = MagicMock()
    loader.enable_lora_switching()
    with pytest.raises(KeyError, match="not registered"):
        with loader.with_stack("ghost"):
            pass


def test_load_and_switch(tmp_path: Path, fake_peft: ModuleType) -> None:
    """Acceptance test per story-3 plan: load + hot-swap + restore."""
    loader = BaseModelLoader(tmp_path)
    loader._backend = "bf16"
    loader._model = MagicMock(name="base-model")
    loader.enable_lora_switching()
    loader.register_adapter("stack-a", tmp_path / "a")
    loader.register_adapter("stack-b", tmp_path / "b")

    # Activate "a" first so "b" is a true hot-swap scenario.
    loader._activate("stack-a")
    assert loader.active_adapter == "stack-a"

    with loader.with_stack("stack-b"):
        assert loader.active_adapter == "stack-b"
        # Inside the block: "a" should have been loaded as adapter,
        # "b" loaded as second adapter, and "b" set active.
        peft_inst = loader.model
        assert isinstance(peft_inst, _FakePeftModel)
        assert "stack-a" in peft_inst.peft_config
        assert "stack-b" in peft_inst.peft_config

    # After exit: restored to "a".
    assert loader.active_adapter == "stack-a"
    # Verify the event trail includes both swaps + the restore set.
    events = loader.model.events
    set_events = [e for e in events if e[0] == "set_adapter"]
    assert [e[1] for e in set_events] == ["stack-a", "stack-b", "stack-a"]


def test_with_stack_disables_when_no_prior(
    tmp_path: Path, fake_peft: ModuleType
) -> None:
    loader = BaseModelLoader(tmp_path)
    loader._backend = "bf16"
    loader._model = MagicMock(name="base-model")
    loader.enable_lora_switching()
    loader.register_adapter("stack-a", tmp_path / "a")

    assert loader.active_adapter is None
    with loader.with_stack("stack-a"):
        assert loader.active_adapter == "stack-a"

    assert loader.active_adapter is None
    # disable_adapter should have been called during exit.
    assert any(e[0] == "disable_adapter" for e in loader.model.events)


def test_with_stack_restores_on_exception(
    tmp_path: Path, fake_peft: ModuleType
) -> None:
    loader = BaseModelLoader(tmp_path)
    loader._backend = "bf16"
    loader._model = MagicMock(name="base-model")
    loader.enable_lora_switching()
    loader.register_adapter("stack-a", tmp_path / "a")
    loader.register_adapter("stack-b", tmp_path / "b")
    loader._activate("stack-a")

    with pytest.raises(ValueError, match="boom"):
        with loader.with_stack("stack-b"):
            raise ValueError("boom")

    # Prior adapter restored despite the exception.
    assert loader.active_adapter == "stack-a"


# ---------------------------------------------------------------------------
# Integration placeholder — skipped by default
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_load_and_switch_real_kxkm_ai() -> None:  # pragma: no cover
    """Real load on kxkm-ai. Run manually with ``pytest -m integration``.

    Skipped by default — base weights live on kxkm-ai, not GrosMac.
    """
    pytest.skip(
        "integration test: run on kxkm-ai via "
        "`ssh kxkm-ai 'cd micro-kiki && uv run pytest -m integration'`"
    )
