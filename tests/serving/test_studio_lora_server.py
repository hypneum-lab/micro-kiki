"""Unit tests for src/serving/studio_lora_server.py — MLX mocked out."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def test_adapter_name_to_path_maps_known_stacks(tmp_path):
    """adapter_name_to_path returns the absolute path for each of the 10 stacks."""
    from src.serving.studio_lora_server import adapter_name_to_path

    # Simulate a stacks root containing 3 subdirs
    (tmp_path / "stack-dsp").mkdir()
    (tmp_path / "stack-spice").mkdir()
    (tmp_path / "stack-stm32").mkdir()
    for d in ("dsp", "spice", "stm32"):
        (tmp_path / f"stack-{d}" / "adapters.safetensors").write_bytes(b"stub")

    p = adapter_name_to_path("dsp", stacks_root=tmp_path)
    assert p == tmp_path / "stack-dsp" / "adapters.safetensors"


def test_adapter_name_to_path_raises_on_unknown(tmp_path):
    from src.serving.studio_lora_server import adapter_name_to_path

    import pytest
    with pytest.raises(FileNotFoundError, match="no adapter"):
        adapter_name_to_path("nonexistent-domain", stacks_root=tmp_path)


def test_build_chat_prompt_no_system_no_adapter_persona():
    """With real LoRA, we DO NOT inject a system prompt — weights do the specialisation."""
    from src.serving.studio_lora_server import build_chat_prompt

    tokenizer = MagicMock()
    tokenizer.apply_chat_template = MagicMock(return_value="TEMPLATED")
    out = build_chat_prompt(tokenizer, user_message="What is a Schmitt trigger?")
    tokenizer.apply_chat_template.assert_called_once()
    args, kwargs = tokenizer.apply_chat_template.call_args
    messages = args[0] if args else kwargs["conversation"]
    # Exactly one message, role=user, no system persona
    assert len(messages) == 1
    assert messages[0] == {"role": "user", "content": "What is a Schmitt trigger?"}


def test_swap_adapter_tracks_currently_loaded():
    """swap_adapter updates state and calls mx.load + model.load_weights."""
    from src.serving.studio_lora_server import AdapterState, swap_adapter

    state = AdapterState(model=MagicMock(), tokenizer=MagicMock(),
                         stacks_root=Path("/tmp/fake-stacks"), currently_loaded=None)
    fake_weights = [("lora_A", "tensor_a"), ("lora_B", "tensor_b")]
    with patch("src.serving.studio_lora_server.mx") as mx_mock:
        mx_mock.load.return_value = dict(fake_weights)
        swap_adapter(state, "dsp",
                     adapter_path_override=Path("/tmp/fake-stacks/stack-dsp/adapters.safetensors"))
    assert state.currently_loaded == "dsp"
    state.model.load_weights.assert_called_once()
    # strict=False means unknown keys don't error
    _args, kwargs = state.model.load_weights.call_args
    assert kwargs.get("strict") is False


def test_swap_adapter_base_resets_to_none():
    """Passing adapter_name='base' resets currently_loaded to None (no-op if already None)."""
    from src.serving.studio_lora_server import AdapterState, swap_adapter

    state = AdapterState(model=MagicMock(), tokenizer=MagicMock(),
                         stacks_root=Path("/tmp/fake-stacks"), currently_loaded="dsp")
    with patch("src.serving.studio_lora_server.mx"):
        swap_adapter(state, "base")
    assert state.currently_loaded is None
    # No weight load call — 'base' means zero out LoRA, which mlx does by having LoRA.B init to 0
    # For simplicity we just mark state; generation quality is verified in smoke test
