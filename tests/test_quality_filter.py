"""Tests for scripts/dataset_quality_filter.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import dataset_quality_filter as dqf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_example(response: str, domain: str = "stm32") -> dict:
    return {
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": response},
        ],
        "domain": domain,
        "source": "test",
    }


# ---------------------------------------------------------------------------
# Test: short response scores low
# ---------------------------------------------------------------------------


def test_score_example_low_for_short_response():
    """A response under 50 chars should score 0 for length."""
    ex = _make_example("OK")  # 2 chars
    score = dqf.score_example(ex, "stm32")
    # Only the non-refusal bonus (0.1) is possible when length < 50
    assert score <= 0.15, f"Expected low score for very short response, got {score}"


def test_score_example_zero_length_score_below_50():
    """Exactly tests that length < 50 contributes 0.0 to length_score."""
    short_text = "x" * 49
    ex = _make_example(short_text, "stm32")
    score = dqf.score_example(ex, "stm32")
    # Max possible: 0.0 (length) + 0.0 (no keywords) + 0.0 (no code) + 0.1 (no refusal) = 0.1
    assert score <= 0.15


# ---------------------------------------------------------------------------
# Test: rich response with domain keywords + code blocks scores high
# ---------------------------------------------------------------------------


def test_score_example_high_for_keyword_rich_response():
    """A response with domain keywords and a code block should score well above 0.5."""
    text = (
        "To configure STM32 GPIO with the HAL_ library:\n"
        "```c\n"
        "HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);\n"
        "HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET);\n"
        "// DMA transfer for UART using HAL_UART_Transmit_DMA\n"
        "```\n"
        "This uses the STM32 CubeMX-generated SPI and TIM peripherals."
    )
    ex = _make_example(text, "stm32")
    score = dqf.score_example(ex, "stm32")
    assert score >= 0.5, f"Expected high score for keyword-rich response, got {score}"


def test_score_example_code_block_adds_bonus():
    """Presence of ``` should add 0.2 to score."""
    base_text = "a" * 300  # length in 200-2000 range → length_score = 0.3
    without_code = _make_example(base_text, "embedded")
    with_code = _make_example(base_text + "\n```c\ncode\n```", "embedded")

    score_no_code = dqf.score_example(without_code, "embedded")
    score_with_code = dqf.score_example(with_code, "embedded")
    assert score_with_code > score_no_code


def test_score_example_refusal_reduces_score():
    """'I cannot' should lose the 0.1 non-refusal bonus."""
    good_text = "Here is how DMA works in FreeRTOS firmware with interrupts: " + "x" * 200
    bad_text = "I cannot provide that. As an AI, I don't assist with this."

    score_good = dqf.score_example(_make_example(good_text, "embedded"), "embedded")
    score_bad = dqf.score_example(_make_example(bad_text, "embedded"), "embedded")
    assert score_good > score_bad


# ---------------------------------------------------------------------------
# Test: filter_dataset removes duplicates and low-score examples
# ---------------------------------------------------------------------------


def test_filter_dataset_removes_duplicates():
    """Identical responses must appear only once in output."""
    text = "HAL_GPIO_Init with STM32 GPIO DMA SPI TIM CubeMX UART " + "x" * 200
    ex1 = _make_example(text, "stm32")
    ex2 = _make_example(text, "stm32")  # exact duplicate

    result = dqf.filter_dataset([ex1, ex2], "stm32", threshold=0.0)
    assert len(result) == 1, "Duplicate example should be removed"


def test_filter_dataset_removes_low_score_examples():
    """Examples below threshold should be filtered out."""
    short_ex = _make_example("Too short", "stm32")  # will score very low
    score = dqf.score_example(short_ex, "stm32")

    # Use a threshold slightly above the score to ensure it's filtered
    threshold = score + 0.01
    result = dqf.filter_dataset([short_ex], "stm32", threshold=threshold)
    assert len(result) == 0, "Low-score example should be filtered"


def test_filter_dataset_preserves_good_examples():
    """Good examples above threshold should be retained."""
    text = (
        "HAL_GPIO_Init configures STM32 GPIO. Use DMA for UART with CubeMX.\n"
        "```c\nHAL_UART_Transmit_DMA(&huart1, buf, len);\n```\n"
        + "Detailed firmware peripheral configuration with SPI and TIM. " * 5
    )
    good_ex = _make_example(text, "stm32")
    score = dqf.score_example(good_ex, "stm32")

    result = dqf.filter_dataset([good_ex], "stm32", threshold=0.3)
    assert len(result) == 1, f"Good example (score={score}) should be kept at threshold=0.3"


def test_filter_dataset_mixed_keeps_only_good(tmp_path):
    """Mixed batch: only good examples survive the threshold."""
    good_text = (
        "KiCad schematic with symbol, footprint, pad and fp_ module.\n"
        "```\n(kicad_pcb (module C_0402 ...))\n```\n"
        + "Detailed explanation of KiCad schematic design. " * 8
    )
    bad_text = "idk"

    good_ex = _make_example(good_text, "kicad-dsl")
    bad_ex = _make_example(bad_text, "kicad-dsl")

    result = dqf.filter_dataset([good_ex, bad_ex], "kicad-dsl", threshold=0.3)
    assert len(result) == 1
    assert dqf._get_response_text(result[0]) == good_text
