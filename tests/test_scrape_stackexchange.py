"""Tests for scripts/scrape_stackexchange.py.

All tests are offline — HTTP calls are mocked via unittest.mock.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure scripts/ is importable
_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from scrape_stackexchange import (
    DOMAIN_TAGS,
    _best_answer,
    _to_training_example,
    scrape_domain,
    strip_html,
)


# ---------------------------------------------------------------------------
# strip_html
# ---------------------------------------------------------------------------


class TestStripHtml:
    def test_removes_simple_tags(self):
        result = strip_html("<p>Hello <b>world</b></p>")
        assert result == "Hello world"

    def test_preserves_plain_text(self):
        result = strip_html("No tags here")
        assert result == "No tags here"

    def test_converts_inline_code_to_markdown(self):
        result = strip_html("<p>Use <code>HAL_Init()</code> first.</p>")
        assert "`HAL_Init()`" in result
        assert "<code>" not in result

    def test_converts_pre_to_fenced_code_block(self):
        result = strip_html("<pre>int main() { return 0; }</pre>")
        assert "```" in result
        assert "int main()" in result
        assert "<pre>" not in result

    def test_strips_nested_tags(self):
        result = strip_html("<div><p><span>deep text</span></p></div>")
        assert result == "deep text"

    def test_handles_empty_string(self):
        assert strip_html("") == ""

    def test_multiline_pre_block(self):
        html = "<pre>line1\nline2\nline3</pre>"
        result = strip_html(html)
        assert "line1" in result
        assert "line2" in result
        assert "```" in result


# ---------------------------------------------------------------------------
# Tag mapping
# ---------------------------------------------------------------------------


class TestDomainTags:
    def test_all_domains_present(self):
        expected = {
            "kicad-dsl", "spice", "emc", "stm32", "embedded",
            "freecad", "platformio", "power", "dsp", "electronics",
        }
        assert set(DOMAIN_TAGS.keys()) == expected

    def test_freecad_has_no_tags(self):
        # freecad is not on electronics.stackexchange
        assert DOMAIN_TAGS["freecad"] == []

    def test_non_empty_domains_have_tags(self):
        for domain, tags in DOMAIN_TAGS.items():
            if domain != "freecad":
                assert len(tags) > 0, f"Domain '{domain}' should have tags"

    def test_stm32_contains_stm32_tag(self):
        assert "stm32" in DOMAIN_TAGS["stm32"]

    def test_power_contains_mosfet(self):
        assert "mosfet" in DOMAIN_TAGS["power"]


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------


class TestOutputFormat:
    def _make_question(self, **kwargs: object) -> dict:
        return {
            "question_id": 1,
            "title": "How to configure SPI on STM32?",
            "body": "<p>I need to configure <code>SPI1</code> on an STM32F4.</p>",
            "score": 15,
            "tags": ["stm32", "spi"],
            **kwargs,
        }

    def _make_answer(self, **kwargs: object) -> dict:
        return {
            "answer_id": 2,
            "body": "<p>Use <code>HAL_SPI_Init()</code> after calling <code>MX_SPI1_Init()</code>.</p>",
            "score": 10,
            "is_accepted": True,
            **kwargs,
        }

    def test_output_has_required_keys(self):
        ex = _to_training_example(self._make_question(), self._make_answer(), "stm32")
        assert "messages" in ex
        assert "domain" in ex
        assert "source" in ex
        assert "se_score" in ex

    def test_messages_has_user_and_assistant(self):
        ex = _to_training_example(self._make_question(), self._make_answer(), "stm32")
        roles = [m["role"] for m in ex["messages"]]
        assert roles == ["user", "assistant"]

    def test_user_content_contains_title_and_body(self):
        ex = _to_training_example(self._make_question(), self._make_answer(), "stm32")
        user_msg = ex["messages"][0]["content"]
        assert "How to configure SPI on STM32?" in user_msg
        assert "SPI1" in user_msg  # from stripped HTML

    def test_assistant_content_is_stripped_html(self):
        ex = _to_training_example(self._make_question(), self._make_answer(), "stm32")
        assistant_msg = ex["messages"][1]["content"]
        assert "<p>" not in assistant_msg
        assert "HAL_SPI_Init()" in assistant_msg

    def test_domain_field_matches(self):
        ex = _to_training_example(self._make_question(), self._make_answer(), "stm32")
        assert ex["domain"] == "stm32"

    def test_source_field(self):
        ex = _to_training_example(self._make_question(), self._make_answer(), "stm32")
        assert ex["source"] == "electronics.stackexchange.com"

    def test_se_score_matches_question_score(self):
        ex = _to_training_example(self._make_question(score=42), self._make_answer(), "stm32")
        assert ex["se_score"] == 42

    def test_jsonl_serializable(self):
        ex = _to_training_example(self._make_question(), self._make_answer(), "stm32")
        dumped = json.dumps(ex)
        loaded = json.loads(dumped)
        assert loaded["domain"] == "stm32"


# ---------------------------------------------------------------------------
# Best answer selection
# ---------------------------------------------------------------------------


class TestBestAnswer:
    def test_returns_accepted_answer(self):
        answers = [
            {"answer_id": 1, "score": 20, "is_accepted": False},
            {"answer_id": 2, "score": 5, "is_accepted": True},
        ]
        best = _best_answer(answers, min_score=5)
        assert best is not None
        assert best["answer_id"] == 2

    def test_returns_highest_score_when_no_accepted(self):
        answers = [
            {"answer_id": 1, "score": 8, "is_accepted": False},
            {"answer_id": 2, "score": 15, "is_accepted": False},
        ]
        best = _best_answer(answers, min_score=5)
        assert best is not None
        assert best["answer_id"] == 2

    def test_returns_none_when_all_below_min_score(self):
        answers = [
            {"answer_id": 1, "score": 2, "is_accepted": False},
            {"answer_id": 2, "score": 1, "is_accepted": False},
        ]
        best = _best_answer(answers, min_score=5)
        assert best is None

    def test_returns_none_for_empty_list(self):
        assert _best_answer([], min_score=5) is None


# ---------------------------------------------------------------------------
# Dry-run (no HTTP calls)
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_makes_no_http_calls(self, tmp_path: Path):
        """--dry-run must not call httpx.Client.get."""
        with patch("scrape_stackexchange.httpx.Client") as mock_client_cls:
            result = scrape_domain(
                "stm32",
                max_pages=2,
                min_score=5,
                api_key=None,
                delay=0,
                dry_run=True,
            )
        mock_client_cls.assert_not_called()
        assert result == 0

    def test_dry_run_returns_zero_for_domain_without_tags(self):
        result = scrape_domain(
            "freecad",
            max_pages=1,
            min_score=5,
            dry_run=True,
        )
        assert result == 0


# ---------------------------------------------------------------------------
# Resume / skip logic
# ---------------------------------------------------------------------------


class TestResume:
    def test_skips_domain_with_existing_output(self, tmp_path: Path):
        """If output file exists with content, scrape_domain should skip and return count."""
        out_dir = tmp_path / "stm32"
        out_dir.mkdir(parents=True)
        out_file = out_dir / "train.jsonl"
        # Pre-populate with 3 dummy examples
        with out_file.open("w") as f:
            for i in range(3):
                f.write(json.dumps({"messages": [], "domain": "stm32", "source": "test", "se_score": i}) + "\n")

        with (
            patch("scrape_stackexchange.OUTPUT_ROOT", tmp_path),
            patch("scrape_stackexchange.httpx.Client") as mock_client_cls,
        ):
            result = scrape_domain("stm32", max_pages=1, min_score=5, dry_run=False)

        mock_client_cls.assert_not_called()
        assert result == 3


# ---------------------------------------------------------------------------
# Scrape integration (mocked HTTP)
# ---------------------------------------------------------------------------


class TestScrapeWithMockedHttp:
    def _make_api_response(self) -> dict:
        return {
            "has_more": False,
            "items": [
                {
                    "question_id": 100,
                    "title": "STM32 DMA double buffer mode",
                    "body": "<p>How do I set up <code>DMA</code> double buffer on STM32H7?</p>",
                    "score": 12,
                    "tags": ["stm32", "dma"],
                    "answers": [
                        {
                            "answer_id": 200,
                            "body": "<p>Enable <code>DMA_SxCR_DBM</code> bit in the DMA stream control register.</p>",
                            "score": 9,
                            "is_accepted": True,
                        }
                    ],
                }
            ],
        }

    def test_writes_jsonl_output(self, tmp_path: Path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = self._make_api_response()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp

        with (
            patch("scrape_stackexchange.OUTPUT_ROOT", tmp_path),
            patch("scrape_stackexchange.httpx.Client", return_value=mock_client),
            patch("scrape_stackexchange.time.sleep"),
        ):
            count = scrape_domain("stm32", max_pages=1, min_score=5, delay=0)

        assert count == 1
        out_file = tmp_path / "stm32" / "train.jsonl"
        assert out_file.exists()
        with out_file.open() as f:
            lines = f.readlines()
        assert len(lines) == 1
        ex = json.loads(lines[0])
        assert ex["domain"] == "stm32"
        assert ex["se_score"] == 12
        assert "STM32 DMA double buffer mode" in ex["messages"][0]["content"]

    def test_skips_low_score_questions(self, tmp_path: Path):
        api_resp = self._make_api_response()
        api_resp["items"][0]["score"] = 2  # below min_score=5

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = api_resp

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp

        with (
            patch("scrape_stackexchange.OUTPUT_ROOT", tmp_path),
            patch("scrape_stackexchange.httpx.Client", return_value=mock_client),
            patch("scrape_stackexchange.time.sleep"),
        ):
            count = scrape_domain("stm32", max_pages=1, min_score=5, delay=0)

        assert count == 0
