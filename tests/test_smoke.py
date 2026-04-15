"""One-line smoke tests for each src/distill module.

These tests exist to catch import regressions, signature drift, and
basic wiring errors fast — before the module-level unit tests have a
chance to run. They rely only on the shared fixtures in
``tests/conftest.py`` and never hit the network or disk beyond
``tmp_path``.

``test_loader_smoke`` is skipped until story 3 lands (loader module
doesn't exist yet); when it does, flip the skip mark off.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from src.distill.dedup import DedupConfig, dedup_directory
from src.distill.generator import GeneratorConfig, generate_examples
from src.distill.teacher_client import (
    GenerateParams,
    TeacherCache,
    TeacherClient,
    cache_key,
)
from tests.conftest import FakeTeacher


# ---------------------------------------------------------------------------
# Package import smoke
# ---------------------------------------------------------------------------


def test_src_package_imports() -> None:
    importlib.import_module("src")
    importlib.import_module("src.distill")


# ---------------------------------------------------------------------------
# Generator smoke — exercises the generator with the FakeTeacher fixture.
# ---------------------------------------------------------------------------


def test_generator_smoke(
    mock_teacher: FakeTeacher,
    sample_prompts: list[str],
    tmp_path: Path,
) -> None:
    out = tmp_path / "generated.jsonl"
    stats = generate_examples(
        sample_prompts[:3],
        mock_teacher,
        out,
        GeneratorConfig(n_per_prompt=1, domain="smoke"),
    )
    assert stats["generated"] == 3
    assert stats["failed"] == 0
    assert out.exists()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Dedup smoke — builds a tiny 2-domain duplicate set and verifies the
# public entry point returns a non-empty result.
# ---------------------------------------------------------------------------


def test_dedup_smoke(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    long_shared = (
        "Quelle est la difference fondamentale entre un moteur electrique "
        "asynchrone et un moteur synchrone, en deux paragraphes."
    )
    (input_dir / "a.jsonl").write_text(
        '{"prompt": "p-a-1 unique contenu alpha beta gamma", '
        '"completion": "c1", "domain": "a", "hash": "h1"}\n'
        '{"prompt": "' + long_shared + '", '
        '"completion": "s", "domain": "a", "hash": "h2"}\n',
        encoding="utf-8",
    )
    (input_dir / "b.jsonl").write_text(
        '{"prompt": "p-b-2 different zeta eta theta iota", '
        '"completion": "c2", "domain": "b", "hash": "h3"}\n'
        '{"prompt": "' + long_shared + '", '
        '"completion": "s", "domain": "b", "hash": "h4"}\n',
        encoding="utf-8",
    )
    out_dir = tmp_path / "dedup"
    report = dedup_directory(input_dir, out_dir, DedupConfig())
    assert out_dir.exists()
    assert (out_dir / "_dedup_report.json").exists()
    # The shared prompt should trigger exactly one cross-domain group.
    assert report["cross_groups"] >= 1
    assert report["dropped"] >= 1


# ---------------------------------------------------------------------------
# Teacher smoke — construct the real TeacherClient with a mocked
# transport, confirm the async + sync surfaces both work, and verify the
# cache hit path.
# ---------------------------------------------------------------------------


def test_teacher_smoke(tmp_path: Path) -> None:
    import httpx

    calls = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200,
            json={
                "id": "x",
                "model": "mistral-large-opus",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "salut"},
                    }
                ],
                "usage": {},
            },
        )

    transport = httpx.MockTransport(handler)
    cache = TeacherCache(path=tmp_path / "cache.sqlite")
    client = TeacherClient(
        endpoints={"mistral-large-opus": "http://teacher.local"},
        cache=cache,
        http_client=httpx.AsyncClient(transport=transport, timeout=2.0),
    )
    try:
        first = client.generate_sync(
            "hi", "mistral-large-opus", params=GenerateParams(max_tokens=8)
        )
        second = client.generate_sync(
            "hi", "mistral-large-opus", params=GenerateParams(max_tokens=8)
        )
    finally:
        import asyncio

        asyncio.run(client.aclose())

    assert first == second == "salut"
    assert calls["n"] == 1
    # Deterministic cache key is stable across construction.
    assert cache_key("hi", "mistral-large-opus", {"max_tokens": 8}) == cache_key(
        "hi", "mistral-large-opus", {"max_tokens": 8}
    )


# ---------------------------------------------------------------------------
# Fixtures sanity
# ---------------------------------------------------------------------------


def test_sample_prompts_fixture(sample_prompts: list[str]) -> None:
    assert 5 <= len(sample_prompts) <= 32
    assert all(isinstance(p, str) and p.strip() for p in sample_prompts)


def test_tmp_model_dir_fixture(tmp_model_dir: Path) -> None:
    assert (tmp_model_dir / "bf16" / "config.json").is_file()


def test_mock_teacher_fixture(mock_teacher: FakeTeacher) -> None:
    out = mock_teacher.complete("x")
    assert out.startswith("mock::")
    assert mock_teacher.call_log[0][0] == "x"


# ---------------------------------------------------------------------------
# Loader smoke — deferred until story 3 ships the loader module.
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="loader pending step 3 (src.base.loader not yet written)")
def test_loader_smoke(tmp_model_dir: Path) -> None:  # pragma: no cover
    from src.base.loader import BaseModelLoader  # type: ignore[import-not-found]

    loader = BaseModelLoader(tmp_model_dir)
    assert loader is not None
