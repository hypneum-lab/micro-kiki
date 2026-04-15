"""Tests for :mod:`src.distill.dedup`."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.distill.dedup import (
    DEFAULT_NUM_PERM,
    DedupConfig,
    LSHIndex,
    dedup_directory,
    dedup_domains,
    jaccard_estimate,
    minhash,
    shingles,
)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def test_shingles_basic() -> None:
    sh = shingles("hello world", k=3)
    assert "hel" in sh
    assert "ell" in sh
    assert "llo" in sh
    assert " wo" in sh  # canonicalized to single spaces


def test_shingles_short_text_falls_back() -> None:
    # Shorter than k → single-shingle fallback.
    assert shingles("hi", k=5) == {"hi"}
    assert shingles("", k=5) == set()


def test_minhash_is_deterministic() -> None:
    sh = shingles("the quick brown fox jumps over the lazy dog")
    sig_a = minhash(sh)
    sig_b = minhash(sh)
    assert sig_a == sig_b
    assert len(sig_a) == DEFAULT_NUM_PERM


def test_jaccard_identical_is_one() -> None:
    sh = shingles("some training prompt about llms")
    sig = minhash(sh)
    assert jaccard_estimate(sig, sig) == 1.0


def test_jaccard_disjoint_is_low() -> None:
    sh1 = shingles("completely unrelated topic about embedded firmware")
    sh2 = shingles("recette de cuisine pour un gâteau au chocolat")
    sig1 = minhash(sh1)
    sig2 = minhash(sh2)
    assert jaccard_estimate(sig1, sig2) < 0.2


def test_jaccard_near_duplicate_is_high() -> None:
    a = "Explain the differential attention mechanism in two paragraphs."
    b = "Explain the differential attention mechanism in 2 paragraphs."
    sig_a = minhash(shingles(a))
    sig_b = minhash(shingles(b))
    assert jaccard_estimate(sig_a, sig_b) >= 0.6


# ---------------------------------------------------------------------------
# LSH index
# ---------------------------------------------------------------------------


def test_lsh_collides_near_duplicates() -> None:
    idx = LSHIndex(bands=32, rows=4)
    sig_a = minhash(shingles("write a function that sums two integers"))
    sig_b = minhash(shingles("write a function that sums two integer"))
    sig_c = minhash(shingles("la philosophie de schopenhauer en 5 points"))
    idx.insert("a", sig_a)
    idx.insert("b", sig_b)
    idx.insert("c", sig_c)
    pairs = idx.candidate_pairs()
    assert ("a", "b") in pairs
    assert ("a", "c") not in pairs
    assert ("b", "c") not in pairs


def test_lsh_validates_signature_length() -> None:
    idx = LSHIndex(bands=32, rows=4)
    with pytest.raises(ValueError):
        idx.insert("x", [0] * 7)


def test_dedup_config_validates_band_rows() -> None:
    with pytest.raises(ValueError):
        DedupConfig(num_perm=128, bands=10, rows=5)


# ---------------------------------------------------------------------------
# Synthetic 3-domain overlap
# ---------------------------------------------------------------------------


def _row(prompt: str, domain: str) -> dict:
    return {"prompt": prompt, "completion": "...", "domain": domain}


def test_dedup_produces_disjoint_partition() -> None:
    """Synthetic set: 3 domains, 30% overlap → output is a disjoint partition."""

    # Each domain gets 10 rows. Rows 0-2 of each domain are NEAR-DUPLICATES
    # of rows 0-2 of the other domains (30% overlap). Rows 3-9 are unique
    # per domain.
    overlap_prompts = [
        "Explain the differential attention mechanism clearly",
        "Write a Python function that reverses a linked list",
        "Décris en français le cycle de l'eau en 3 phrases",
    ]

    # Per-domain distinct filler prompts; each list has 7 items that share
    # no obvious phrase with the other domains' fillers.
    fillers = {
        "code": [
            "write quicksort in rust with property tests",
            "implement a ring buffer in c without memcpy",
            "refactor this python dataclass to pydantic",
            "bitwise trick to count set bits efficiently",
            "explain the borrow checker to a beginner",
            "design a lockfree MPMC queue interface",
            "document a restful pagination cursor scheme",
        ],
        "chat-fr": [
            "raconte une histoire drôle de chaton breton",
            "quels vins servir avec une raclette savoyarde",
            "poeme sur la lune et les étoiles filantes",
            "dis bonjour à ma grand mère en normand",
            "conseil pour réussir sa tarte tatin maison",
            "quelles randonnées en corse en octobre",
            "idées cadeau artisanal pour un mariage",
        ],
        "reasoning": [
            "prove sqrt of 2 is irrational step by step",
            "resolve the monty hall paradox formally",
            "bayesian update after two positive medical tests",
            "derive the pigeonhole principle from scratch",
            "explain goedels incompleteness in plain words",
            "construct a counterexample to naive induction",
            "estimate the ramanujan taxi number via sieve",
        ],
    }

    def per_domain(name: str, twist: str) -> list[dict]:
        rows: list[dict] = []
        for p in overlap_prompts:
            # Near-duplicate: tiny formatting change per domain.
            rows.append(_row(p + twist, name))
        for p in fillers[name]:
            rows.append(_row(p, name))
        return rows

    rows_by_domain = {
        "code": per_domain("code", "."),
        "chat-fr": per_domain("chat-fr", "!"),
        "reasoning": per_domain("reasoning", ""),
    }

    partitioned, report = dedup_domains(
        rows_by_domain, DedupConfig(similarity_threshold=0.6)
    )

    # Every prompt appears in exactly one domain in the output.
    seen: dict[str, str] = {}
    for domain, rows in partitioned.items():
        for row in rows:
            # Canonicalize on the base overlap_prompt (strip any twist char).
            base = row["prompt"].rstrip(".! ")
            if base not in seen:
                seen[base] = domain
            else:
                assert seen[base] == domain, (
                    f"prompt {base!r} appears in {seen[base]} AND {domain}"
                )

    # Exactly 6 rows dropped (2 per overlap_prompt × 3 prompts).
    assert report["dropped"] == 6
    assert report["cross_groups"] == 3

    # Each domain loses ≤ 3 rows.
    for domain in ("code", "chat-fr", "reasoning"):
        assert report["domains"][domain]["input"] == 10
        assert report["domains"][domain]["output"] >= 7


def test_dedup_is_deterministic_on_ties() -> None:
    """Same input → identical partition regardless of call order."""
    rows_by_domain = {
        "alpha": [_row("identical prompt here", "alpha")],
        "beta": [_row("identical prompt here", "beta")],
    }
    a, _ = dedup_domains(rows_by_domain, DedupConfig(similarity_threshold=0.5))
    b, _ = dedup_domains(rows_by_domain, DedupConfig(similarity_threshold=0.5))
    assert a == b
    # The tie is broken lexicographically → "alpha" wins.
    assert len(a["alpha"]) == 1
    assert a["beta"] == []


def test_dedup_preserves_non_duplicates() -> None:
    rows_by_domain = {
        "a": [_row("alpha unique 1", "a"), _row("alpha unique 2", "a")],
        "b": [_row("beta totally different", "b")],
    }
    partitioned, report = dedup_domains(rows_by_domain)
    assert report["dropped"] == 0
    assert len(partitioned["a"]) == 2
    assert len(partitioned["b"]) == 1


# ---------------------------------------------------------------------------
# Directory driver + CLI
# ---------------------------------------------------------------------------


def test_dedup_directory_writes_report_and_outputs(tmp_path: Path) -> None:
    inp = tmp_path / "raw"
    out = tmp_path / "dedup"
    inp.mkdir()

    (inp / "code.jsonl").write_text(
        json.dumps(_row("shared prompt xyz", "code"), ensure_ascii=False)
        + "\n"
        + json.dumps(
            _row("quicksort implementation in rust", "code"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (inp / "chat.jsonl").write_text(
        json.dumps(_row("shared prompt xyz", "chat"), ensure_ascii=False)
        + "\n"
        + json.dumps(
            _row("poeme sur la lune et les etoiles", "chat"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = dedup_directory(inp, out, DedupConfig(similarity_threshold=0.5))

    assert (out / "code.jsonl").exists()
    assert (out / "chat.jsonl").exists()
    assert (out / "_dedup_report.json").exists()
    assert report["dropped"] == 1
    # Report file is valid JSON and round-trips.
    assert json.loads((out / "_dedup_report.json").read_text()) == report


def test_cli_runs(tmp_path: Path) -> None:
    inp = tmp_path / "raw"
    out = tmp_path / "dedup"
    inp.mkdir()
    (inp / "x.jsonl").write_text(
        json.dumps(_row("only one row", "x")) + "\n", encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.distill.dedup",
            "--input",
            str(inp),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "x.jsonl").exists()
