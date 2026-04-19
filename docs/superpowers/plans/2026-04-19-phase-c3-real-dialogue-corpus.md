# Phase C3 — Real Dialogue Corpus Curation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the synthetic `data/final/` corpus with a curated real-dialogue corpus covering the same 10 domains, validated by emergent clustering vs the existing taxonomy, so Paper A's results generalise beyond the synthetic benchmark.

**Architecture:** Three-stage pipeline: (1) INGEST — pull conversational logs from Kiki/Yi sources, apply regex sanitization for secrets, NER anonymisation for PII; (2) VALIDATE — embed with MiniLM, cluster with HDBSCAN in latent space, compare clusters to existing 10 domains via a cost-matrix / Hungarian assignment; (3) AUGMENT — for domains under-represented after clustering, generate synthetic completions with a teacher LLM using real fragments as seeds. Final artifact is `data/corpus-real/<domain>/train.jsonl` — drop-in replacement for `data/final/`.

**Tech Stack:** Python 3.14, spaCy (NER), `hdbscan` package (density clustering), sentence-transformers (already used), Qwen3-Coder-480B on Studio (teacher for augmentation), existing `data/final/` as baseline to compare against.

**Decisions baked in** (from Phase C brainstorm 2026-04-19):
- **Source**: Kiki/Yi logs (existing internal) + synthetic augmentation for gaps — NO new data collection
- **Labeling**: clustering emergent + sanity-check vs 10 domains (NOT hand-relabelling)
- **Privacy**: regex sweep (emails, IPs, API keys, tokens) + NER anonymisation for names + pre-publication audit

---

## File Structure

**Files to create:**
- `src/data/sanitization.py` — regex + NER sweep
- `src/data/corpus_validator.py` — embed + cluster + match to existing taxonomy
- `src/data/augmenter.py` — teacher-based synthetic completion for under-represented domains
- `tests/data/test_sanitization.py` — unit tests for regex + NER
- `tests/data/test_corpus_validator.py` — unit tests for cluster matching
- `scripts/build_corpus_real.py` — end-to-end pipeline CLI
- `data/corpus-real/<domain>/train.jsonl` — output, 10 files
- `results/c3-corpus-stats.json` — cluster stats, augmentation stats, sanity reports
- `docs/paper-a/c3-corpus-validation.md` — paper narrative + cluster diagram

**Files to modify:**
- `.gitignore` — ensure `data/corpus-real/*/raw/` is gitignored (raw PII-bearing inputs never committed)
- `docs/superpowers/plans/2026-04-19-phase-c-roadmap.md` (status update at end)

---

### Task 1: Inventory available log sources

**Files:**
- Create: `docs/c3-source-inventory.md` (discovery doc, committed)

- [ ] **Step 1: List existing internal conversational sources**

Run:
```bash
find ~/Documents/Lelectron_rare ~/Documents/Projets -name '*.jsonl' -path '*kiki*' -o -name '*.jsonl' -path '*yi*' 2>/dev/null | head -30
find ~/Documents/Projets/Factory\ 4\ Life -name '*chat*' -o -name '*conversation*' 2>/dev/null | head -20
```

- [ ] **Step 2: Count lines per candidate source**

For each `.jsonl` found, run `wc -l <path>` and record.

- [ ] **Step 3: Write inventory doc**

Write `docs/c3-source-inventory.md` with a table:

```markdown
# C3 — Dialogue source inventory (2026-04-19)

| Source path | Est. lines | Format | Est. on-topic fraction | Has PII? | Keep? |
|---|---|---|---|---|---|
| ~/Documents/...path1.jsonl | N | {role, content} | 80% technical | yes (names, emails) | yes |
| ~/Documents/...path2.jsonl | N | custom | 20% generic chat | some (IPs) | yes |

## Targets per domain

| Domain | Current data/final lines | Target real lines | Gap | Augmentation strategy |
|---|---|---|---|---|
| dsp | 2821 | 500 | 0 | none needed if source has >500 |
| electronics | N | 500 | TBD | ... |
| ...

## Sources NOT used
- ~/... (reason: contains customer financial data, cannot be sanitised per GDPR)
- ...
```

Fill in actual counts. If a source turns out to be unusable (e.g., all consent-locked), document and skip.

- [ ] **Step 4: Commit inventory**

```bash
git add docs/c3-source-inventory.md
git commit -m "docs(c3): dialogue source inventory"
```

**Acceptance for this task:** If total available usable lines across all sources is < 2000 for our 10 domains combined, STOP and report BLOCKED — C3 cannot proceed without real data. User may decide to collect new data (outside this plan) or de-scope C3.

---

### Task 2: Write failing tests for `sanitization.py`

**Files:**
- Create: `tests/data/__init__.py` (empty)
- Create: `tests/data/test_sanitization.py`

- [ ] **Step 1: Create the test directory**

```bash
mkdir -p tests/data && touch tests/data/__init__.py
```

- [ ] **Step 2: Create the test file** with EXACTLY this content:

```python
"""Tests for src/data/sanitization.py — regex + NER-based PII removal."""
from __future__ import annotations

import pytest


def test_redact_emails():
    from src.data.sanitization import redact_secrets_regex

    text = "Contact me at john.doe@example.com or jane+tag@sub.example.org for details."
    out = redact_secrets_regex(text)
    assert "@example.com" not in out
    assert "@sub.example.org" not in out
    assert "[REDACTED_EMAIL]" in out


def test_redact_ipv4():
    from src.data.sanitization import redact_secrets_regex

    text = "The server at 192.168.1.100 and 10.0.0.5 are down."
    out = redact_secrets_regex(text)
    assert "192.168.1.100" not in out
    assert "10.0.0.5" not in out
    assert "[REDACTED_IP]" in out


def test_redact_api_keys():
    from src.data.sanitization import redact_secrets_regex

    text = "Use sk-proj-AbCdEf1234567890abcdefghij or hf_XxXxXxXxXxXxXxXxXxXx for auth."
    out = redact_secrets_regex(text)
    assert "sk-proj-AbCdEf" not in out
    assert "hf_XxXxXxXx" not in out
    assert "[REDACTED_KEY]" in out


def test_redact_preserves_content():
    """Technical content must survive sanitization."""
    from src.data.sanitization import redact_secrets_regex

    text = "The 8051 microcontroller uses MOV A, R0 opcode at address 0x20."
    out = redact_secrets_regex(text)
    assert out == text, "innocuous hex + registers should NOT be redacted"


def test_anonymize_person_names_via_ner():
    """Full-name mentions become [PERSON]."""
    pytest.importorskip("spacy")
    from src.data.sanitization import anonymize_names_ner

    text = "John Smith asked Maria Gonzalez about the ESP32 driver."
    out = anonymize_names_ner(text)
    assert "John Smith" not in out
    assert "Maria Gonzalez" not in out
    assert "[PERSON]" in out
    # domain term must survive
    assert "ESP32" in out


def test_sanitize_pipeline_combines_both():
    pytest.importorskip("spacy")
    from src.data.sanitization import sanitize

    text = "John Smith at john@example.com debugged 10.0.0.5 with sk-proj-abcdefghij1234567890."
    out = sanitize(text)
    for pii in ["John Smith", "john@example.com", "10.0.0.5", "sk-proj-"]:
        assert pii not in out, f"pii {pii!r} leaked into sanitized output: {out!r}"
```

- [ ] **Step 3: Run to verify fail**

```bash
uv run python -m pytest tests/data/test_sanitization.py -v 2>&1 | tail -12
```

Expected: 6/6 FAIL with `ModuleNotFoundError: No module named 'src.data.sanitization'`.

- [ ] **Step 4: Commit**

```bash
git add tests/data/__init__.py tests/data/test_sanitization.py
git commit -m "test(c3): sanitization tests (red)"
```

---

### Task 3: Implement `sanitization.py` + install deps

**Files:**
- Create: `src/data/sanitization.py`
- Create: `src/data/__init__.py` (empty)
- Modify: `pyproject.toml` (add spacy to eval extras)

- [ ] **Step 1: Add spacy + hdbscan to eval extras**

In `pyproject.toml`, append to the `eval` list (already contains scikit-learn + matplotlib):

```toml
eval = [
  "scikit-learn>=1.5",
  "matplotlib>=3.9",
  "spacy>=3.8",
  "hdbscan>=0.8.40",
]
```

- [ ] **Step 2: Install + download spacy English model**

```bash
uv sync --extra eval
uv run python -m spacy download en_core_web_sm
```

Expected: both succeed. Model download is ~13MB.

- [ ] **Step 3: Create `src/data/__init__.py`** — empty file.

- [ ] **Step 4: Create `src/data/sanitization.py`** with EXACTLY this content:

```python
"""Regex + NER-based PII and secret removal for C3 real-dialogue corpus.

- Regex sweep: emails, IPv4, common API key prefixes (sk-, hf_, xai-, ghp_, gho_).
- NER anonymisation: spaCy en_core_web_sm, replaces PERSON entities with [PERSON].
- Tokens matching our domain vocabulary (ESP32, STM32, hex literals, opcodes) are NOT
  touched even if they look like alphanumeric secrets.
"""
from __future__ import annotations

import re

# Order matters: more specific patterns first so generic regex doesn't swallow them
_PATTERNS = [
    ("[REDACTED_EMAIL]", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("[REDACTED_IP]", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    # Common API key prefixes (length-bounded to avoid matching random alphanumerics)
    ("[REDACTED_KEY]", re.compile(r"\b(?:sk-(?:proj-)?|hf_|xai-|ghp_|gho_|ghs_|xoxb-|xoxp-)[A-Za-z0-9_-]{10,}\b")),
    # Long bare hex secrets (>= 32 chars)
    ("[REDACTED_HEX_SECRET]", re.compile(r"\b[a-f0-9]{32,}\b", re.IGNORECASE)),
]


def redact_secrets_regex(text: str) -> str:
    """Apply all regex patterns in order to redact secrets/PII."""
    for replacement, pattern in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


_NLP = None


def _load_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def anonymize_names_ner(text: str) -> str:
    """Replace PERSON entities identified by spaCy NER with [PERSON]."""
    nlp = _load_nlp()
    doc = nlp(text)
    # Build replacement list sorted by position descending (so indices stay valid)
    persons = [(ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ == "PERSON"]
    persons.sort(reverse=True)
    out = text
    for start, end in persons:
        out = out[:start] + "[PERSON]" + out[end:]
    return out


def sanitize(text: str) -> str:
    """Full pipeline: regex + NER."""
    return anonymize_names_ner(redact_secrets_regex(text))
```

- [ ] **Step 5: Run tests to verify pass**

```bash
uv run python -m pytest tests/data/test_sanitization.py -v 2>&1 | tail -12
```

Expected: 6/6 PASSED. If `test_redact_preserves_content` fails, your regex is too aggressive — tighten it to require prefix markers (`sk-`, `hf_`, etc.) for API keys, which the template already does.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/data/__init__.py src/data/sanitization.py
git commit -m "feat(c3): regex+NER sanitization pipeline"
```

---

### Task 4: Write failing tests for `corpus_validator.py`

**Files:**
- Create: `tests/data/test_corpus_validator.py`

- [ ] **Step 1: Create** with EXACTLY this content:

```python
"""Tests for src/data/corpus_validator.py — cluster vs existing taxonomy matching."""
from __future__ import annotations

import numpy as np
import pytest


def test_hungarian_match_perfect_overlap():
    """When K clusters exactly match K domains (after permutation), overlap = 1.0."""
    from src.data.corpus_validator import match_clusters_to_domains

    # 3 domains, 3 clusters, perfectly separated after relabelling
    true_domain = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    cluster_id  = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])  # permuted
    m = match_clusters_to_domains(true_domain, cluster_id, n_domains=3)
    assert m["mean_overlap"] == pytest.approx(1.0, abs=1e-9)
    assert m["assignment"] == {2: 0, 0: 1, 1: 2}


def test_hungarian_match_zero_overlap():
    """Entirely mismatched clusters give mean_overlap = 1 / n_domains (chance)."""
    from src.data.corpus_validator import match_clusters_to_domains

    rng = np.random.default_rng(0)
    true_domain = rng.integers(0, 5, size=200)
    cluster_id = rng.integers(0, 5, size=200)
    m = match_clusters_to_domains(true_domain, cluster_id, n_domains=5)
    # Chance overlap for uniform random assignment ≈ 1/5 = 0.2 (± noise)
    assert 0.1 < m["mean_overlap"] < 0.35


def test_hungarian_match_unequal_k_and_n_domains():
    """More clusters than domains: only n_domains clusters get assigned."""
    from src.data.corpus_validator import match_clusters_to_domains

    true_domain = np.array([0, 0, 0, 1, 1, 1])
    cluster_id  = np.array([0, 0, 1, 2, 2, 3])  # 4 clusters, 2 domains
    m = match_clusters_to_domains(true_domain, cluster_id, n_domains=2)
    assert len(m["assignment"]) == 2  # only 2 clusters mapped to domains
    # Domain 0 is 3 items (all cluster 0 majority). Domain 1 is 3 items, split across clusters 2 & 3.
    assert m["mean_overlap"] >= 0.5  # better than chance


def test_cluster_embeddings_returns_labels():
    from src.data.corpus_validator import cluster_embeddings_hdbscan

    rng = np.random.default_rng(0)
    # 3 well-separated clusters
    centers = rng.uniform(-5, 5, size=(3, 16))
    X = np.vstack([centers[i] + rng.normal(0, 0.3, size=(30, 16)) for i in range(3)])
    labels = cluster_embeddings_hdbscan(X, min_cluster_size=10)
    assert len(labels) == 90
    # HDBSCAN may label noise as -1; count unique non-noise clusters
    uniq = set(labels) - {-1}
    assert 2 <= len(uniq) <= 4, f"expected ~3 clusters, got {len(uniq)}"
```

- [ ] **Step 2: Run to verify fail**

```bash
uv run python -m pytest tests/data/test_corpus_validator.py -v 2>&1 | tail -12
```

Expected: 4/4 FAIL with `ModuleNotFoundError: No module named 'src.data.corpus_validator'`.

- [ ] **Step 3: Commit**

```bash
git add tests/data/test_corpus_validator.py
git commit -m "test(c3): corpus validator tests (red)"
```

---

### Task 5: Implement `corpus_validator.py`

**Files:**
- Create: `src/data/corpus_validator.py`

- [ ] **Step 1: Create** with EXACTLY this content:

```python
"""Corpus validator: cluster real-dialogue embeddings and match to existing taxonomy.

Uses HDBSCAN (density-based) so we don't force a fixed K; noise points (-1)
remain unlabelled. Matches clusters to the fixed 10-domain taxonomy via
Hungarian assignment on a confusion-matrix-style overlap cost.
"""
from __future__ import annotations

import numpy as np


def cluster_embeddings_hdbscan(X: np.ndarray, *, min_cluster_size: int = 20,
                                min_samples: int | None = None) -> np.ndarray:
    """Run HDBSCAN on X, return cluster labels (noise = -1)."""
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
        metric="euclidean",
    )
    return clusterer.fit_predict(X)


def match_clusters_to_domains(true_domain: np.ndarray, cluster_id: np.ndarray,
                               n_domains: int) -> dict:
    """Hungarian assignment: each cluster → one domain, maximising overlap.

    Returns dict with 'assignment' (cluster_id → domain_idx), 'mean_overlap'
    (per-domain accuracy under the assignment), 'per_domain_overlap' (list).
    """
    from scipy.optimize import linear_sum_assignment

    # Exclude noise cluster (-1) from assignment
    valid_mask = cluster_id != -1
    td = true_domain[valid_mask]
    ci = cluster_id[valid_mask]

    unique_clusters = sorted(set(int(c) for c in ci))
    n_clusters = len(unique_clusters)
    # Confusion matrix: rows = clusters, cols = domains, entries = overlap count
    conf = np.zeros((n_clusters, n_domains), dtype=np.int64)
    for ck_idx, ck in enumerate(unique_clusters):
        mask = ci == ck
        for d in range(n_domains):
            conf[ck_idx, d] = int(((td == d) & mask).sum())

    # Hungarian on NEGATIVE overlap (it minimises cost)
    # linear_sum_assignment requires rectangular; pad if n_clusters > n_domains by
    # truncating (extra clusters unassigned) or n_domains > n_clusters (some domains
    # unmatched — we still report).
    if n_clusters >= n_domains:
        cost = -conf[:, :n_domains]
        row_ind, col_ind = linear_sum_assignment(cost)
        assignment = {unique_clusters[r]: int(col_ind[i]) for i, r in enumerate(row_ind)}
    else:
        cost = -conf.T  # shape (n_domains, n_clusters)
        row_ind, col_ind = linear_sum_assignment(cost[:n_clusters])
        assignment = {unique_clusters[int(c)]: int(r) for r, c in zip(row_ind, col_ind)}

    # Compute mean overlap per assigned cluster
    overlaps = []
    for ck, d in assignment.items():
        ck_mask = ci == ck
        overlaps.append(float((td[ck_mask] == d).mean()) if ck_mask.sum() > 0 else 0.0)

    return {
        "assignment": assignment,
        "mean_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "per_cluster_overlap": overlaps,
        "n_noise_points": int((~valid_mask).sum()),
        "n_valid_points": int(valid_mask.sum()),
    }
```

- [ ] **Step 2: Run tests to verify pass**

```bash
uv run python -m pytest tests/data/test_corpus_validator.py -v 2>&1 | tail -10
```

Expected: 4/4 PASSED.

- [ ] **Step 3: Commit**

```bash
git add src/data/corpus_validator.py
git commit -m "feat(c3): HDBSCAN + Hungarian taxonomy matcher"
```

---

### Task 6: Implement the `augmenter.py` for under-represented domains

**Files:**
- Create: `src/data/augmenter.py`
- Create: `tests/data/test_augmenter.py`

- [ ] **Step 1: Write failing test**

Create `tests/data/test_augmenter.py`:

```python
"""Tests for src/data/augmenter.py — teacher-based synthetic completion."""
from __future__ import annotations

from unittest.mock import MagicMock


def test_augment_domain_uses_seeds_from_existing_samples():
    from src.data.augmenter import augment_domain_via_teacher

    seeds = [
        "How do I configure a Schmitt trigger input?",
        "What is the difference between BJT and MOSFET biasing?",
    ]
    teacher = MagicMock(return_value="A synthetic electronics question about ...")

    augmented = augment_domain_via_teacher(
        domain="electronics",
        seeds=seeds,
        n_to_generate=5,
        teacher_fn=teacher,
    )
    assert len(augmented) == 5
    assert all(isinstance(x, str) for x in augmented)
    assert teacher.call_count == 5
    # Every call must have received at least one seed in its prompt
    for call in teacher.call_args_list:
        prompt = call.args[0] if call.args else call.kwargs.get("prompt", "")
        has_seed = any(s in prompt for s in seeds)
        assert has_seed, f"teacher prompt missing seed: {prompt!r}"


def test_augment_domain_zero_n_returns_empty():
    from src.data.augmenter import augment_domain_via_teacher

    out = augment_domain_via_teacher(
        domain="x", seeds=["s"], n_to_generate=0, teacher_fn=MagicMock()
    )
    assert out == []
```

- [ ] **Step 2: Create `src/data/augmenter.py`** with EXACTLY this content:

```python
"""Teacher-based synthetic completion for under-represented domains.

When C3's real corpus has fewer than target_n samples in a domain, we seed the
teacher LLM with actual real fragments and ask it to generate similar ones.
The seeds keep the synthetic output domain-coherent; the teacher's coverage
provides diversity.
"""
from __future__ import annotations

import random
from typing import Callable

_PROMPT = """You are helping expand a training corpus in the domain of {domain}.

Given these real example questions from the domain:

{seeds}

Generate ONE more question in the same domain, style, and technical depth. Do not explain, do not preface. Output only the question text itself."""


def augment_domain_via_teacher(
    domain: str,
    seeds: list[str],
    n_to_generate: int,
    teacher_fn: Callable[[str], str],
    seeds_per_prompt: int = 3,
    random_state: int = 0,
) -> list[str]:
    """Use the teacher LLM to generate n_to_generate new questions in the domain.

    Each generation prompt includes a random subset of `seeds` for grounding.
    """
    if n_to_generate <= 0:
        return []
    rng = random.Random(random_state)
    out: list[str] = []
    for _ in range(n_to_generate):
        sample_seeds = rng.sample(seeds, min(seeds_per_prompt, len(seeds)))
        seed_block = "\n".join(f"- {s}" for s in sample_seeds)
        prompt = _PROMPT.format(domain=domain, seeds=seed_block)
        generated = teacher_fn(prompt).strip()
        out.append(generated)
    return out
```

- [ ] **Step 3: Run tests**

```bash
uv run python -m pytest tests/data/test_augmenter.py -v 2>&1 | tail -8
```

Expected: 2/2 PASSED.

- [ ] **Step 4: Commit**

```bash
git add tests/data/test_augmenter.py src/data/augmenter.py
git commit -m "feat(c3): teacher-seeded domain augmenter"
```

---

### Task 7: Build the end-to-end pipeline script

**Files:**
- Create: `scripts/build_corpus_real.py`

- [ ] **Step 1: Create** with EXACTLY this content:

```python
#!/usr/bin/env python3
"""Phase C3 end-to-end: ingest logs → sanitize → cluster → augment → write corpus.

Usage:
    uv run python scripts/build_corpus_real.py \\
        --source-manifest docs/c3-source-inventory.md \\
        --target-per-domain 500 \\
        --teacher-url http://studio:18000 \\
        --teacher-model qwen3-coder-480b-mxfp4 \\
        --output-dir data/corpus-real
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.augmenter import augment_domain_via_teacher
from src.data.corpus_validator import (
    cluster_embeddings_hdbscan,
    match_clusters_to_domains,
)
from src.data.sanitization import sanitize

logger = logging.getLogger(__name__)


DOMAINS = [
    "dsp", "electronics", "emc", "embedded", "freecad",
    "kicad-dsl", "platformio", "power", "spice", "stm32",
]


def _ingest_source(path: Path) -> list[str]:
    """Read a jsonl file, extract the 'content' field of user-authored turns."""
    texts: list[str] = []
    with open(path) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Support both {role, content} and {question, ...} schemas
            if "content" in obj and obj.get("role") == "user":
                texts.append(obj["content"])
            elif "question" in obj:
                texts.append(obj["question"])
    return texts


def _embed(texts: list[str], backbone: str = "models/niche-embeddings",
           seq_len: int = 32) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(backbone, device="cpu")
    tok = st.tokenizer
    m = st[0].auto_model.to("cpu")
    out = []
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=seq_len, padding="max_length")
        with torch.no_grad():
            h = m(**enc).last_hidden_state
        out.append(h.squeeze(0).mean(dim=0).cpu().numpy())
    return np.stack(out).astype(np.float64)


def _teacher_call(url: str, model: str, prompt: str) -> str:
    resp = requests.post(
        f"{url}/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": 200, "temperature": 0.7},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", nargs="+", required=True,
                   help="JSONL files with user questions (per docs/c3-source-inventory.md)")
    p.add_argument("--target-per-domain", type=int, default=500)
    p.add_argument("--teacher-url", default="http://studio:18000")
    p.add_argument("--teacher-model", default="qwen3-coder-480b-mxfp4")
    p.add_argument("--existing-data-dir", type=Path, default=Path("data/final"),
                   help="Used to label clusters against the known 10-domain taxonomy")
    p.add_argument("--output-dir", type=Path, default=Path("data/corpus-real"))
    p.add_argument("--stats-output", type=Path, default=Path("results/c3-corpus-stats.json"))
    p.add_argument("--dry-run-augment", action="store_true",
                   help="Skip real teacher calls; use stub augmentation")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # STAGE 1 — Ingest + sanitize
    raw_texts: list[str] = []
    for src in args.sources:
        t = _ingest_source(Path(src))
        logger.info("ingested %d texts from %s", len(t), src)
        raw_texts.extend(t)
    if not raw_texts:
        logger.error("no texts ingested — check --sources paths")
        return 2

    logger.info("sanitizing %d texts (regex + NER)…", len(raw_texts))
    clean_texts = [sanitize(t) for t in raw_texts]

    # STAGE 2 — Embed + cluster
    logger.info("embedding %d texts…", len(clean_texts))
    embs = _embed(clean_texts)

    logger.info("clustering with HDBSCAN…")
    cluster_ids = cluster_embeddings_hdbscan(embs, min_cluster_size=30)
    logger.info("  found %d clusters (excluding noise)", len(set(cluster_ids) - {-1}))

    # Match clusters to existing taxonomy:
    # 1. Load a sample from data/final per domain, embed it, get its "centroid"
    # 2. Assign each cluster to the nearest centroid (greedy — avoids needing
    #    a real ground-truth cluster-to-domain mapping)
    from src.routing.text_jepa.dataset import load_domain_corpus
    existing_samples = load_domain_corpus(args.existing_data_dir, domains=DOMAINS, max_per_domain=50)
    existing_texts = [s.text for s in existing_samples]
    existing_labels = np.array(
        [DOMAINS.index(s.domain) for s in existing_samples], dtype=np.int64
    )
    existing_embs = _embed(existing_texts)
    existing_centroids = np.stack(
        [existing_embs[existing_labels == d].mean(axis=0) for d in range(len(DOMAINS))]
    )  # shape (10, 384)

    # For each non-noise point, its soft label is nearest-centroid
    assigned_domain = np.full(len(embs), -1, dtype=np.int64)
    for i in range(len(embs)):
        if cluster_ids[i] == -1:
            continue
        dists = np.linalg.norm(existing_centroids - embs[i], axis=1)
        assigned_domain[i] = int(np.argmin(dists))

    # Evaluate cluster-to-taxonomy cohesion
    match = match_clusters_to_domains(assigned_domain, cluster_ids, n_domains=len(DOMAINS))
    logger.info("  cluster-taxonomy mean_overlap = %.3f (chance ≈ 0.1)", match["mean_overlap"])

    # STAGE 3 — Count per domain, augment under-represented
    per_domain_texts: dict[int, list[str]] = defaultdict(list)
    for i, d in enumerate(assigned_domain):
        if d >= 0:
            per_domain_texts[int(d)].append(clean_texts[i])

    stats = {
        "stage1_raw": len(raw_texts),
        "stage1_sanitized": len(clean_texts),
        "stage2_noise_points": int((assigned_domain == -1).sum()),
        "stage2_cluster_taxonomy_overlap": match["mean_overlap"],
        "stage3_per_domain_real": {DOMAINS[d]: len(v) for d, v in per_domain_texts.items()},
        "stage3_per_domain_augmented": {},
        "stage3_per_domain_final": {},
    }

    def teacher(prompt: str) -> str:
        if args.dry_run_augment:
            return "SYNTHETIC: placeholder question about the domain."
        return _teacher_call(args.teacher_url, args.teacher_model, prompt)

    for d, name in enumerate(DOMAINS):
        have = len(per_domain_texts[d])
        need = max(0, args.target_per_domain - have)
        if need == 0:
            stats["stage3_per_domain_augmented"][name] = 0
            continue
        seeds = per_domain_texts[d] if have > 0 else [f"Example question for {name}"]
        logger.info("augmenting %s: %d real + %d synthetic → %d target",
                    name, have, need, args.target_per_domain)
        new_items = augment_domain_via_teacher(
            domain=name, seeds=seeds, n_to_generate=need, teacher_fn=teacher,
        )
        per_domain_texts[d].extend(new_items)
        stats["stage3_per_domain_augmented"][name] = len(new_items)

    for d, name in enumerate(DOMAINS):
        stats["stage3_per_domain_final"][name] = len(per_domain_texts[d])

    # STAGE 4 — Write output jsonl per domain
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for d, name in enumerate(DOMAINS):
        out_path = args.output_dir / name / "train.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for text in per_domain_texts[d]:
                f.write(json.dumps({"text": text, "domain": name}) + "\n")
        logger.info("  wrote %s (%d lines)", out_path, len(per_domain_texts[d]))

    # STAGE 5 — Stats JSON
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(json.dumps(stats, indent=2))
    logger.info("wrote %s", args.stats_output)

    print("\n=== C3 Corpus Stats ===")
    print(f"Raw texts:          {stats['stage1_raw']}")
    print(f"Sanitized:          {stats['stage1_sanitized']}")
    print(f"Cluster overlap:    {stats['stage2_cluster_taxonomy_overlap']:.3f}")
    print(f"{'Domain':<12} {'real':>6} {'aug':>6} {'final':>6}")
    for name in DOMAINS:
        print(f"{name:<12} "
              f"{stats['stage3_per_domain_real'].get(name, 0):>6d} "
              f"{stats['stage3_per_domain_augmented'].get(name, 0):>6d} "
              f"{stats['stage3_per_domain_final'].get(name, 0):>6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Add `data/corpus-real/*/raw/` to .gitignore**

Check: `grep -q 'corpus-real/.*raw' .gitignore || echo 'data/corpus-real/*/raw/' >> .gitignore`

Also add: `data/corpus-real/` as top-level ignore — we commit ONLY the sanitized JSONL, never the raw inputs. Simpler: `data/corpus-real/**/raw.jsonl`.

- [ ] **Step 3: Dry-run the script on a small input**

Create a mock source:

```bash
mkdir -p /tmp/c3
cat > /tmp/c3/mock.jsonl <<EOF
{"role": "user", "content": "How do I configure a Schmitt trigger input on STM32?"}
{"role": "user", "content": "Contact me at john@example.com about the FreeCAD model."}
{"role": "user", "content": "The ESP32 at 192.168.1.1 is drawing too much current."}
EOF

uv run python scripts/build_corpus_real.py \
    --sources /tmp/c3/mock.jsonl \
    --target-per-domain 5 \
    --output-dir /tmp/c3-out \
    --stats-output /tmp/c3-stats.json \
    --dry-run-augment 2>&1 | tail -20
```

Expected: runs to completion, prints a stats table, writes 10 jsonl files under `/tmp/c3-out/`, each with 5 entries (1-2 from the real seed + 3-4 synthetic stubs). None of the written files contain `john@example.com` or `192.168.1.1` (sanitized).

Verify sanitization:

```bash
grep -r 'john@example.com\|192.168.1' /tmp/c3-out/ && echo "LEAK DETECTED" || echo "sanitized clean"
```

Expected: `sanitized clean`.

- [ ] **Step 4: Commit the script + gitignore update**

```bash
git add scripts/build_corpus_real.py .gitignore
git commit -m "feat(c3): end-to-end corpus build pipeline"
```

---

### Task 8: Real run + audit + commit corpus

**Files:**
- Create: `data/corpus-real/<domain>/train.jsonl` × 10
- Create: `results/c3-corpus-stats.json`
- Create: `docs/c3-pii-audit.md`

- [ ] **Step 1: Real run against inventoried sources**

Using the source list from `docs/c3-source-inventory.md` Task 1, run:

```bash
uv run python scripts/build_corpus_real.py \
    --sources PATH_1 PATH_2 PATH_3 \
    --target-per-domain 500 \
    --teacher-url http://studio:18000 \
    --teacher-model qwen3-coder-480b-mxfp4 \
    --output-dir data/corpus-real \
    --stats-output results/c3-corpus-stats.json 2>&1 | tee results/.c3-build.log
```

Replace `PATH_1 PATH_2 ...` with actual paths from your inventory. Expected runtime: 15-30 min (dominated by embedding ~5-20k texts + augmentation teacher calls for gap-filling).

- [ ] **Step 2: PII audit pass**

Grep the entire output corpus for common PII patterns:

```bash
! grep -rE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' data/corpus-real/ && echo "no emails leaked"
! grep -rE '\b(?:\d{1,3}\.){3}\d{1,3}\b' data/corpus-real/ && echo "no IPs leaked"
! grep -rE '\b(sk-|hf_|xai-|ghp_|gho_|xoxb-)[A-Za-z0-9_-]{10,}\b' data/corpus-real/ && echo "no keys leaked"
```

All three lines should print "no ... leaked". If any grep finds a hit, the PII leaked through sanitization — STOP, do NOT commit the corpus, fix the regex, re-run.

Write the audit log to `docs/c3-pii-audit.md`:

```markdown
# C3 PII Audit (YYYY-MM-DD)

Grep sweep of `data/corpus-real/` after build:

| Pattern | Hits | Notes |
|---|---|---|
| email regex | 0 | clean |
| IPv4 | 0 | clean |
| API keys | 0 | clean |

Sanity checks:
- `wc -l data/corpus-real/*/train.jsonl` matches `results/c3-corpus-stats.json` counts.
- Random sample of 20 lines reviewed manually; no PII or secrets observed.
- Signed off 2026-04-XX.
```

- [ ] **Step 3: Commit**

```bash
git add data/corpus-real/ results/c3-corpus-stats.json docs/c3-pii-audit.md
git commit -m "data(c3): real dialogue corpus + PII audit clean"
```

Do NOT commit `results/.c3-build.log` (contains raw text before sanitization).

---

### Task 9: Re-run C1 against the real corpus + paper narrative

**Files:**
- Create: `results/c3-c1-rerun.json`
- Create: `docs/paper-a/c3-corpus-validation.md`

- [ ] **Step 1: Re-run C1 with the new corpus**

```bash
uv run python scripts/bench_classical_vs_vqc.py \
    --data-dir data/corpus-real \
    --domains dsp,electronics,emc,embedded,freecad,kicad-dsl,platformio,power,spice,stm32 \
    --max-per-domain 50 \
    --backbone models/niche-embeddings \
    --embeddings-npz results/.c3-cache.npz \
    --seeds 0,1,2,3,4 \
    --epochs 300 \
    --output results/c3-c1-rerun.json
```

Expected: same 5 baselines, different numbers. Compare to C1:
- If `logreg` raw accuracy on real corpus is still above 0.80 kill threshold, Paper A's premise that routing is non-trivial is refuted for real data — document and pivot.
- If `torch_vqc` accuracy is dramatically different (>15 pt) on real data, the synthetic corpus was misleading — a finding to highlight.

- [ ] **Step 2: Write paper narrative**

Create `docs/paper-a/c3-corpus-validation.md`:

```markdown
# C3: Real Corpus Validation

**Setup.** Replaced the synthetic `data/final/` corpus (used in C1, C2, C5) with a curated real-dialogue corpus built from {SOURCES} via the pipeline in `scripts/build_corpus_real.py` (ingest → regex + NER sanitize → HDBSCAN cluster → teacher-seeded augmentation for under-represented domains).

**Source statistics** (from `results/c3-corpus-stats.json`):
- Raw texts: {raw}
- After sanitization: {sanitized}
- Noise cluster points: {noise}
- Cluster-taxonomy overlap: {overlap} (chance ≈ 0.1)

**Per-domain composition:**

| Domain | Real | Augmented | Total |
|---|---|---|---|
(fill with stats JSON values)

**C1 re-run on real corpus** (5 seeds, 300 epochs, 50/domain):

| Baseline | Synthetic (C1) | Real (C3) | Δ |
|---|---|---|---|
| Stratified | 0.118 | {c3_stratified} | ... |
| LogReg PCA-4 | 0.364 | {c3_logreg_pca} | ... |
| Torch VQC | 0.246 | {c3_vqc} | ... |
| MLP | 0.546 | {c3_mlp} | ... |
| LogReg raw | 0.546 | {c3_logreg} | ... |

**Interpretation.**

1. **Corpus validity.** Cluster-taxonomy overlap of {overlap} vs chance 0.10 indicates real dialogues DO fall into our 10-domain taxonomy, but less cleanly than synthetic data. HDBSCAN finds [N] natural clusters vs our 10 hand-designed domains, suggesting [agreement / disagreement].

2. **Does C1's result generalise?** On real data, classical baseline LogReg raw hits {c3_logreg} (vs 0.546 synthetic). The VQC hits {c3_vqc}. The rank ordering [preserved / inverted] relative to synthetic — see table Δ column.

3. **Augmentation disclosure.** {aug_pct}% of the training corpus is synthetic (teacher-generated) due to gaps in real coverage of {underrep_domains}. This is a transparency note for reproducibility; Paper A discusses robustness in §6 "Limitations".

**Conclusion.** The real-corpus results {confirm / moderate / refute} the synthetic-corpus findings. Paper A's central numeric claims update to the real-corpus values. The synthetic-corpus results are retained as an ablation showing the [upper-bound / idealised-case] for each classifier.
```

- [ ] **Step 3: Commit**

```bash
git add results/c3-c1-rerun.json docs/paper-a/c3-corpus-validation.md
git commit -m "docs(c3): real-corpus rerun + validation narrative"
```

---

### Task 10: Push + roadmap update

- [ ] **Step 1: Full regression**

```bash
uv run python -m pytest tests/data/ tests/routing/test_classical_baselines.py -v
```

Expected: all PASSED.

- [ ] **Step 2: Push**

```bash
git push origin main
```

- [ ] **Step 3: Roadmap update**

In `docs/superpowers/plans/2026-04-19-phase-c-roadmap.md`, mark C3 done with commit range. Commit + push.

---

## Kill criteria (inline)

1. **Task 1 fallback**: if < 2000 usable lines across all sources, C3 is BLOCKED — need user decision (new data collection or de-scope).
2. **Task 8 PII audit**: if ANY grep finds a hit, STOP, do not commit corpus, fix regex, re-run.
3. **Task 9 validity**: if cluster-taxonomy overlap < 0.25 (chance + 2× margin), our 10-domain taxonomy does NOT match real dialogues — document as a finding and retract or pivot Paper A's taxonomy-specific claims.

## Out of scope

- Legal review of each source's ToS/consent — assumed done by user before inventory.
- GDPR "right to erasure" implementation — dataset is static; no ongoing deletion workflow.
- Comparative clustering (HDBSCAN vs KMeans vs Louvain) — HDBSCAN chosen by fiat.
- Cross-language: the sanitization spaCy model is English-only.
- Downstream re-runs of C2 on real corpus — separate plan once C2 is done.

## Total estimated time

- Task 1 (inventory): 1-2 hours (manual audit of logs)
- Task 2-6 (TDD pipeline): ~4 hours
- Task 7 (orchestrator): 1 hour
- Task 8 (real run + audit): 2-3 hours
- Task 9 (rerun + narrative): 1-2 hours
- Task 10 (push): 15 min

**Total: ~10-12 hours engineering + ~30 min compute, 2-3 days on calendar (dominated by source inventory + PII audit manual review).**
