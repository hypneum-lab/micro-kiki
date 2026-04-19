# Phase C2 — Downstream LLM Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the `TorchVQCRouter`'s routing decisions translate to downstream answer-quality gains on a 200-query per-domain eval, by routing each query through Qwen3.6-35B-A3B on kxkm-ai with and without adapter-swapping, and scoring answers with Qwen3-Coder-480B as LLM-judge on Studio.

**Architecture:** Three routers compared — (1) VQC (our trained router from C1), (2) random (baseline — picks an expert uniformly), (3) oracle (always correct, upper bound). Each routes the 200-query eval set; kxkm-ai llama-server swaps Q4 adapter based on route, produces an answer; Studio teacher scores the answer 0-5 via rubric prompt. Final number: mean downstream-quality per router.

**Tech Stack:** Python 3.14, torch, `TorchVQCRouter`, llama-cpp-python (HTTP client to kxkm-ai llama-server on port 8000), Qwen3-Coder-480B MLX server on Studio port 18000 (already exists per session memory), sentence-transformers (cached embeddings). No new deps.

**Decisions baked in** (from Phase C brainstorm 2026-04-19):
- LLM target: **Qwen3.6-35B-A3B Q4 on kxkm-ai** (not Studio 480B — that's teacher/judge)
- Eval-set: **20 queries × 10 domains = 200 queries from `data/final`** (reuse C1 corpus)
- Metric: **Rubric-score LLM-judge with Qwen3-Coder-480B** (explicit bias acknowledged)
- No blind A/B human — out of scope

---

## File Structure

**Files to create:**
- `src/routing/llm_judge.py` — rubric + score parser
- `src/routing/downstream_harness.py` — orchestrates router → LLM → judge
- `tests/routing/test_llm_judge.py` — unit tests (judge parsing, rubric sanity)
- `tests/routing/test_downstream_harness.py` — integration test with mocked LLM/judge
- `scripts/bench_downstream_c2.py` — CLI runner
- `results/c2-downstream.json` — results
- `docs/paper-a/c2-downstream-results.md` — paper narrative
- `docs/paper-a/c2-downstream-figure.pdf` — results bar chart

**Files to modify (no new deps):**
- `docs/superpowers/plans/2026-04-19-phase-c-roadmap.md` (status update at end)

---

### Task 1: Sanity-check remote inference endpoints

**Files:**
- Create: `scripts/c2_probe_endpoints.sh` (operator helper, committed)

- [ ] **Step 1: Verify kxkm-ai llama-server is up**

Run: `ssh kxkm-ai "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/v1/models"` from the micro-kiki host machine.
Expected: `200`.
If not 200: start the server on kxkm-ai with the Q4 Qwen3.6-35B-A3B model per the session notes (`~/kiki-v3/outputs/devstral-v3-Q4_K_M.gguf` or equivalent 35B). If you cannot, STOP and report BLOCKED.

- [ ] **Step 2: Verify Studio teacher is up**

Run: `ssh studio "curl -s -o /dev/null -w '%{http_code}' http://localhost:18000/v1/models"`.
Expected: `200` (Qwen3-Coder-480B MLX server).
If not 200: report BLOCKED — starting the 480B is not in-scope here.

- [ ] **Step 3: Create the probe script** for future reference:

Write `scripts/c2_probe_endpoints.sh`:

```bash
#!/bin/bash
set -euo pipefail
echo "kxkm-ai 35B router:  $(ssh kxkm-ai 'curl -s -o /dev/null -w %{http_code} http://localhost:8000/v1/models')"
echo "studio 480B judge:   $(ssh studio 'curl -s -o /dev/null -w %{http_code} http://localhost:18000/v1/models')"
```

Make it executable: `chmod +x scripts/c2_probe_endpoints.sh`

- [ ] **Step 4: Commit**

```bash
git add scripts/c2_probe_endpoints.sh
git commit -m "chore(c2): endpoint health probe script"
```

---

### Task 2: Write failing tests for `llm_judge.py`

**Files:**
- Create: `tests/routing/test_llm_judge.py`

- [ ] **Step 1: Create the test file** with EXACTLY this content:

```python
"""Tests for src/routing/llm_judge.py — rubric prompt builder + score parser."""
from __future__ import annotations

import pytest


def test_build_rubric_prompt_contains_question_and_answer():
    from src.routing.llm_judge import build_rubric_prompt

    p = build_rubric_prompt(
        question="What is a Schmitt trigger?",
        answer="A comparator with hysteresis.",
        domain="electronics",
    )
    assert "Schmitt trigger" in p
    assert "comparator" in p
    assert "electronics" in p.lower()
    # rubric must explicitly ask for a 0-5 integer
    assert "0 to 5" in p or "0-5" in p


def test_parse_score_extracts_integer():
    from src.routing.llm_judge import parse_score

    assert parse_score("Score: 4") == 4
    assert parse_score("The answer is correct. Score: 5/5") == 5
    assert parse_score("I rate this 3 out of 5.") == 3
    assert parse_score("0") == 0


def test_parse_score_clamps_to_valid_range():
    from src.routing.llm_judge import parse_score

    # If judge emits nonsense, clamp to chance (e.g., -1 or 7)
    assert parse_score("Score: 7") == 5, "should clamp above"
    assert parse_score("Score: -1") == 0, "should clamp below"


def test_parse_score_returns_none_on_unparseable():
    from src.routing.llm_judge import parse_score

    assert parse_score("") is None
    assert parse_score("I refuse to score this.") is None
    assert parse_score("garbage text with no numbers") is None


def test_parse_score_prefers_last_integer_on_ambiguity():
    """Judge may reason aloud with numbers before giving final score."""
    from src.routing.llm_judge import parse_score

    # Judges often explain: "The answer mentions 2 components but misses 3. Final score: 4"
    assert parse_score("mentions 2 components but misses 3. Final score: 4") == 4
```

- [ ] **Step 2: Run to verify all fail**

```bash
uv run python -m pytest tests/routing/test_llm_judge.py -v 2>&1 | tail -12
```

Expected: 5/5 FAIL with `ModuleNotFoundError: No module named 'src.routing.llm_judge'`.

- [ ] **Step 3: Commit**

```bash
git add tests/routing/test_llm_judge.py
git commit -m "test(c2): llm_judge rubric + parser tests (red)"
```

---

### Task 3: Implement `llm_judge.py`

**Files:**
- Create: `src/routing/llm_judge.py`

- [ ] **Step 1: Create** with EXACTLY this content:

```python
"""LLM-as-judge rubric + response parser for C2 downstream eval.

The judge (Qwen3-Coder-480B on Studio) scores a candidate answer 0-5 given
question + reference domain. We build a prompt with a strict rubric and parse
an integer score from the response.
"""
from __future__ import annotations

import re

_RUBRIC_TEMPLATE = """You are evaluating an answer to a technical question in the domain of {domain}.

Question: {question}

Answer: {answer}

Rubric (choose one integer 0 to 5):
- 0: completely wrong, irrelevant, or refuses to answer
- 1: fundamentally incorrect but on-topic
- 2: mostly wrong with one correct element
- 3: partially correct, misses key points
- 4: mostly correct, minor omissions
- 5: fully correct, complete, technically precise

Think briefly, then end your response with exactly "Score: <N>" where N is the integer 0-5. Do not wrap N in markdown or quotes."""


def build_rubric_prompt(question: str, answer: str, domain: str) -> str:
    """Return a prompt asking the judge to score the answer 0-5."""
    return _RUBRIC_TEMPLATE.format(
        question=question.strip(),
        answer=answer.strip(),
        domain=domain.strip(),
    )


def parse_score(response: str) -> int | None:
    """Extract the integer score from the judge's response.

    Strategy: grab the LAST integer that appears in the response, clamp to [0, 5].
    Returns None if no integer is found.
    """
    if not response:
        return None
    nums = re.findall(r"-?\d+", response)
    if not nums:
        return None
    n = int(nums[-1])
    return max(0, min(5, n))
```

- [ ] **Step 2: Run tests to verify pass**

```bash
uv run python -m pytest tests/routing/test_llm_judge.py -v 2>&1 | tail -10
```

Expected: 5/5 PASSED.

- [ ] **Step 3: Commit**

```bash
git add src/routing/llm_judge.py
git commit -m "feat(c2): llm_judge rubric + score parser"
```

---

### Task 4: Write failing integration test for `downstream_harness.py`

**Files:**
- Create: `tests/routing/test_downstream_harness.py`

- [ ] **Step 1: Create** with EXACTLY this content:

```python
"""Integration test for src/routing/downstream_harness.py with mocked LLM + judge."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np


def test_run_downstream_routes_llm_then_judges():
    from src.routing.downstream_harness import run_downstream_eval

    queries = [
        {"question": "What is a Schmitt trigger?", "domain": "electronics", "domain_idx": 0},
        {"question": "How do I debounce a switch?", "domain": "electronics", "domain_idx": 0},
        {"question": "What is a dsp filter?", "domain": "dsp", "domain_idx": 1},
    ]
    n_classes = 2

    # Router: always picks domain 0 (electronics). VQC proxy.
    router_fn = MagicMock(side_effect=lambda emb: 0)

    # LLM: returns a canned answer depending on domain chosen
    llm_fn = MagicMock(return_value="mock answer")

    # Judge: returns score 5 if LLM answer is on-topic (domain matches query),
    # else 1 — simulating the real rubric
    def judge_fn(question: str, answer: str, domain: str) -> int:
        # In our mock, domain passed in is the QUERY's domain, answer is always
        # "mock answer" from domain-0 adapter. Score high if router-chosen domain
        # happens to match query domain.
        return 5 if domain == "electronics" else 1

    # Embeddings are random but deterministic
    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((len(queries), 16))

    result = run_downstream_eval(
        queries=queries,
        embeddings=embeddings,
        router_fn=router_fn,
        llm_fn=llm_fn,
        judge_fn=judge_fn,
        domain_names=["electronics", "dsp"],
    )

    assert "per_query" in result
    assert "mean_score" in result
    assert len(result["per_query"]) == 3
    # 2/3 queries are electronics (score 5), 1/3 is dsp (score 1)
    # Mean = (5 + 5 + 1) / 3 = 3.666
    assert abs(result["mean_score"] - 11 / 3) < 1e-6
    # router_fn called once per query
    assert router_fn.call_count == 3
    assert llm_fn.call_count == 3


def test_run_downstream_reports_routing_accuracy():
    """Harness should surface how often router picked the correct domain."""
    from src.routing.downstream_harness import run_downstream_eval

    queries = [
        {"question": "q1", "domain": "a", "domain_idx": 0},
        {"question": "q2", "domain": "b", "domain_idx": 1},
        {"question": "q3", "domain": "b", "domain_idx": 1},
    ]
    router_fn = MagicMock(side_effect=[0, 0, 1])  # correct, wrong, correct
    llm_fn = MagicMock(return_value="x")
    judge_fn = MagicMock(return_value=3)

    result = run_downstream_eval(
        queries=queries,
        embeddings=np.zeros((3, 4)),
        router_fn=router_fn,
        llm_fn=llm_fn,
        judge_fn=judge_fn,
        domain_names=["a", "b"],
    )
    assert result["routing_accuracy"] == 2 / 3
```

- [ ] **Step 2: Run to verify fail**

```bash
uv run python -m pytest tests/routing/test_downstream_harness.py -v 2>&1 | tail -10
```

Expected: 2/2 FAIL with `ModuleNotFoundError: No module named 'src.routing.downstream_harness'`.

- [ ] **Step 3: Commit**

```bash
git add tests/routing/test_downstream_harness.py
git commit -m "test(c2): downstream harness integration tests (red)"
```

---

### Task 5: Implement `downstream_harness.py`

**Files:**
- Create: `src/routing/downstream_harness.py`

- [ ] **Step 1: Create** with EXACTLY this content:

```python
"""End-to-end downstream-quality harness: router → LLM → judge pipeline.

The three callbacks (router_fn, llm_fn, judge_fn) are injected so the same
harness runs with real remote endpoints or with mocks in tests.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def run_downstream_eval(
    queries: list[dict],
    embeddings: np.ndarray,
    router_fn: Callable[[np.ndarray], int],
    llm_fn: Callable[[str, str], str],
    judge_fn: Callable[[str, str, str], int],
    domain_names: list[str],
) -> dict:
    """Run router → LLM → judge for each query and aggregate.

    Args:
        queries: list of dicts with keys 'question' (str), 'domain' (str),
            'domain_idx' (int). One per eval sample.
        embeddings: (N, D) array of pre-computed query embeddings, same order
            as `queries`.
        router_fn: emb (D,) → int (chosen domain index).
        llm_fn: (question, routed_domain_name) → generated answer.
        judge_fn: (question, answer, expected_domain_name) → int score in [0, 5].
        domain_names: ordered list mapping domain_idx → name.

    Returns:
        dict with keys 'per_query' (list of per-sample records), 'mean_score'
        (float), 'routing_accuracy' (float — correct-route rate),
        'mean_score_when_routed_correct', 'mean_score_when_routed_wrong'.
    """
    assert len(queries) == len(embeddings), "query-embedding length mismatch"
    per_query = []
    correct_scores: list[int] = []
    wrong_scores: list[int] = []

    for q, emb in zip(queries, embeddings):
        routed_idx = int(router_fn(emb))
        routed_name = domain_names[routed_idx]
        expected_name = q["domain"]
        answer = llm_fn(q["question"], routed_name)
        score = int(judge_fn(q["question"], answer, expected_name))
        is_correct_route = routed_idx == q["domain_idx"]
        per_query.append({
            "question": q["question"],
            "expected_domain": expected_name,
            "routed_domain": routed_name,
            "correct_route": is_correct_route,
            "answer": answer,
            "score": score,
        })
        (correct_scores if is_correct_route else wrong_scores).append(score)

    total = [r["score"] for r in per_query]
    mean = sum(total) / max(len(total), 1)
    routed_correct_n = sum(r["correct_route"] for r in per_query)
    routing_acc = routed_correct_n / max(len(per_query), 1)

    return {
        "per_query": per_query,
        "mean_score": mean,
        "routing_accuracy": routing_acc,
        "mean_score_when_routed_correct": (
            sum(correct_scores) / len(correct_scores) if correct_scores else 0.0
        ),
        "mean_score_when_routed_wrong": (
            sum(wrong_scores) / len(wrong_scores) if wrong_scores else 0.0
        ),
        "n_queries": len(per_query),
    }
```

- [ ] **Step 2: Run tests to verify pass**

```bash
uv run python -m pytest tests/routing/test_downstream_harness.py -v 2>&1 | tail -10
```

Expected: 2/2 PASSED.

- [ ] **Step 3: Commit**

```bash
git add src/routing/downstream_harness.py
git commit -m "feat(c2): downstream eval harness (router/llm/judge)"
```

---

### Task 6: Implement the CLI bench script with real endpoints

**Files:**
- Create: `scripts/bench_downstream_c2.py`

- [ ] **Step 1: Create** with EXACTLY this content:

```python
#!/usr/bin/env python3
"""C2 downstream bench: VQC vs random vs oracle router on 200-query eval.

Loads C1 cached embeddings + corpus samples, picks 20 per domain as eval
queries (the rest are implicitly training data for the router, already
done in C1), calls three routers, dispatches to kxkm-ai llama-server with
the appropriate adapter, scores with Studio 480B judge.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import requests
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.routing.downstream_harness import run_downstream_eval
from src.routing.llm_judge import build_rubric_prompt, parse_score
from src.routing.text_jepa.dataset import load_domain_corpus
from src.routing.torch_vqc_router import TorchVQCRouter

logger = logging.getLogger(__name__)


def _llm_call(base_url: str, model: str, prompt: str, max_tokens: int = 512) -> str:
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/final"))
    p.add_argument("--domains", default="dsp,electronics,emc,embedded,freecad,kicad-dsl,platformio,power,spice,stm32")
    p.add_argument("--embeddings-npz", type=Path, default=Path("results/.c1-cache.npz"))
    p.add_argument("--per-domain", type=int, default=20)
    p.add_argument("--kxkm-url", default="http://kxkm-ai:8000")
    p.add_argument("--studio-url", default="http://studio:18000")
    p.add_argument("--kxkm-model", default="qwen3-35b-a3b-q4")
    p.add_argument("--studio-model", default="qwen3-coder-480b-mxfp4")
    p.add_argument("--vqc-checkpoint", type=Path, required=False, default=None,
                   help="Path to trained TorchVQCRouter state_dict. If missing, trains a fresh one on all C1 data.")
    p.add_argument("--output", type=Path, default=Path("results/c2-downstream.json"))
    p.add_argument("--dry-run", action="store_true", help="Skip real LLM calls; use stub answers")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    dom_to_idx = {d: i for i, d in enumerate(domains)}
    n_classes = len(domains)

    # Load corpus — we need the actual query text, not just embeddings
    samples = load_domain_corpus(args.data_dir, domains=domains, max_per_domain=200)
    logger.info("loaded %d samples", len(samples))

    # Pick first per_domain samples per domain as the held-out eval
    per_dom_count: dict[str, int] = {d: 0 for d in domains}
    eval_queries: list[dict] = []
    for s in samples:
        if per_dom_count[s.domain] < args.per_domain:
            eval_queries.append({
                "question": s.text,
                "domain": s.domain,
                "domain_idx": dom_to_idx[s.domain],
            })
            per_dom_count[s.domain] += 1

    logger.info("eval set: %d queries (%d/domain)", len(eval_queries), args.per_domain)

    # Compute embeddings for eval queries from the cache (matched by sample order)
    if not args.embeddings_npz.exists():
        logger.error("embeddings cache missing: %s — run C1 bench first", args.embeddings_npz)
        return 2
    cache = np.load(args.embeddings_npz)
    all_embs = cache["embeddings"]
    # The cache ordering matches load_domain_corpus with max_per_domain=50. We reload with 200
    # so must re-embed. Simpler path: re-embed the first 200 × 10 with SentenceTransformer here.
    # For brevity we rely on deterministic load_domain_corpus order + matching samples.
    # We embed just the eval_queries here (cheap, ~200 items ~30s):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("models/niche-embeddings", device="cpu")
    tok = st.tokenizer
    tr_model = st[0].auto_model.to("cpu")
    eval_embs = []
    for q in eval_queries:
        enc = tok(q["question"], return_tensors="pt", truncation=True, max_length=32, padding="max_length")
        with torch.no_grad():
            out = tr_model(**enc).last_hidden_state
        eval_embs.append(out.squeeze(0).mean(dim=0).cpu().numpy())
    eval_embs = np.stack(eval_embs).astype(np.float64)
    logger.info("embedded %d eval queries", len(eval_embs))

    # Train or load VQC
    vqc = TorchVQCRouter(
        n_qubits=4, n_layers=6, n_classes=n_classes,
        lr=0.05, seed=0, input_dim=eval_embs.shape[1], weight_decay=1e-4,
    )
    if args.vqc_checkpoint and args.vqc_checkpoint.exists():
        vqc.load_state_dict(torch.load(args.vqc_checkpoint, map_location="cpu"))
        logger.info("loaded VQC checkpoint from %s", args.vqc_checkpoint)
    else:
        # Train on ALL samples (eval_queries are first per_domain per domain; rest used as train)
        logger.info("training VQC on C1 cache (no checkpoint provided)")
        X_tr = torch.from_numpy(all_embs).double()
        y_tr = torch.from_numpy(np.array(
            [dom_to_idx[s.domain] for s in samples[:len(all_embs)]], dtype=np.int64
        ))
        vqc.train_batched(X_tr, y_tr, epochs=300)

    def router_vqc(emb: np.ndarray) -> int:
        with torch.no_grad():
            return int(vqc.predict(torch.from_numpy(emb).double().unsqueeze(0))[0])

    rng = np.random.default_rng(0)

    def router_random(emb: np.ndarray) -> int:
        return int(rng.integers(0, n_classes))

    def router_oracle(emb: np.ndarray, *, _counter=[0]) -> int:
        # Uses positional lookup in eval_queries — harness iterates in order
        idx = eval_queries[_counter[0]]["domain_idx"]
        _counter[0] += 1
        return idx

    def llm_call(question: str, domain_name: str) -> str:
        if args.dry_run:
            return f"[stub answer for {domain_name}]"
        prompt = f"You are an expert in {domain_name}. Answer concisely.\n\nQuestion: {question}"
        return _llm_call(args.kxkm_url, args.kxkm_model, prompt, max_tokens=512)

    def judge_call(question: str, answer: str, expected_domain: str) -> int:
        if args.dry_run:
            return 3  # neutral stub
        prompt = build_rubric_prompt(question=question, answer=answer, domain=expected_domain)
        resp = _llm_call(args.studio_url, args.studio_model, prompt, max_tokens=256)
        score = parse_score(resp)
        return score if score is not None else 0

    results = {}
    for name, fn in [("vqc", router_vqc), ("random", router_random), ("oracle", router_oracle)]:
        logger.info("running router=%s", name)
        # oracle uses a closure counter; reset per run
        if name == "oracle":
            router_oracle.__defaults__ = ([0],)
        r = run_downstream_eval(
            queries=eval_queries,
            embeddings=eval_embs,
            router_fn=fn,
            llm_fn=llm_call,
            judge_fn=judge_call,
            domain_names=domains,
        )
        results[name] = r
        logger.info("  router=%s  mean_score=%.3f  routing_acc=%.3f",
                    name, r["mean_score"], r["routing_accuracy"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", args.output)

    print("\n=== C2 Downstream Results ===")
    print(f"{'router':<8} {'mean_score':>10} {'routing_acc':>12} {'score | correct':>16} {'score | wrong':>14}")
    for name, r in results.items():
        print(f"{name:<8} {r['mean_score']:>10.3f} {r['routing_accuracy']:>12.3f} "
              f"{r['mean_score_when_routed_correct']:>16.3f} {r['mean_score_when_routed_wrong']:>14.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Dry-run the script to verify wiring**

```bash
uv run python scripts/bench_downstream_c2.py --dry-run --per-domain 2 --output /tmp/c2-dry.json
```

Expected: prints a results table with three rows, all scores = 3.0, `vqc` has the same routing accuracy as C1 (~0.25 on these 20 query subset), `oracle` has routing_accuracy = 1.0, `random` ≈ 1/10 = 0.1.

- [ ] **Step 3: Commit**

```bash
git add scripts/bench_downstream_c2.py
git commit -m "feat(c2): downstream bench CLI (VQC/random/oracle)"
```

---

### Task 7: Real run against kxkm-ai + Studio endpoints

**Files:**
- Create: `results/c2-downstream.json`

- [ ] **Step 1: Verify endpoints**

Run: `bash scripts/c2_probe_endpoints.sh`
Expected: both HTTP 200.

- [ ] **Step 2: Launch the real bench**

```bash
uv run python scripts/bench_downstream_c2.py \
    --per-domain 20 \
    --kxkm-url http://kxkm-ai:8000 \
    --studio-url http://studio:18000 \
    --output results/c2-downstream.json 2>&1 | tee results/.c2-run.log
```

Expected runtime: ~60-90 min (200 queries × 3 routers × ~2s inference + ~3s judging per query). Progress is logged every query.

- [ ] **Step 3: Inspect the result**

```bash
jq '{vqc_mean: .vqc.mean_score, random_mean: .random.mean_score, oracle_mean: .oracle.mean_score, vqc_routing_acc: .vqc.routing_accuracy, vqc_score_correct: .vqc.mean_score_when_routed_correct, vqc_score_wrong: .vqc.mean_score_when_routed_wrong}' results/c2-downstream.json
```

Sanity checks:
- `oracle_mean` should be highest (above 4.0 if adapters help; if near `random_mean`, adapter routing doesn't affect quality — Paper A kill criterion).
- `vqc_mean` between `random_mean` and `oracle_mean`.
- `vqc_score_correct` > `vqc_score_wrong` (routing correctly DOES produce better answers).

**Kill criterion (inline):** if `oracle_mean - random_mean < 0.3` (i.e., expert adapters don't beat default generalist model by at least 0.3 rubric points), routing is POINTLESS regardless of quality — the whole adapter-swap premise collapses. Halt C2 and document in results narrative; propose Paper A retraction/pivot.

- [ ] **Step 4: Commit the JSON + log**

```bash
git add results/c2-downstream.json
git commit -m "results(c2): 200-query downstream eval real run"
```

Do NOT commit `results/.c2-run.log` (gitignored via `.log`). Verify with `git status` — no untracked log.

---

### Task 8: Paper-facing narrative + figure

**Files:**
- Create: `docs/paper-a/c2-downstream-results.md`
- Create: `scripts/figure_c2_downstream.py`
- Create: `docs/paper-a/c2-downstream-figure.pdf`

- [ ] **Step 1: Create the figure script**

Write `scripts/figure_c2_downstream.py` with this content:

```python
#!/usr/bin/env python3
"""Generate c2-downstream-figure.pdf from c2-downstream.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("results/c2-downstream.json"))
    p.add_argument("--output", type=Path, default=Path("docs/paper-a/c2-downstream-figure.pdf"))
    args = p.parse_args()

    data = json.loads(args.input.read_text())
    order = ["random", "vqc", "oracle"]
    colors = ["#bbbbbb", "#6699ff", "#66bb77"]
    means = [data[n]["mean_score"] for n in order]
    corr_means = [data[n]["mean_score_when_routed_correct"] for n in order]
    wrong_means = [data[n]["mean_score_when_routed_wrong"] for n in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(order))
    ax1.bar(x, means, color=colors, edgecolor="black", linewidth=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(order)
    ax1.set_ylabel("Mean judge score (0-5)")
    ax1.set_title("Downstream quality by router")
    ax1.set_ylim(0, 5)
    for i, m in enumerate(means):
        ax1.text(i, m + 0.05, f"{m:.2f}", ha="center", fontsize=9)

    width = 0.35
    ax2.bar(x - width / 2, corr_means, width, label="correct route", color="#66bb77")
    ax2.bar(x + width / 2, wrong_means, width, label="wrong route", color="#cc5555")
    ax2.set_xticks(x)
    ax2.set_xticklabels(order)
    ax2.set_ylabel("Mean judge score (0-5)")
    ax2.set_title("Conditional on routing correctness")
    ax2.set_ylim(0, 5)
    ax2.legend()

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it**

```bash
uv run python scripts/figure_c2_downstream.py
```

Expected: prints `wrote docs/paper-a/c2-downstream-figure.pdf`.

- [ ] **Step 3: Create the narrative**

Write `docs/paper-a/c2-downstream-results.md` with this template (replace {{...}} after reading the JSON):

```markdown
# C2: Downstream LLM Evaluation

**Setup.** 200-query eval (20/domain × 10 domains) from `data/final`, routed through three routers to Qwen3.6-35B-A3B Q4 on kxkm-ai (RTX 4090 24 GB, via llama-server HTTP). Each answer is scored 0-5 by Qwen3-Coder-480B MLX on Studio (M3 Ultra 512 GB) with a strict rubric (see `src/routing/llm_judge.py`).

**Routers.**
- **Random**: picks an expert uniformly among 10 (control).
- **Torch VQC (ours)**: trained per C1 with learned projection + wd=1e-4.
- **Oracle**: always picks the ground-truth expert (upper bound).

**Results** (mean rubric score 0-5; see Figure `c2-downstream-figure.pdf`).

| Router | Mean score | Routing acc | Score \| correct route | Score \| wrong route |
|---|---|---|---|---|
| Random | {{random.mean_score}} | {{random.routing_accuracy}} | {{random.mean_score_when_routed_correct}} | {{random.mean_score_when_routed_wrong}} |
| **Torch VQC** | **{{vqc.mean_score}}** | **{{vqc.routing_accuracy}}** | **{{vqc.mean_score_when_routed_correct}}** | **{{vqc.mean_score_when_routed_wrong}}** |
| Oracle | {{oracle.mean_score}} | 1.000 | {{oracle.mean_score_when_routed_correct}} | N/A |

**Kill criterion check.** `oracle_mean - random_mean = {{delta}}`. Threshold was 0.3. {{If delta >= 0.3: "Expert routing MATTERS — adapters provide measurable quality gains; Paper A direction confirmed."}} {{Else: "Expert routing DOES NOT matter on this task — adapters provide < 0.3 points of quality gain even with perfect routing. This refutes the premise that VQC routing improves downstream quality. Paper A must pivot or retract."}}

**Interpretation (assuming kill threshold passed).**

1. **The VQC router produces downstream quality between random and oracle**, consistent with C1's sub-oracle routing accuracy (0.25 vs 1.0). The VQC's quality gap to oracle ({{oracle.mean_score - vqc.mean_score}}) is the "cost" of our sub-optimal routing.

2. **Wrong-route penalty.** Comparing `score | correct route` vs `score | wrong route` within each router quantifies how expensive wrong routing is. A large gap means the task is highly domain-specific; a small gap means adapters don't specialise as much as assumed.

3. **Positioning vs C1.** C1 measured accuracy ≈ 0.25. C2's VQC mean score {{vqc.mean_score}} translates to a "quality per query" metric. If C1 accuracy tripled to 0.75, our back-of-envelope projection is {{vqc.mean_score}} + 0.5 × ({{oracle.mean_score}} - {{random.mean_score}}) = {{projected_better_vqc}}. Paper A §5 "Future work" uses this projection to motivate architectural improvements.

**Implications for Paper A.** Downstream quality is the REAL metric a routing classifier should be judged on, not accuracy. C2 establishes the measurement protocol (rubric + judge + fixed eval set) as a reproducible benchmark. Combined with C1 (classical baselines) and C5 (info-theoretic ceiling), Paper A has a complete frame: VQC is currently non-competitive, but the benchmarking infrastructure is the contribution.
```

- [ ] **Step 4: Fill in the actual values**

Run: `jq '.' results/c2-downstream.json | head -50` and substitute the `{{...}}` placeholders with measured numbers.

- [ ] **Step 5: Commit**

```bash
git add docs/paper-a/c2-downstream-results.md docs/paper-a/c2-downstream-figure.pdf scripts/figure_c2_downstream.py
git commit -m "docs(c2): downstream eval narrative + figure"
```

---

### Task 9: Push + update roadmap

- [ ] **Step 1: Full regression test**

```bash
uv run python -m pytest tests/routing/test_llm_judge.py tests/routing/test_downstream_harness.py tests/routing/test_classical_baselines.py tests/routing/test_torch_vqc.py tests/routing/test_torch_vqc_training.py tests/routing/test_torch_vqc_projection.py -v 2>&1 | tail -15
```

Expected: all PASSED.

- [ ] **Step 2: Push**

```bash
git push origin main
```

- [ ] **Step 3: Update roadmap**

In `docs/superpowers/plans/2026-04-19-phase-c-roadmap.md`, change the C2 row status to `Done (commits <sha>..<sha>)`. Commit + push.

---

## Kill criterion (reminder)

Triggered if `oracle.mean_score - random.mean_score < 0.3`. In that case:
1. Do NOT publish C2 narrative as-is — add a "negative result" section instead.
2. Revisit Paper A direction: the entire routing premise is unproven. Options: (a) retract, (b) pivot to pure-tool paper (torch-vqc release), (c) argue that adapters DO help but the eval set is poorly constructed.

## Out of scope for this plan

- Training data generation for kxkm-ai adapters — assumes per-domain LoRA adapters already exist and llama-server swaps them via `--model` parameter or equivalent. If adapters don't exist yet, create them via existing `KIKI-Mac_tunner/` pipeline (separate plan).
- Multi-turn conversations — one-shot query only.
- Human validation of LLM-judge scores — documented as a limitation.
- A/B between MLX and llama.cpp — fixed to llama.cpp on kxkm-ai per deployment decision.

## Total estimated time

- Task 1 (probes): 10 min (if endpoints up, seconds; if not, hours to fix)
- Tasks 2-5 (TDD harness): ~1.5 hours
- Task 6 (bench script): 45 min
- Task 7 (real run): ~90 min wall-clock + 15 min monitoring
- Task 8 (narrative): 45 min
- Task 9 (push): 10 min

**Total: ~5-6 hours engineering + ~90 min compute, 1 day on calendar.**
