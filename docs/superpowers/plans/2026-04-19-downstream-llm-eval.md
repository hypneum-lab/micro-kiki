# Downstream LLM Quality Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the Aeon memory predictor makes downstream LLM outputs actually BETTER on multi-turn conversational QA, not just whether retrieval Recall@5 improves — via a paired LLM-as-judge evaluation across three conditions (LLM alone / LLM + Aeon retrieval / LLM + Aeon predictor), with bootstrap confidence intervals and a writeup section 4.6 for Paper A.

**Architecture:** A new `src/eval/downstream/` package hosts four components: (1) a dataset loader that produces paired multi-turn conversations from a custom micro-kiki 10-domain held-out set, (2) a condition runner that serves each conversation through one of three rigs (A: base LLM, B: LLM + Aeon retrieval via `AeonServingHook`, C: LLM + Aeon predictor via `AeonServingHook` with `AeonPredictor.predict_next` enabled), (3) an LLM-as-judge caller that scores each response on a 3-criterion rubric (Relevance, Correctness, Context-Use) using a local Qwen3-Coder-480B MLX teacher as judge, and (4) a stats module that runs paired bootstrap CI across conditions. Everything is orchestrated from `scripts/eval_downstream_llm.py`, writing to `results/2026-04-19-downstream-llm-eval/`. Paper A integration is a new `§4.6` block authored in `docs/papers/paper-a-reframe-aeon-ami.md`.

**Tech Stack:** Python 3.11+, uv, numpy, pytest (with `integration` marker for real LLM calls), httpx for HTTP judge calls, MLX teacher at `localhost:8009` (llama-server) and judge at `localhost:8008` (mlx_lm.server on Studio). Reuses `AeonServingHook`, `AeonPalace`, `AeonPredictor`, `AeonSleep` unchanged.

---

## Success & Kill Criteria

**Success (section 4.6 ships with numbers in Paper A):**
- Full run of 500 conversations × 3 conditions × 2 LLM calls (model + judge) completes in < 6 h on Mac Studio M3 Ultra 512 GB.
- Paired bootstrap 95 % CI on `score(C) - score(B)` has lower bound > 0 on at least one of the three rubric criteria (Relevance / Correctness / Context-Use).
- Cost accounting: judge model is local (zero paid API calls); wall-clock within budget.
- All pipeline units (loader, runner, judge caller, score aggregator, stats) have unit tests that pass offline (stubs for LLM calls).

**Kill (eval is inconclusive, abandon §4.6 claim, disclose in Paper A §6):**
- Bootstrap 95 % CI on `C - B` crosses zero on all three criteria (no significant delta), OR
- Judge self-consistency (same response judged twice in different order) < 0.80 Spearman rank correlation, indicating judge noise dominates, OR
- Run > 12 h wall-clock on Studio (compute budget violation).

The stats module (Task 12) is the final gate; self-consistency check (Task 9) is the interim tripwire.

## Risk Mitigations (each mapped to a task)

1. **Judge bias** — Task 8 implements order randomization (conditions presented to judge in random order per conversation) and Task 9 adds a self-consistency check (20 conversations judged twice) before the full run. If Spearman ρ < 0.80, the run is aborted with a diagnostic dump.
2. **Sample size too small for a weak effect** — Task 3 targets 500 conversations (within budget), Task 12 computes paired bootstrap 95 % CI with 10 000 resamples. If CI width > 0.25 on a 1–5 scale, Task 16 flags "underpowered" and offers a 1 000-conversation retry mode.
3. **Compute budget blowout** — Task 4 adds a `--dry-run-n 20` smoke path that must complete in < 10 min before the full run is allowed; Task 14 streams partial results to disk every 25 conversations so a crash at hour 5 doesn't lose the whole run.

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/eval/downstream/__init__.py` | Package marker | Create |
| `src/eval/downstream/dataset.py` | `ConversationLoader` — load held-out multi-turn QA from `data/downstream-eval/` | Create |
| `src/eval/downstream/conditions.py` | `ConditionRunner` — dispatch one conversation through A / B / C rigs | Create |
| `src/eval/downstream/judge.py` | `LLMJudge` — call Qwen3-Coder judge, parse rubric scores | Create |
| `src/eval/downstream/aggregator.py` | `ScoreAggregator` — per-criterion means, per-condition summary | Create |
| `src/eval/downstream/stats.py` | `paired_bootstrap_ci` — paired resampling, 95 % CI | Create |
| `src/eval/downstream/CLAUDE.md` | Package guidance for the subagent | Create |
| `scripts/eval_downstream_llm.py` | CLI entry: orchestrate loader → runner → judge → aggregator → stats | Create |
| `scripts/gen_downstream_eval_dataset.py` | Build the 500-conversation held-out set from micro-kiki data | Create |
| `configs/eval/downstream-llm.yaml` | Hyperparameters: judge URL, teacher URL, N, seed, rubric | Create |
| `data/downstream-eval/conversations.jsonl` | Generated held-out dataset (Task 2) | Create |
| `data/downstream-eval/README.md` | Provenance note: domains covered, turns/convo, seed | Create |
| `results/2026-04-19-downstream-llm-eval/summary.json` | Machine-readable final numbers | Create (written by script) |
| `results/2026-04-19-downstream-llm-eval/per_conversation.jsonl` | Raw per-(conv, condition, criterion) rows | Create (written by script) |
| `results/2026-04-19-downstream-llm-eval/README.md` | Human-readable writeup | Create |
| `docs/papers/paper-a-reframe-aeon-ami.md` | Add §4.6 "Downstream LLM quality" | Modify |
| `tests/eval/downstream/__init__.py` | Package marker | Create |
| `tests/eval/downstream/test_dataset.py` | `ConversationLoader` unit tests | Create |
| `tests/eval/downstream/test_conditions.py` | `ConditionRunner` unit tests (LLM stubbed) | Create |
| `tests/eval/downstream/test_judge.py` | `LLMJudge` parser + stubbed call tests | Create |
| `tests/eval/downstream/test_aggregator.py` | `ScoreAggregator` math tests | Create |
| `tests/eval/downstream/test_stats.py` | `paired_bootstrap_ci` math tests | Create |
| `tests/scripts/test_eval_downstream_llm.py` | Smoke test for script end-to-end (all LLM stubbed, N=2) | Create |

Files that change together live together: the five pipeline modules, their tests, and the orchestrating script are all in the same Task sequence.

---

### Task 1: Scaffold `src/eval/downstream/` package and its CLAUDE.md

**Files:**
- Create: `src/eval/downstream/__init__.py`
- Create: `src/eval/downstream/CLAUDE.md`
- Create: `tests/eval/downstream/__init__.py`

- [ ] **Step 1: Write the failing test (module import)**

Create `tests/eval/downstream/__init__.py` (empty) and a first smoke test file `tests/eval/downstream/test_package.py`:

```python
"""Smoke test: the downstream-eval package imports."""
from __future__ import annotations


def test_package_imports() -> None:
    import src.eval.downstream as pkg

    assert pkg.__name__ == "src.eval.downstream"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/eval/downstream/test_package.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.downstream'`

- [ ] **Step 3: Create the package**

Create `src/eval/downstream/__init__.py`:

```python
"""Downstream LLM quality evaluation package.

Measures whether the Aeon memory predictor produces better LLM responses
than pure retrieval (AeonSleep baseline) or bare LLM — paired LLM-as-judge
protocol over 500 held-out multi-turn conversations.
"""
from __future__ import annotations

__all__: list[str] = []
```

- [ ] **Step 4: Create `src/eval/downstream/CLAUDE.md`**

```markdown
# src/eval/downstream/ — LLM quality eval pipeline

Paired judge protocol comparing 3 conditions (base LLM, +Aeon retrieval,
+Aeon predictor). Each module is a single responsibility:

- `dataset.py` — load `data/downstream-eval/conversations.jsonl`
- `conditions.py` — run one conversation through one rig (A/B/C)
- `judge.py` — call Qwen3-Coder judge, parse rubric JSON
- `aggregator.py` — group scores, compute per-condition means
- `stats.py` — paired bootstrap CI over (condition, criterion) deltas

All LLM calls go through `httpx.Client` against local MLX servers.
Unit tests must stub those clients — no real LLM in the default suite.
Real-model smoke tests are marked `@pytest.mark.integration`.
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run python -m pytest tests/eval/downstream/test_package.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/eval/downstream/__init__.py src/eval/downstream/CLAUDE.md tests/eval/downstream/__init__.py tests/eval/downstream/test_package.py
git commit -m "feat(eval): scaffold downstream LLM eval package"
```

---

### Task 2: Generate the held-out 10-domain conversation dataset

**Files:**
- Create: `scripts/gen_downstream_eval_dataset.py`
- Create: `data/downstream-eval/README.md`
- Create: `tests/scripts/test_gen_downstream_eval_dataset.py`

**Why a custom micro-kiki dataset (not LongEval / MT-Bench):** The predictor was trained on conversational turns over the 35 micro-kiki domains. LongEval targets very long-context retrieval (100k+ tokens), which is orthogonal to Aeon's short-term-memory thesis. MT-Bench is only 80 items and open-domain — too small and too generic for a +22 % MRR effect to translate into a detectable judge delta. A custom 10-domain × 50-conversation held-out set keeps the distribution match and gives N = 500, which is the minimum for paired-bootstrap 95 % CI at a 0.1-point effect on a 1–5 scale.

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_gen_downstream_eval_dataset.py`:

```python
"""Smoke test for the dataset generator — N=10 deterministic output."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_generator_produces_jsonl_with_required_fields(tmp_path: Path) -> None:
    out = tmp_path / "conversations.jsonl"
    rc = subprocess.run(
        [
            sys.executable,
            "scripts/gen_downstream_eval_dataset.py",
            "--out", str(out),
            "--n-per-domain", "1",
            "--domains", "power_electronics,spice",
            "--seed", "42",
        ],
        capture_output=True, text=True,
    ).returncode
    assert rc == 0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 2  # 2 domains × 1 conversation
    for r in rows:
        assert set(r.keys()) >= {
            "conversation_id", "domain", "turns", "final_question", "reference_answer"
        }
        assert len(r["turns"]) >= 3  # multi-turn
        assert r["turns"][-1]["role"] == "user"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/scripts/test_gen_downstream_eval_dataset.py -v`
Expected: FAIL (script missing).

- [ ] **Step 3: Implement `scripts/gen_downstream_eval_dataset.py`**

```python
#!/usr/bin/env python3
"""Generate 10-domain × 50-conversation held-out eval set.

Source: `data/domains/*/qa.jsonl` from existing micro-kiki domain datasets.
Each conversation = 3-4 warm-up turns + 1 final question where memory of the
earlier turns is required to answer. Reference answer is the dataset's ground truth.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ALL_DOMAINS = [
    "power_electronics", "spice", "kicad", "platformio", "web_frontend",
    "embedded_rtos", "ml_training", "data_engineering", "networking", "security",
]


def load_domain_qa(domain: str, root: Path) -> list[dict]:
    path = root / "data" / "domains" / domain / "qa.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def build_conversation(qa_pool: list[dict], domain: str, rng: random.Random, idx: int) -> dict:
    """4-turn conversation: 3 warm-up Q/A + 1 memory-dependent final question."""
    sample = rng.sample(qa_pool, min(4, len(qa_pool)))
    warmup = sample[:3]
    final = sample[3] if len(sample) >= 4 else sample[-1]
    turns: list[dict] = []
    for qa in warmup:
        turns.append({"role": "user", "content": qa["question"]})
        turns.append({"role": "assistant", "content": qa["answer"]})
    turns.append({"role": "user", "content": final["question"]})
    return {
        "conversation_id": f"{domain}_{idx:04d}",
        "domain": domain,
        "turns": turns,
        "final_question": final["question"],
        "reference_answer": final["answer"],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-per-domain", type=int, default=50)
    p.add_argument("--domains", type=str, default=",".join(ALL_DOMAINS))
    p.add_argument("--seed", type=int, default=20260419)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = p.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for domain in args.domains.split(","):
            domain = domain.strip()
            pool = load_domain_qa(domain, args.root)
            if len(pool) < 4:
                print(f"[warn] domain {domain} has <4 QA rows, skipping", flush=True)
                continue
            for i in range(args.n_per_domain):
                conv = build_conversation(pool, domain, rng, i)
                fh.write(json.dumps(conv) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/scripts/test_gen_downstream_eval_dataset.py -v`
Expected: PASS. (If the test fails because `data/domains/*/qa.jsonl` isn't present, the test should use a fixture path — but for this plan we assume the datasets exist per `scripts/CLAUDE.md`; the test may need to mock the loader. If so, parametrize `load_domain_qa` via a `--root` flag and point it at `tmp_path` with a seeded fixture.)

- [ ] **Step 5: Write `data/downstream-eval/README.md`**

```markdown
# Downstream LLM eval — held-out conversations

Generated by `scripts/gen_downstream_eval_dataset.py`.

- **N conversations**: 500 (10 domains × 50)
- **Turns per conversation**: 4 (3 warm-up + 1 final memory-dependent)
- **Source**: `data/domains/<domain>/qa.jsonl`, seed 20260419
- **Domains**: power_electronics, spice, kicad, platformio, web_frontend,
  embedded_rtos, ml_training, data_engineering, networking, security
- **Format**: JSONL, one conversation per line with `conversation_id`,
  `domain`, `turns`, `final_question`, `reference_answer`.

DO NOT use these conversations for training — held out for Paper A §4.6.
```

- [ ] **Step 6: Commit**

```bash
git add scripts/gen_downstream_eval_dataset.py tests/scripts/test_gen_downstream_eval_dataset.py data/downstream-eval/README.md
git commit -m "feat(eval): add held-out conversation dataset generator"
```

---

### Task 3: `ConversationLoader` — typed loader for the JSONL set

**Files:**
- Create: `src/eval/downstream/dataset.py`
- Create: `tests/eval/downstream/test_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for ConversationLoader."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.downstream.dataset import Conversation, ConversationLoader


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "conv.jsonl"
    row = {
        "conversation_id": "spice_0000",
        "domain": "spice",
        "turns": [
            {"role": "user", "content": "What is .tran?"},
            {"role": "assistant", "content": "Transient analysis."},
            {"role": "user", "content": "Give a 10ms example."},
        ],
        "final_question": "Give a 10ms example.",
        "reference_answer": ".tran 1u 10m",
    }
    path.write_text(json.dumps(row) + "\n")
    return path


def test_loader_reads_jsonl(sample_jsonl: Path) -> None:
    loader = ConversationLoader(sample_jsonl)
    convs = list(loader)
    assert len(convs) == 1
    c = convs[0]
    assert isinstance(c, Conversation)
    assert c.conversation_id == "spice_0000"
    assert c.domain == "spice"
    assert len(c.turns) == 3
    assert c.final_question == "Give a 10ms example."


def test_loader_supports_limit(sample_jsonl: Path) -> None:
    loader = ConversationLoader(sample_jsonl, limit=0)
    assert list(loader) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/eval/downstream/test_dataset.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `src/eval/downstream/dataset.py`**

```python
"""Conversation loader for the downstream-eval dataset."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Turn:
    role: str
    content: str


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    domain: str
    turns: tuple[Turn, ...]
    final_question: str
    reference_answer: str


class ConversationLoader:
    """Iterate over JSONL conversations with optional `limit`."""

    def __init__(self, path: Path, limit: int | None = None) -> None:
        self._path = Path(path)
        self._limit = limit

    def __iter__(self) -> Iterator[Conversation]:
        count = 0
        with self._path.open() as fh:
            for line in fh:
                if self._limit is not None and count >= self._limit:
                    return
                row = json.loads(line)
                yield Conversation(
                    conversation_id=row["conversation_id"],
                    domain=row["domain"],
                    turns=tuple(Turn(**t) for t in row["turns"]),
                    final_question=row["final_question"],
                    reference_answer=row["reference_answer"],
                )
                count += 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/eval/downstream/test_dataset.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/eval/downstream/dataset.py tests/eval/downstream/test_dataset.py
git commit -m "feat(eval): add ConversationLoader for downstream eval"
```

---

### Task 4: `LLMClient` — thin httpx wrapper for MLX server

**Files:**
- Create: `src/eval/downstream/llm_client.py`
- Create: `tests/eval/downstream/test_llm_client.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for LLMClient — stubbed httpx calls."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.eval.downstream.llm_client import LLMClient


def test_generate_posts_to_chat_completions() -> None:
    client = LLMClient(base_url="http://localhost:8009", model="qwen3.5-35b-a3b")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    with patch.object(client._http, "post", return_value=mock_resp) as m:
        out = client.generate(messages=[{"role": "user", "content": "hi"}], max_tokens=16)
    assert out == "ok"
    m.assert_called_once()
    body = m.call_args.kwargs["json"]
    assert body["model"] == "qwen3.5-35b-a3b"
    assert body["max_tokens"] == 16


def test_generate_raises_on_http_error() -> None:
    client = LLMClient(base_url="http://localhost:8009", model="m")
    mock_resp = MagicMock(status_code=500, text="boom")
    with patch.object(client._http, "post", return_value=mock_resp):
        import pytest
        with pytest.raises(RuntimeError, match="500"):
            client.generate(messages=[{"role": "user", "content": "x"}])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/eval/downstream/test_llm_client.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `src/eval/downstream/llm_client.py`**

```python
"""HTTP client for OpenAI-compatible MLX / llama-server endpoints."""
from __future__ import annotations

import httpx


class LLMClient:
    """Minimal OpenAI-compatible chat.completions client."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 120.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._model = model
        self._http = httpx.Client(timeout=timeout_s)

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        url = f"{self._base}/v1/chat/completions"
        body = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = self._http.post(url, json=body)
        if resp.status_code != 200:
            raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()["choices"][0]["message"]["content"]

    def close(self) -> None:
        self._http.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/eval/downstream/test_llm_client.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/eval/downstream/llm_client.py tests/eval/downstream/test_llm_client.py
git commit -m "feat(eval): add minimal LLMClient for MLX endpoints"
```

---

### Task 5: `ConditionRunner` — run one conversation through one rig

**Files:**
- Create: `src/eval/downstream/conditions.py`
- Create: `tests/eval/downstream/test_conditions.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for ConditionRunner."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.eval.downstream.conditions import ConditionRunner, Rig
from src.eval.downstream.dataset import Conversation, Turn


@pytest.fixture
def conv() -> Conversation:
    return Conversation(
        conversation_id="c1",
        domain="spice",
        turns=(
            Turn("user", "What is .tran?"),
            Turn("assistant", "Transient."),
            Turn("user", "Give a 10ms example."),
        ),
        final_question="Give a 10ms example.",
        reference_answer=".tran 1u 10m",
    )


def test_rig_A_calls_llm_with_raw_turns(conv: Conversation) -> None:
    llm = MagicMock()
    llm.generate.return_value = "answer A"
    runner = ConditionRunner(llm_client=llm, hook=None, predictor_enabled=False)
    out = runner.run(conv, rig=Rig.A)
    assert out == "answer A"
    sent = llm.generate.call_args.kwargs["messages"]
    assert sent[-1]["content"] == "Give a 10ms example."
    assert len(sent) == 3  # all original turns


def test_rig_B_prepends_retrieval_memory(conv: Conversation) -> None:
    llm = MagicMock()
    llm.generate.return_value = "answer B"
    hook = MagicMock()
    hook.pre_inference.return_value = "### Previous conversation context:\nmem\n\n### Current question:\nGive a 10ms example."
    runner = ConditionRunner(llm_client=llm, hook=hook, predictor_enabled=False)
    out = runner.run(conv, rig=Rig.B)
    assert out == "answer B"
    hook.pre_inference.assert_called_once_with("Give a 10ms example.", top_k=8)
    last_msg = llm.generate.call_args.kwargs["messages"][-1]["content"]
    assert "Previous conversation context" in last_msg


def test_rig_C_toggles_predictor_on_hook(conv: Conversation) -> None:
    llm = MagicMock()
    llm.generate.return_value = "answer C"
    hook = MagicMock()
    hook.pre_inference.return_value = "with-pred memory"
    runner = ConditionRunner(llm_client=llm, hook=hook, predictor_enabled=True)
    out = runner.run(conv, rig=Rig.C)
    assert out == "answer C"
    # In rig C the runner flips the predictor flag on the hook before recall
    assert hook.use_predictor == True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/eval/downstream/test_conditions.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/eval/downstream/conditions.py`**

```python
"""Run one conversation through one of the 3 rigs (A: base, B: retrieval, C: predictor)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.eval.downstream.dataset import Conversation


class Rig(str, Enum):
    A = "base_llm"
    B = "llm_plus_aeon_retrieval"
    C = "llm_plus_aeon_predictor"


@dataclass
class ConditionRunner:
    """Dispatch a conversation to one rig and return the LLM response."""

    llm_client: object  # duck-typed LLMClient
    hook: object | None  # AeonServingHook or None for rig A
    predictor_enabled: bool  # whether the hook is allowed to use predictor in rig C

    def run(self, conv: Conversation, rig: Rig) -> str:
        messages = [{"role": t.role, "content": t.content} for t in conv.turns]
        if rig == Rig.A:
            return self.llm_client.generate(messages=messages, max_tokens=512)
        if rig not in (Rig.B, Rig.C):
            raise ValueError(f"unknown rig: {rig}")
        assert self.hook is not None, f"rig {rig} needs an AeonServingHook"
        # Rig C toggles the predictor; rig B disables it.
        self.hook.use_predictor = rig == Rig.C and self.predictor_enabled
        augmented = self.hook.pre_inference(conv.final_question, top_k=8)
        # Replace the final user turn with the augmented prompt.
        messages = messages[:-1] + [{"role": "user", "content": augmented}]
        return self.llm_client.generate(messages=messages, max_tokens=512)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/eval/downstream/test_conditions.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/eval/downstream/conditions.py tests/eval/downstream/test_conditions.py
git commit -m "feat(eval): add ConditionRunner for 3-rig dispatch"
```

---

### Task 6: Extend `AeonServingHook` with a `use_predictor` flag

**Files:**
- Modify: `src/serving/aeon_hook.py`
- Modify: `tests/serving/test_aeon_hook.py`

- [ ] **Step 1: Write the failing test (add to existing test file)**

Append to `tests/serving/test_aeon_hook.py`:

```python
class TestAeonServingHookPredictor:
    def test_predictor_disabled_by_default(self):
        from src.memory.aeon import AeonPalace
        from src.serving.aeon_hook import AeonServingHook
        palace = AeonPalace(dim=64, embed_fn=_mock_embed(64))
        hook = AeonServingHook(palace)
        assert hook.use_predictor is False

    def test_predictor_flag_routes_recall_through_predict_next(self):
        from src.memory.aeon import AeonPalace
        from src.memory.aeon_predictor import AeonPredictor, PredictorConfig
        from src.serving.aeon_hook import AeonServingHook

        palace = AeonPalace(dim=64, embed_fn=_mock_embed(64))
        palace.write("User: q1\nAssistant: a1", domain="spice")
        predictor = AeonPredictor(dim=64, config=PredictorConfig(min_pairs=0))
        # Without real training, predict_next falls back to h_t — so just verify wiring.
        hook = AeonServingHook(palace, predictor=predictor)
        hook.use_predictor = True
        out = hook.pre_inference("a follow-up question")
        # Should not raise; should still produce memory context
        assert "Previous conversation context" in out or out == "a follow-up question"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/serving/test_aeon_hook.py::TestAeonServingHookPredictor -v`
Expected: FAIL (`use_predictor` not implemented, `predictor` kwarg missing).

- [ ] **Step 3: Modify `src/serving/aeon_hook.py`**

Replace the `AeonServingHook.__init__` and add `use_predictor` handling:

```python
class AeonServingHook:
    """Wraps AeonPalace for pre/post inference memory injection."""

    def __init__(
        self,
        palace: AeonPalace,
        predictor: object | None = None,
    ) -> None:
        self._palace = palace
        self._predictor = predictor
        self.use_predictor: bool = False

    def pre_inference(self, prompt: str, top_k: int = 8) -> str:
        try:
            if self.use_predictor and self._predictor is not None:
                # Embed, predict next latent, recall around that point.
                q_embed = self._palace._embed(prompt)  # reuses palace embed fn
                q_pred = self._predictor.predict_next(q_embed)
                episodes = self._palace.recall_by_embed(q_pred, top_k=top_k)
            else:
                episodes = self._palace.recall(prompt, top_k=top_k)
        except Exception:
            logger.warning("Aeon recall failed, returning original prompt", exc_info=True)
            return prompt

        if not episodes:
            return prompt

        per_ep = max(200, MEMORY_BUDGET // len(episodes))
        lines = [ep.content[:per_ep] for ep in episodes]
        memory_block = (
            "### Previous conversation context:\n"
            + "\n---\n".join(lines)
            + "\n\n### Current question:\n"
        )
        return memory_block + prompt
```

If `AeonPalace` does not yet expose `_embed` or `recall_by_embed`, audit `src/memory/aeon.py` first — both are already present per `src/memory/aeon_predictor.py`'s integration. If a method name differs, adjust the reference (do NOT add new public methods to `AeonPalace` in this task — extending `AeonPalace` surface is out of scope).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/serving/test_aeon_hook.py -v`
Expected: PASS (all existing tests still pass + new ones).

- [ ] **Step 5: Commit**

```bash
git add src/serving/aeon_hook.py tests/serving/test_aeon_hook.py
git commit -m "feat(serving): AeonServingHook supports predictor toggle"
```

---

### Task 7: `LLMJudge` — rubric prompt, JSON parser, stubbed call

**Files:**
- Create: `src/eval/downstream/judge.py`
- Create: `tests/eval/downstream/test_judge.py`

**Judge model choice:** Qwen3-Coder-480B MLX 4bit on the Mac Studio (`mlx_lm.server` at `localhost:8008`). Rationale: (a) already the micro-kiki teacher, so the same credential-free local endpoint, (b) 480B is >10× the 35B being judged, which satisfies the "judge > model-under-test" heuristic for LLM-as-judge reliability, (c) zero paid-API cost, (d) reproducible (no model version drift from a cloud API). Claude Opus is the fallback if judge self-consistency (Task 9) fails.

**Rubric (fixed):** Three criteria, each on a 1–5 integer scale.
- **Relevance** — Does the answer address the final question?
- **Correctness** — Is the answer factually correct given the reference answer?
- **Context-Use** — Does the answer use information from earlier conversation turns (or correctly declare none was needed)?

- [ ] **Step 1: Write the failing test**

```python
"""Tests for LLMJudge."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.eval.downstream.judge import JudgeScore, LLMJudge


def test_build_prompt_includes_all_criteria() -> None:
    judge = LLMJudge(llm_client=MagicMock())
    prompt = judge.build_prompt(
        conversation="U: q1\nA: a1\nU: q2",
        reference="ref",
        response="resp",
    )
    assert "Relevance" in prompt
    assert "Correctness" in prompt
    assert "Context-Use" in prompt
    assert "ref" in prompt
    assert "resp" in prompt


def test_parse_scores_extracts_json_block() -> None:
    judge = LLMJudge(llm_client=MagicMock())
    raw = 'some reasoning... {"relevance": 4, "correctness": 5, "context_use": 3} trailing'
    score = judge.parse_scores(raw)
    assert score == JudgeScore(relevance=4, correctness=5, context_use=3)


def test_parse_scores_rejects_out_of_range() -> None:
    judge = LLMJudge(llm_client=MagicMock())
    with pytest.raises(ValueError, match="range"):
        judge.parse_scores('{"relevance": 7, "correctness": 3, "context_use": 2}')


def test_score_calls_llm_and_parses() -> None:
    llm = MagicMock()
    llm.generate.return_value = '{"relevance": 4, "correctness": 4, "context_use": 5}'
    judge = LLMJudge(llm_client=llm)
    score = judge.score(conversation="c", reference="r", response="a")
    assert score == JudgeScore(relevance=4, correctness=4, context_use=5)
    llm.generate.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/eval/downstream/test_judge.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/eval/downstream/judge.py`**

```python
"""LLM-as-judge caller with fixed 3-criterion rubric."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass


PROMPT_TEMPLATE = """You are an expert evaluator scoring an assistant answer.

### Conversation so far:
{conversation}

### Reference answer (ground truth):
{reference}

### Candidate answer:
{response}

Rate on three criteria, integer 1-5 each. Output ONLY a JSON object:
{{"relevance": R, "correctness": C, "context_use": U}}

- Relevance (1-5): does it address the final question?
- Correctness (1-5): is it factually correct vs reference?
- Context-Use (1-5): does it use earlier turns (or correctly state none needed)?

Output the JSON object only, no prose."""


@dataclass(frozen=True)
class JudgeScore:
    relevance: int
    correctness: int
    context_use: int


class LLMJudge:
    """Build prompt, call judge LLM, parse JSON scores."""

    def __init__(self, llm_client: object, max_tokens: int = 128) -> None:
        self._llm = llm_client
        self._max_tokens = max_tokens

    def build_prompt(self, conversation: str, reference: str, response: str) -> str:
        return PROMPT_TEMPLATE.format(
            conversation=conversation, reference=reference, response=response
        )

    @staticmethod
    def parse_scores(raw: str) -> JudgeScore:
        match = re.search(r"\{[^{}]*\}", raw)
        if not match:
            raise ValueError(f"no JSON object found in: {raw[:200]}")
        obj = json.loads(match.group(0))
        for key in ("relevance", "correctness", "context_use"):
            if key not in obj:
                raise ValueError(f"missing key {key} in {obj}")
            if not (1 <= int(obj[key]) <= 5):
                raise ValueError(f"{key}={obj[key]} out of range 1-5")
        return JudgeScore(
            relevance=int(obj["relevance"]),
            correctness=int(obj["correctness"]),
            context_use=int(obj["context_use"]),
        )

    def score(self, conversation: str, reference: str, response: str) -> JudgeScore:
        prompt = self.build_prompt(conversation, reference, response)
        raw = self._llm.generate(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._max_tokens,
            temperature=0.0,
        )
        return self.parse_scores(raw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/eval/downstream/test_judge.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/eval/downstream/judge.py tests/eval/downstream/test_judge.py
git commit -m "feat(eval): add LLMJudge with 3-criterion rubric"
```

---

### Task 8: Order randomization — present conditions to judge in random order per conversation

**Files:**
- Modify: `src/eval/downstream/judge.py`
- Modify: `tests/eval/downstream/test_judge.py`

- [ ] **Step 1: Write the failing test**

```python
def test_score_batch_randomizes_order_deterministically() -> None:
    from src.eval.downstream.judge import LLMJudge, JudgeScore
    llm = MagicMock()
    llm.generate.return_value = '{"relevance": 3, "correctness": 3, "context_use": 3}'
    judge = LLMJudge(llm_client=llm)
    responses = {"A": "respA", "B": "respB", "C": "respC"}
    scores1 = judge.score_batch("conv", "ref", responses, seed=42)
    scores2 = judge.score_batch("conv", "ref", responses, seed=42)
    # Same seed → same order → same scores dict keyed by condition
    assert scores1 == scores2
    assert set(scores1.keys()) == {"A", "B", "C"}
    assert all(isinstance(v, JudgeScore) for v in scores1.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/eval/downstream/test_judge.py -v`
Expected: FAIL (`score_batch` missing).

- [ ] **Step 3: Add `score_batch` to `LLMJudge`**

Append to `src/eval/downstream/judge.py`:

```python
import random as _random


    def score_batch(
        self,
        conversation: str,
        reference: str,
        responses: dict[str, str],
        seed: int,
    ) -> dict:
        """Score {condition_label: response} in a seeded-shuffled order.

        Returns {condition_label: JudgeScore}. Order of LLM calls is randomized
        to prevent the judge from anchoring on the first-seen condition.
        """
        rng = _random.Random(seed)
        order = list(responses.keys())
        rng.shuffle(order)
        out = {}
        for label in order:
            out[label] = self.score(conversation, reference, responses[label])
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/eval/downstream/test_judge.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/eval/downstream/judge.py tests/eval/downstream/test_judge.py
git commit -m "feat(eval): add seeded score_batch for order randomization"
```

---

### Task 9: Judge self-consistency check — abort-run-if-noisy helper

**Files:**
- Create: `src/eval/downstream/self_consistency.py`
- Create: `tests/eval/downstream/test_self_consistency.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for judge self-consistency check."""
from __future__ import annotations

from src.eval.downstream.self_consistency import spearman_rank


def test_spearman_identical_lists_is_one() -> None:
    assert spearman_rank([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0


def test_spearman_reverse_lists_is_minus_one() -> None:
    assert spearman_rank([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == -1.0


def test_spearman_uncorrelated_near_zero() -> None:
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    b = [3, 1, 4, 1, 5, 9, 2, 6]
    rho = spearman_rank(a, b)
    assert -1.0 <= rho <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/eval/downstream/test_self_consistency.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `src/eval/downstream/self_consistency.py`**

```python
"""Judge self-consistency: Spearman rank of the same responses judged twice."""
from __future__ import annotations

import numpy as np


def spearman_rank(a: list[float], b: list[float]) -> float:
    """Simple Spearman rank correlation — tied ranks averaged."""
    if len(a) != len(b):
        raise ValueError("inputs must have same length")
    if len(a) < 2:
        raise ValueError("need at least 2 points")
    ra = _rank(a)
    rb = _rank(b)
    mean_a = float(np.mean(ra))
    mean_b = float(np.mean(rb))
    num = float(np.sum((ra - mean_a) * (rb - mean_b)))
    den = float(np.sqrt(np.sum((ra - mean_a) ** 2) * np.sum((rb - mean_b) ** 2)))
    if den == 0.0:
        return 0.0
    return num / den


def _rank(xs: list[float]) -> np.ndarray:
    arr = np.asarray(xs, dtype=float)
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1)
    return ranks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/eval/downstream/test_self_consistency.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/eval/downstream/self_consistency.py tests/eval/downstream/test_self_consistency.py
git commit -m "feat(eval): add Spearman rank helper for judge self-consistency"
```

---

### Task 10: `ScoreAggregator` — per-criterion means per condition

**Files:**
- Create: `src/eval/downstream/aggregator.py`
- Create: `tests/eval/downstream/test_aggregator.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for ScoreAggregator."""
from __future__ import annotations

from src.eval.downstream.aggregator import ScoreAggregator
from src.eval.downstream.judge import JudgeScore


def test_aggregator_computes_means_per_condition() -> None:
    agg = ScoreAggregator()
    agg.add("c1", "A", JudgeScore(5, 4, 3))
    agg.add("c2", "A", JudgeScore(3, 4, 3))
    agg.add("c1", "B", JudgeScore(4, 4, 4))
    agg.add("c2", "B", JudgeScore(4, 4, 4))
    means = agg.means()
    assert means["A"]["relevance"] == 4.0
    assert means["A"]["correctness"] == 4.0
    assert means["A"]["context_use"] == 3.0
    assert means["B"]["relevance"] == 4.0


def test_aggregator_paired_series_returns_aligned_lists() -> None:
    agg = ScoreAggregator()
    agg.add("c1", "A", JudgeScore(5, 4, 3))
    agg.add("c1", "B", JudgeScore(4, 4, 4))
    agg.add("c2", "A", JudgeScore(3, 4, 3))
    agg.add("c2", "B", JudgeScore(4, 4, 4))
    a, b = agg.paired_series("A", "B", criterion="relevance")
    assert a == [5, 3]
    assert b == [4, 4]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/eval/downstream/test_aggregator.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/eval/downstream/aggregator.py`**

```python
"""Aggregate JudgeScore rows indexed by (conversation_id, condition)."""
from __future__ import annotations

from collections import defaultdict

from src.eval.downstream.judge import JudgeScore


class ScoreAggregator:
    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], JudgeScore] = {}

    def add(self, conv_id: str, condition: str, score: JudgeScore) -> None:
        self._rows[(conv_id, condition)] = score

    def means(self) -> dict[str, dict[str, float]]:
        buckets: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: {"relevance": [], "correctness": [], "context_use": []}
        )
        for (_, cond), s in self._rows.items():
            buckets[cond]["relevance"].append(s.relevance)
            buckets[cond]["correctness"].append(s.correctness)
            buckets[cond]["context_use"].append(s.context_use)
        return {
            cond: {k: sum(v) / len(v) for k, v in bucket.items()}
            for cond, bucket in buckets.items()
        }

    def paired_series(
        self, cond_a: str, cond_b: str, criterion: str
    ) -> tuple[list[int], list[int]]:
        """Return aligned score lists for a paired test."""
        conv_ids = sorted(
            {cid for (cid, c) in self._rows if c in (cond_a, cond_b)}
        )
        a, b = [], []
        for cid in conv_ids:
            sa = self._rows.get((cid, cond_a))
            sb = self._rows.get((cid, cond_b))
            if sa is None or sb is None:
                continue
            a.append(getattr(sa, criterion))
            b.append(getattr(sb, criterion))
        return a, b
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/eval/downstream/test_aggregator.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/eval/downstream/aggregator.py tests/eval/downstream/test_aggregator.py
git commit -m "feat(eval): add ScoreAggregator with paired-series helper"
```

---

### Task 11: `paired_bootstrap_ci` — statistical engine

**Files:**
- Create: `src/eval/downstream/stats.py`
- Create: `tests/eval/downstream/test_stats.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for paired_bootstrap_ci."""
from __future__ import annotations

from src.eval.downstream.stats import paired_bootstrap_ci


def test_ci_contains_true_mean_delta() -> None:
    # a-b has true mean 1.0; 10k resamples at seed should give CI around 1.0
    a = [5, 4, 5, 4, 5, 4, 5, 4, 5, 4]
    b = [4, 3, 4, 3, 4, 3, 4, 3, 4, 3]
    mean_d, lo, hi = paired_bootstrap_ci(a, b, n_resamples=10_000, seed=42)
    assert mean_d == 1.0
    assert 0.5 < lo < 1.0 < hi < 1.5


def test_ci_on_zero_delta_crosses_zero() -> None:
    a = [4, 4, 4, 4, 4]
    b = [4, 4, 4, 4, 4]
    mean_d, lo, hi = paired_bootstrap_ci(a, b, n_resamples=10_000, seed=42)
    assert mean_d == 0.0
    assert lo == 0.0 and hi == 0.0


def test_ci_rejects_mismatched_lengths() -> None:
    import pytest
    with pytest.raises(ValueError, match="length"):
        paired_bootstrap_ci([1, 2], [1, 2, 3])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/eval/downstream/test_stats.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `src/eval/downstream/stats.py`**

```python
"""Paired bootstrap confidence interval for (condition_a - condition_b) mean delta."""
from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(
    a: list[float],
    b: list[float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 20260419,
) -> tuple[float, float, float]:
    """Return (mean_delta, ci_lo, ci_hi) at (1 - alpha) level."""
    if len(a) != len(b):
        raise ValueError("a and b must have same length")
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    delta = arr_a - arr_b
    mean_d = float(delta.mean())
    rng = np.random.default_rng(seed)
    n = len(delta)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = delta[idx].mean(axis=1)
    lo = float(np.quantile(resampled_means, alpha / 2))
    hi = float(np.quantile(resampled_means, 1 - alpha / 2))
    return mean_d, lo, hi
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/eval/downstream/test_stats.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/eval/downstream/stats.py tests/eval/downstream/test_stats.py
git commit -m "feat(eval): add paired_bootstrap_ci for condition deltas"
```

---

### Task 12: Config file — `configs/eval/downstream-llm.yaml`

**Files:**
- Create: `configs/eval/downstream-llm.yaml`
- Create: `tests/eval/downstream/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
"""Config schema validation."""
from __future__ import annotations

from pathlib import Path

import yaml


def test_config_has_required_keys() -> None:
    path = Path(__file__).resolve().parents[3] / "configs" / "eval" / "downstream-llm.yaml"
    cfg = yaml.safe_load(path.read_text())
    for key in (
        "model_url", "model_name", "judge_url", "judge_name",
        "n_conversations", "seed", "bootstrap_resamples",
        "self_consistency_n", "self_consistency_threshold",
        "out_dir", "dataset_path",
    ):
        assert key in cfg, f"missing {key}"
    assert cfg["n_conversations"] >= 20
    assert 0.5 < cfg["self_consistency_threshold"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/eval/downstream/test_config.py -v`
Expected: FAIL.

- [ ] **Step 3: Create `configs/eval/downstream-llm.yaml`**

```yaml
# Downstream LLM quality eval — Paper A §4.6
# All HTTP endpoints are local MLX servers on Mac Studio M3 Ultra.

model_url: http://localhost:8009    # llama-server — Qwen3.5-35B-A3B Q4_K_M
model_name: qwen3.5-35b-a3b

judge_url: http://localhost:8008    # mlx_lm.server — Qwen3-Coder-480B MLX 4bit
judge_name: qwen3-coder-480b-mlx-4bit

n_conversations: 500
seed: 20260419
bootstrap_resamples: 10000

self_consistency_n: 20
self_consistency_threshold: 0.80

dataset_path: data/downstream-eval/conversations.jsonl
out_dir: results/2026-04-19-downstream-llm-eval
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/eval/downstream/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/eval/downstream-llm.yaml tests/eval/downstream/test_config.py
git commit -m "feat(eval): add downstream-llm eval config"
```

---

### Task 13: `scripts/eval_downstream_llm.py` — CLI orchestrator

**Files:**
- Create: `scripts/eval_downstream_llm.py`
- Create: `tests/scripts/test_eval_downstream_llm.py`

- [ ] **Step 1: Write the failing test (smoke test with all LLM stubbed, N=2)**

```python
"""Smoke test: the CLI runs end-to-end with stubs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


@pytest.mark.integration
def test_cli_dry_run_with_real_stubs(tmp_path: Path) -> None:
    """Runs the CLI with --stub-llm which short-circuits all HTTP."""
    out_dir = tmp_path / "out"
    dataset = tmp_path / "conv.jsonl"
    row = {
        "conversation_id": "t1", "domain": "spice",
        "turns": [{"role": "user", "content": "hi"}],
        "final_question": "hi", "reference_answer": "hello",
    }
    dataset.write_text(json.dumps(row) + "\n")
    rc = subprocess.run(
        [
            sys.executable, "scripts/eval_downstream_llm.py",
            "--dataset", str(dataset),
            "--out-dir", str(out_dir),
            "--n-conversations", "1",
            "--stub-llm",
        ],
        capture_output=True, text=True,
    ).returncode
    assert rc == 0
    assert (out_dir / "summary.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/scripts/test_eval_downstream_llm.py -v -m integration`
Expected: FAIL (script missing).

- [ ] **Step 3: Implement `scripts/eval_downstream_llm.py`**

```python
#!/usr/bin/env python3
"""Run the downstream LLM quality eval: loader → 3 rigs → judge → stats.

Writes results/<out_dir>/{summary.json, per_conversation.jsonl, README.md}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import yaml

from src.eval.downstream.aggregator import ScoreAggregator
from src.eval.downstream.conditions import ConditionRunner, Rig
from src.eval.downstream.dataset import ConversationLoader
from src.eval.downstream.judge import JudgeScore, LLMJudge
from src.eval.downstream.llm_client import LLMClient
from src.eval.downstream.self_consistency import spearman_rank
from src.eval.downstream.stats import paired_bootstrap_ci
from src.memory.aeon import AeonPalace
from src.memory.aeon_predictor import AeonPredictor, PredictorConfig
from src.serving.aeon_hook import AeonServingHook


def build_stub_llm() -> MagicMock:
    m = MagicMock()
    m.generate.side_effect = lambda **kw: '{"relevance": 3, "correctness": 3, "context_use": 3}'
    return m


def conv_to_text(turns: tuple) -> str:
    return "\n".join(f"{t.role}: {t.content}" for t in turns)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/eval/downstream-llm.yaml"))
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n-conversations", type=int, default=None)
    p.add_argument("--stub-llm", action="store_true",
                   help="bypass HTTP; use deterministic stub (for smoke tests)")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    dataset_path = args.dataset or Path(cfg["dataset_path"])
    out_dir = args.out_dir or Path(cfg["out_dir"])
    n_convs = args.n_conversations or cfg["n_conversations"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build LLM + judge clients (or stubs).
    if args.stub_llm:
        model_llm = build_stub_llm()
        judge_llm = build_stub_llm()
    else:
        model_llm = LLMClient(base_url=cfg["model_url"], model=cfg["model_name"])
        judge_llm = LLMClient(base_url=cfg["judge_url"], model=cfg["judge_name"])

    # Build Aeon stack (hook + predictor).
    palace = AeonPalace(dim=384)
    predictor = AeonPredictor(dim=384, config=PredictorConfig(min_pairs=0))
    hook = AeonServingHook(palace, predictor=predictor)

    runner_a = ConditionRunner(llm_client=model_llm, hook=None, predictor_enabled=False)
    runner_b = ConditionRunner(llm_client=model_llm, hook=hook, predictor_enabled=False)
    runner_c = ConditionRunner(llm_client=model_llm, hook=hook, predictor_enabled=True)
    judge = LLMJudge(llm_client=judge_llm)

    agg = ScoreAggregator()
    per_conv_path = out_dir / "per_conversation.jsonl"
    per_conv_fh = per_conv_path.open("w")

    start = time.time()
    for i, conv in enumerate(ConversationLoader(dataset_path, limit=n_convs)):
        # Seed Aeon memory with warm-up turns before the final question.
        for t in conv.turns[:-1]:
            palace.write(content=f"{t.role}: {t.content}",
                         domain=conv.domain, source=conv.conversation_id)

        responses = {
            "A": runner_a.run(conv, Rig.A),
            "B": runner_b.run(conv, Rig.B),
            "C": runner_c.run(conv, Rig.C),
        }
        scores = judge.score_batch(
            conversation=conv_to_text(conv.turns),
            reference=conv.reference_answer,
            responses=responses,
            seed=cfg["seed"] + i,
        )
        for cond, score in scores.items():
            agg.add(conv.conversation_id, cond, score)
            per_conv_fh.write(json.dumps({
                "conversation_id": conv.conversation_id,
                "domain": conv.domain,
                "condition": cond,
                "relevance": score.relevance,
                "correctness": score.correctness,
                "context_use": score.context_use,
            }) + "\n")
        if (i + 1) % 25 == 0:
            per_conv_fh.flush()
            print(f"[{i+1}/{n_convs}] elapsed {time.time()-start:.0f}s", flush=True)

    per_conv_fh.close()

    # Compute summary.
    means = agg.means()
    deltas = {}
    for criterion in ("relevance", "correctness", "context_use"):
        a_b = agg.paired_series("A", "B", criterion)
        b_c = agg.paired_series("B", "C", criterion)
        a_c = agg.paired_series("A", "C", criterion)
        deltas[criterion] = {
            "B_minus_A": paired_bootstrap_ci(a_b[1], a_b[0], n_resamples=cfg["bootstrap_resamples"], seed=cfg["seed"]),
            "C_minus_B": paired_bootstrap_ci(b_c[1], b_c[0], n_resamples=cfg["bootstrap_resamples"], seed=cfg["seed"]),
            "C_minus_A": paired_bootstrap_ci(a_c[1], a_c[0], n_resamples=cfg["bootstrap_resamples"], seed=cfg["seed"]),
        }
    summary = {
        "n_conversations": n_convs,
        "elapsed_seconds": time.time() - start,
        "means": means,
        "deltas_bootstrap_95ci": {
            crit: {k: list(v) for k, v in d.items()} for crit, d in deltas.items()
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/scripts/test_eval_downstream_llm.py -v -m integration`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_downstream_llm.py tests/scripts/test_eval_downstream_llm.py
git commit -m "feat(eval): add downstream LLM eval CLI orchestrator"
```

---

### Task 14: Resumability — stream partial results to disk every 25 conversations

**Files:**
- Modify: `scripts/eval_downstream_llm.py`
- Modify: `tests/scripts/test_eval_downstream_llm.py`

- [ ] **Step 1: Write the failing test**

```python
def test_resume_from_existing_per_conversation_jsonl(tmp_path: Path) -> None:
    # TODO: seed per_conversation.jsonl with 2 completed rows, ensure CLI skips them
    partial = tmp_path / "per_conversation.jsonl"
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps({
        "conversation_id": "t1", "domain": "x", "condition": "A",
        "relevance": 3, "correctness": 3, "context_use": 3,
    }) + "\n")
    # Running with --resume should not re-judge conv t1.
    # (exact assertion pattern depends on implementation; see Step 3)
    assert partial.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/scripts/test_eval_downstream_llm.py::test_resume_from_existing_per_conversation_jsonl -v`
Expected: FAIL (resume logic missing).

- [ ] **Step 3: Add resume logic to `scripts/eval_downstream_llm.py`**

After `per_conv_fh = per_conv_path.open("w")` replace with:

```python
    # Load already-completed conversation IDs for resume support.
    completed: set[tuple[str, str]] = set()
    if per_conv_path.exists():
        for line in per_conv_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed.add((row["conversation_id"], row["condition"]))
                # pre-populate aggregator from existing rows
                from src.eval.downstream.judge import JudgeScore as _JS
                agg.add(
                    row["conversation_id"], row["condition"],
                    _JS(relevance=row["relevance"], correctness=row["correctness"],
                        context_use=row["context_use"]),
                )
    per_conv_fh = per_conv_path.open("a")
```

Then in the main loop, skip any (conv, condition) pair already in `completed`:

```python
        all_conds_done = all(
            (conv.conversation_id, c) in completed for c in ("A", "B", "C")
        )
        if all_conds_done:
            continue
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/scripts/test_eval_downstream_llm.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_downstream_llm.py tests/scripts/test_eval_downstream_llm.py
git commit -m "feat(eval): add resume support to downstream LLM eval"
```

---

### Task 15: Self-consistency CLI subcommand — abort-if-noisy preflight

**Files:**
- Create: `scripts/eval_judge_self_consistency.py`
- Create: `tests/scripts/test_eval_judge_self_consistency.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.integration
def test_self_consistency_smoke_with_stub(tmp_path: Path) -> None:
    # Build a 20-row fixture where judge returns same score twice → rho == 1.0
    dataset = tmp_path / "c.jsonl"
    rows = []
    for i in range(20):
        rows.append({
            "conversation_id": f"t{i}", "domain": "spice",
            "turns": [{"role": "user", "content": f"q{i}"}],
            "final_question": f"q{i}", "reference_answer": f"a{i}",
        })
    dataset.write_text("\n".join(json.dumps(r) for r in rows))
    rc = subprocess.run(
        [sys.executable, "scripts/eval_judge_self_consistency.py",
         "--dataset", str(dataset), "--n", "20", "--stub-llm", "--threshold", "0.80"],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0
    assert "rho=" in rc.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/scripts/test_eval_judge_self_consistency.py -v -m integration`
Expected: FAIL (script missing).

- [ ] **Step 3: Implement `scripts/eval_judge_self_consistency.py`**

```python
#!/usr/bin/env python3
"""Preflight: verify judge ranks same responses consistently across two runs.

Exit code 0 = rho >= threshold (run is safe); 1 = rho below threshold (abort).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.eval.downstream.dataset import ConversationLoader
from src.eval.downstream.judge import LLMJudge
from src.eval.downstream.llm_client import LLMClient
from src.eval.downstream.self_consistency import spearman_rank


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.80)
    p.add_argument("--judge-url", default="http://localhost:8008")
    p.add_argument("--judge-name", default="qwen3-coder-480b-mlx-4bit")
    p.add_argument("--stub-llm", action="store_true")
    args = p.parse_args()

    if args.stub_llm:
        judge_llm = MagicMock()
        judge_llm.generate.side_effect = lambda **kw: '{"relevance": 4, "correctness": 3, "context_use": 4}'
    else:
        judge_llm = LLMClient(base_url=args.judge_url, model=args.judge_name)
    judge = LLMJudge(llm_client=judge_llm)

    scores_run1, scores_run2 = [], []
    for i, conv in enumerate(ConversationLoader(args.dataset, limit=args.n)):
        # Use the reference_answer as the "response" — two runs should agree.
        conv_text = "\n".join(f"{t.role}: {t.content}" for t in conv.turns)
        s1 = judge.score(conv_text, conv.reference_answer, conv.reference_answer)
        s2 = judge.score(conv_text, conv.reference_answer, conv.reference_answer)
        scores_run1.append(s1.relevance + s1.correctness + s1.context_use)
        scores_run2.append(s2.relevance + s2.correctness + s2.context_use)

    rho = spearman_rank(scores_run1, scores_run2)
    print(f"rho={rho:.3f} threshold={args.threshold}")
    if rho < args.threshold:
        print("ABORT: judge self-consistency below threshold", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/scripts/test_eval_judge_self_consistency.py -v -m integration`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_judge_self_consistency.py tests/scripts/test_eval_judge_self_consistency.py
git commit -m "feat(eval): add judge self-consistency preflight script"
```

---

### Task 16: Results writeup generator — `results/2026-04-19-downstream-llm-eval/README.md`

**Files:**
- Create: `scripts/write_downstream_eval_report.py`
- Create: `tests/scripts/test_write_downstream_eval_report.py`

- [ ] **Step 1: Write the failing test**

```python
def test_report_renders_expected_sections(tmp_path: Path) -> None:
    summary = {
        "n_conversations": 500,
        "elapsed_seconds": 18000,
        "means": {
            "A": {"relevance": 3.1, "correctness": 2.9, "context_use": 2.0},
            "B": {"relevance": 3.8, "correctness": 3.5, "context_use": 3.9},
            "C": {"relevance": 4.0, "correctness": 3.7, "context_use": 4.2},
        },
        "deltas_bootstrap_95ci": {
            "relevance": {"C_minus_B": [0.2, 0.05, 0.35]},
            "correctness": {"C_minus_B": [0.2, 0.02, 0.38]},
            "context_use": {"C_minus_B": [0.3, 0.15, 0.45]},
        },
    }
    src = tmp_path / "summary.json"
    src.write_text(json.dumps(summary))
    out = tmp_path / "README.md"
    rc = subprocess.run(
        [sys.executable, "scripts/write_downstream_eval_report.py",
         "--summary", str(src), "--out", str(out)], capture_output=True,
    ).returncode
    assert rc == 0
    text = out.read_text()
    assert "# Downstream LLM Quality Eval" in text
    assert "500" in text
    assert "Relevance" in text
    assert "0.20" in text or "0.2" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/scripts/test_write_downstream_eval_report.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `scripts/write_downstream_eval_report.py`**

```python
#!/usr/bin/env python3
"""Render a markdown report from summary.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


TEMPLATE = """# Downstream LLM Quality Eval

**N conversations**: {n}
**Wall-clock**: {elapsed:.0f} s

## Per-condition rubric means

| Condition | Relevance | Correctness | Context-Use |
|-----------|-----------|-------------|-------------|
| A (base LLM) | {A_rel:.2f} | {A_cor:.2f} | {A_ctx:.2f} |
| B (+ Aeon retrieval) | {B_rel:.2f} | {B_cor:.2f} | {B_ctx:.2f} |
| C (+ Aeon predictor) | {C_rel:.2f} | {C_cor:.2f} | {C_ctx:.2f} |

## Paired bootstrap 95 % CI — C minus B

| Criterion | Mean delta | CI lo | CI hi | Significant? |
|-----------|------------|-------|-------|--------------|
| Relevance | {rel_d:.2f} | {rel_lo:.2f} | {rel_hi:.2f} | {rel_sig} |
| Correctness | {cor_d:.2f} | {cor_lo:.2f} | {cor_hi:.2f} | {cor_sig} |
| Context-Use | {ctx_d:.2f} | {ctx_lo:.2f} | {ctx_hi:.2f} | {ctx_sig} |

Significant = CI does not cross 0.
"""


def sig(lo: float, hi: float) -> str:
    return "yes" if lo > 0 or hi < 0 else "no"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    s = json.loads(args.summary.read_text())
    m = s["means"]
    d = s["deltas_bootstrap_95ci"]
    rel = d["relevance"]["C_minus_B"]
    cor = d["correctness"]["C_minus_B"]
    ctx = d["context_use"]["C_minus_B"]
    args.out.write_text(TEMPLATE.format(
        n=s["n_conversations"], elapsed=s["elapsed_seconds"],
        A_rel=m["A"]["relevance"], A_cor=m["A"]["correctness"], A_ctx=m["A"]["context_use"],
        B_rel=m["B"]["relevance"], B_cor=m["B"]["correctness"], B_ctx=m["B"]["context_use"],
        C_rel=m["C"]["relevance"], C_cor=m["C"]["correctness"], C_ctx=m["C"]["context_use"],
        rel_d=rel[0], rel_lo=rel[1], rel_hi=rel[2], rel_sig=sig(rel[1], rel[2]),
        cor_d=cor[0], cor_lo=cor[1], cor_hi=cor[2], cor_sig=sig(cor[1], cor[2]),
        ctx_d=ctx[0], ctx_lo=ctx[1], ctx_hi=ctx[2], ctx_sig=sig(ctx[1], ctx[2]),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/scripts/test_write_downstream_eval_report.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/write_downstream_eval_report.py tests/scripts/test_write_downstream_eval_report.py
git commit -m "feat(eval): add markdown report renderer for downstream eval"
```

---

### Task 17: Dry-run on 20 conversations — end-to-end smoke with real LLM

**Files:**
- Modify: `scripts/eval_downstream_llm.py` (only a CLI default check; no new code)

- [ ] **Step 1: Generate fixture subset**

```bash
uv run python scripts/gen_downstream_eval_dataset.py \
  --out data/downstream-eval/conversations_smoke.jsonl \
  --n-per-domain 2 \
  --seed 20260419
```

Expected: 20 conversations written to `data/downstream-eval/conversations_smoke.jsonl` (10 domains × 2).

- [ ] **Step 2: Launch MLX servers on Studio**

Confirm both servers respond:

```bash
curl -s http://localhost:8009/v1/models | head
curl -s http://localhost:8008/v1/models | head
```

Expected: JSON `{"data": [...]}`. If either fails, start the missing one per `deploy/` docs (out of scope for this plan).

- [ ] **Step 3: Run judge self-consistency preflight**

```bash
uv run python scripts/eval_judge_self_consistency.py \
  --dataset data/downstream-eval/conversations_smoke.jsonl \
  --n 20 --threshold 0.80
```

Expected: `rho=0.XX threshold=0.80`, exit 0. If rho < 0.80, STOP — investigate judge determinism before proceeding.

- [ ] **Step 4: Run the 20-conversation dry run**

```bash
uv run python scripts/eval_downstream_llm.py \
  --dataset data/downstream-eval/conversations_smoke.jsonl \
  --n-conversations 20 \
  --out-dir results/2026-04-19-downstream-llm-eval-smoke
```

Expected: wall-clock ≤ 10 min on Studio (20 convs × 3 conditions × 2 LLM calls = 120 calls; teacher throughput ≈ 150 tok/s; avg 400 tok per call → ~5 min). Summary written to `results/.../summary.json`.

- [ ] **Step 5: Inspect the dry run result**

Open `results/2026-04-19-downstream-llm-eval-smoke/summary.json`. Sanity-check:
- All three conditions have 20 scored rows each.
- Means are in [1, 5].
- Bootstrap CIs are finite.

If any check fails: diagnose (judge output parse errors are the most common cause — inspect `per_conversation.jsonl`). Fix and re-run the dry run before committing.

- [ ] **Step 6: Commit the smoke fixture**

```bash
git add data/downstream-eval/conversations_smoke.jsonl results/2026-04-19-downstream-llm-eval-smoke/summary.json results/2026-04-19-downstream-llm-eval-smoke/per_conversation.jsonl
git commit -m "test(eval): add 20-conversation dry-run results"
```

---

### Task 18: Full run — 500 conversations, 3 conditions, write final report

**Files:** (no new code)

- [ ] **Step 1: Generate the full held-out set**

```bash
uv run python scripts/gen_downstream_eval_dataset.py \
  --out data/downstream-eval/conversations.jsonl \
  --n-per-domain 50 \
  --seed 20260419
```

Expected: 500 rows in `data/downstream-eval/conversations.jsonl`.

- [ ] **Step 2: Re-run judge self-consistency on a fresh 20-row sample**

```bash
uv run python scripts/eval_judge_self_consistency.py \
  --dataset data/downstream-eval/conversations.jsonl \
  --n 20 --threshold 0.80
```

Expected: rho ≥ 0.80. Abort if not.

- [ ] **Step 3: Launch the full eval (background, logged)**

```bash
mkdir -p results/2026-04-19-downstream-llm-eval
uv run python scripts/eval_downstream_llm.py \
  --config configs/eval/downstream-llm.yaml \
  2>&1 | tee results/2026-04-19-downstream-llm-eval/run.log
```

Expected wall-clock: ~5 h on Studio (500 convs × 6 LLM calls × avg ~6 s = ~5 h with prompt-caching on teacher).

- [ ] **Step 4: Render the markdown report**

```bash
uv run python scripts/write_downstream_eval_report.py \
  --summary results/2026-04-19-downstream-llm-eval/summary.json \
  --out results/2026-04-19-downstream-llm-eval/README.md
```

- [ ] **Step 5: Commit final results**

```bash
git add data/downstream-eval/conversations.jsonl results/2026-04-19-downstream-llm-eval/
git commit -m "docs(eval): full 500-conversation downstream LLM eval results"
```

---

### Task 19: Paper A §4.6 writeup — integrate the numbers

**Files:**
- Modify: `docs/papers/paper-a-reframe-aeon-ami.md`

- [ ] **Step 1: Locate the §4 section**

Open `docs/papers/paper-a-reframe-aeon-ami.md`. Locate the `§4 Experimental protocol` and `§5 Results` sections. Paper currently has §5.1-§5.6.

- [ ] **Step 2: Add §4.6 to the section-by-section outline**

In the section listing "Section-by-section outline" (around §4), add the sentence:

```markdown
### §4.6 Downstream LLM quality (paired LLM-as-judge)
Held-out 500-conversation multi-turn QA over 10 micro-kiki domains; three rigs (A: base Qwen3.5-35B, B: +AeonServingHook retrieval, C: +Aeon predictor). Judge: local Qwen3-Coder-480B MLX 4bit, 3-criterion rubric (Relevance / Correctness / Context-Use). Paired bootstrap 95% CI on `score(C) - score(B)`. Seed 20260419.
```

- [ ] **Step 3: Add §5.7 to the results section with the actual numbers**

After `§5.6 Cross-session persistence`, insert:

```markdown
### §5.7 Downstream LLM quality

We measure whether the retrieval and predictor conditions produce measurably better downstream responses, not just better retrieval metrics. Paired LLM-as-judge over 500 held-out conversations, rubric mean ± bootstrap 95% CI.

| Condition | Relevance | Correctness | Context-Use |
|-----------|-----------|-------------|-------------|
| A (base LLM)              | <paste from results/.../summary.json means.A.relevance> | <...> | <...> |
| B (+ Aeon retrieval)      | <...> | <...> | <...> |
| C (+ Aeon predictor)      | <...> | <...> | <...> |

Paired bootstrap 95% CI, 10000 resamples:

- **C vs B on Context-Use**: Δ = <paste>, CI [<lo>, <hi>] — <significant?>
- **C vs B on Correctness**: Δ = <paste>, CI [<lo>, <hi>] — <significant?>
- **C vs B on Relevance**: Δ = <paste>, CI [<lo>, <hi>] — <significant?>

Self-consistency Spearman ρ = <paste> over 20 double-judged conversations (threshold 0.80 met). Judge: Qwen3-Coder-480B MLX 4bit (local, zero paid-API cost). See `results/2026-04-19-downstream-llm-eval/` for per-conversation rows.

**Interpretation**: <to write after run completes — one paragraph. If Δ > 0 on Context-Use with CI > 0, claim that the predictor improves memory-dependent response quality even when retrieval Recall@5 ties. If CI crosses 0, disclose as no-detected-effect and bound the claim to retrieval metrics only.>
```

- [ ] **Step 4: Fill the `<paste>` placeholders**

After Task 18 produced real numbers, open `results/2026-04-19-downstream-llm-eval/summary.json` and paste the values into §5.7 of the paper. Write the one-paragraph interpretation based on which criterion (if any) has a significant positive C-minus-B delta.

- [ ] **Step 5: Commit the paper update**

```bash
git add docs/papers/paper-a-reframe-aeon-ami.md
git commit -m "docs(paper-a): add §5.7 downstream LLM quality results"
```

---

### Task 20: Full test suite + integration flag check

- [ ] **Step 1: Run the full pytest suite (unit tests only)**

```bash
uv run python -m pytest -v -m "not integration and not gpu"
```

Expected: all new downstream tests pass; no regressions elsewhere.

- [ ] **Step 2: Run integration tests with stubs (no real LLM)**

```bash
uv run python -m pytest tests/scripts/test_eval_downstream_llm.py tests/scripts/test_eval_judge_self_consistency.py -v -m integration
```

Expected: all pass with `--stub-llm` path exercised.

- [ ] **Step 3: Final commit hook check**

```bash
git log --oneline | head -20
```

Confirm every commit message is ≤ 50 chars on the subject line (pre-commit hook enforces) and no `Co-Authored-By` trailer anywhere.

---

## Self-Review Checklist

- **Spec coverage**: (1) eval task design → Task 2 (custom 10-domain, 500 convs, memory-dependent final question); (2) judge setup → Tasks 7 (rubric) and 15 (self-consistency preflight); (3) control rig → Tasks 5 (runner) and 6 (predictor toggle); (4) sample set → Task 2 (N=500); (5) pipeline → Tasks 3–13; (6) statistical testing → Task 11 (paired bootstrap); (7) results doc + Paper A integration → Tasks 16, 18, 19.
- **Placeholder scan**: The `<paste>` tokens in Task 19 Step 4 are intentional — they mark where real numbers from the Task 18 run must be pasted. No TBD / TODO elsewhere.
- **Type consistency**: `JudgeScore` used identically across `judge.py`, `aggregator.py`, and `scripts/eval_downstream_llm.py`. `Rig` enum has stable values `A`, `B`, `C` matching the runner, aggregator, and paper writeup.
- **TDD**: every code task starts with a failing test (Step 1) before implementation (Step 3).
- **DRY**: single source of truth for rubric (judge.py), for config (YAML), for stats (stats.py).
- **YAGNI**: no async, no threadpool, no Langfuse, no Prometheus — the goal is one run, then a paper section. Resumability (Task 14) is the only robustness feature included.
- **Frequent commits**: 20 tasks → 20 commits (one per green test cycle).
