# Domain-Specific Embedding Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hash-based embedding stub in Aeon Memory with a contrastive sentence-transformers model trained on the 10 niche domains.

**Architecture:** Fine-tune `all-MiniLM-L6-v2` (384d) using MNRL + TripletLoss hard negatives on domain texts from `data/final/`. Wire the trained model into `AeonPalace` via an `embed_fn` callable, removing the hash fallback entirely. Atlas dimension becomes configurable at runtime.

**Tech Stack:** sentence-transformers >= 3.0, numpy, pytest

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `pyproject.toml` | Add `[embeddings]` optional dep | Modify |
| `src/memory/aeon.py` | Remove hash fallback, add `load_st_model()`, new constructor API | Modify |
| `tests/memory/test_aeon.py` | Update fixtures to use mock `embed_fn`, test ImportError on missing model | Modify |
| `scripts/train_embeddings.py` | Rewrite: MNRL + hard negatives, drop mlx_tune | Rewrite |
| `tests/scripts/test_train_embeddings.py` | Unit tests for data loading, pair building, oversampling | Create |

---

### Task 1: Add `[embeddings]` optional dependency

**Files:**
- Modify: `pyproject.toml:16-38`

- [ ] **Step 1: Add the dependency group**

In `pyproject.toml`, after the `mlx` group (line 29), add:

```toml
embeddings = [
  "sentence-transformers>=3.0",
]
```

- [ ] **Step 2: Verify pyproject.toml parses**

Run: `python3 -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat(deps): add embeddings optional group"
```

---

### Task 2: Refactor AeonPalace — remove hash fallback

**Files:**
- Modify: `src/memory/aeon.py`
- Modify: `tests/memory/test_aeon.py`

- [ ] **Step 1: Write failing test — ImportError on missing embed_fn**

Add to `tests/memory/test_aeon.py`:

```python
class TestAeonPalaceEmbedFn:
    def test_raises_without_embed_fn_or_model_path(self):
        with pytest.raises(ImportError, match="requires an embed_fn"):
            AeonPalace()

    def test_accepts_custom_embed_fn(self):
        fn = lambda text: np.random.randn(64).astype(np.float32)
        aeon = AeonPalace(dim=64, embed_fn=fn)
        eid = aeon.write("test content", domain="test")
        assert len(eid) == 16

    def test_accepts_model_path_string(self):
        """model_path with non-existent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            AeonPalace(model_path="/nonexistent/model")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/memory/test_aeon.py::TestAeonPalaceEmbedFn -v`
Expected: FAIL — `AeonPalace()` currently succeeds (hash fallback)

- [ ] **Step 3: Update AeonPalace constructor**

Replace the full `AeonPalace` class `__init__` and `_default_embed` in `src/memory/aeon.py`:

```python
class AeonPalace:
    """Unified memory palace combining vector search + episodic graph."""

    def __init__(
        self,
        dim: int | None = None,
        embed_fn=None,
        model_path: str | None = None,
    ) -> None:
        if embed_fn is not None:
            self._embed_fn = embed_fn
            self._dim = dim or 384
        elif model_path is not None:
            self._embed_fn, self._dim = _load_st_model(model_path)
        else:
            raise ImportError(
                "AeonPalace requires an embed_fn or model_path. "
                "Train one with: python3 scripts/train_embeddings.py --all"
            )
        self._atlas = AtlasIndex(dim=self._dim)
        self._trace = TraceGraph()
```

Remove the `_default_embed` static method entirely.

Add the helper function above the class:

```python
def _load_st_model(model_path: str) -> tuple:
    """Load a sentence-transformers model and return (embed_fn, dim).

    Raises FileNotFoundError if path does not exist.
    Raises ImportError if sentence-transformers is not installed.
    """
    from pathlib import Path

    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Embedding model not found at {model_path}. "
            "Train one with: python3 scripts/train_embeddings.py --all"
        )
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for model_path loading. "
            "Install with: pip install 'micro-kiki[embeddings]'"
        )
    model = SentenceTransformer(str(p))
    dim = model.get_sentence_embedding_dimension()
    logger.info("Loaded embedding model from %s (dim=%d)", p, dim)

    def embed_fn(text: str) -> np.ndarray:
        return model.encode(text, normalize_embeddings=True).astype(np.float32)

    return embed_fn, dim
```

- [ ] **Step 4: Update existing tests to pass embed_fn**

In `tests/memory/test_aeon.py`, update `TestAeonPalace` to use a mock:

```python
def _mock_embed(dim: int = 64):
    """Return a deterministic hash-based embed_fn for tests."""
    import hashlib

    def fn(text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec = rng.randn(dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)

    return fn


class TestAeonPalace:
    def test_write_and_recall(self):
        aeon = AeonPalace(dim=64, embed_fn=_mock_embed(64))
        eid = aeon.write("Test episode about MoE-LoRA", domain="ml")
        results = aeon.recall("MoE-LoRA", top_k=1)
        assert len(results) >= 1

    def test_write_with_links(self):
        aeon = AeonPalace(dim=64, embed_fn=_mock_embed(64))
        e1 = aeon.write("First event", domain="test")
        e2 = aeon.write("Second event", domain="test", links=[e1])
        walked = aeon.walk(e1, max_depth=2)
        assert len(walked) >= 2

    def test_compress(self):
        aeon = AeonPalace(dim=64, embed_fn=_mock_embed(64))
        old_ts = datetime.now() - timedelta(days=60)
        aeon.write("Old content that is very long " * 20,
                   domain="test", timestamp=old_ts)
        compressed = aeon.compress(
            older_than=datetime.now() - timedelta(days=30))
        assert compressed == 1

    def test_stats(self):
        aeon = AeonPalace(dim=64, embed_fn=_mock_embed(64))
        aeon.write("Episode 1", domain="a")
        aeon.write("Episode 2", domain="b")
        stats = aeon.stats
        assert stats["vectors"] == 2
        assert stats["episodes"] == 2
```

- [ ] **Step 5: Run all memory tests**

Run: `python3 -m pytest tests/memory/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/memory/aeon.py tests/memory/test_aeon.py
git commit -m "feat(aeon): remove hash fallback, require embed_fn"
```

---

### Task 3: Rewrite train_embeddings.py

**Files:**
- Rewrite: `scripts/train_embeddings.py`
- Create: `tests/scripts/test_train_embeddings.py`

- [ ] **Step 1: Write tests for data loading and pair building**

Create `tests/scripts/test_train_embeddings.py`:

```python
"""Tests for train_embeddings data pipeline (no model needed)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

# Patch REPO_ROOT before import
import scripts.train_embeddings as te


class TestLoadTexts:
    def test_loads_messages_format(self, tmp_path):
        d = tmp_path / "spice"
        d.mkdir()
        f = d / "train.jsonl"
        f.write_text(
            json.dumps({"messages": [
                {"role": "user", "content": "Design a buck converter"},
                {"role": "assistant", "content": "Here is the SPICE netlist..."},
            ]}) + "\n"
        )
        texts = te.load_texts_from_file(f)
        assert len(texts) == 2
        assert "buck converter" in texts[0]

    def test_loads_prompt_format(self, tmp_path):
        d = tmp_path / "emc"
        d.mkdir()
        f = d / "train.jsonl"
        f.write_text(
            json.dumps({"prompt": "Design an EMI filter"}) + "\n"
        )
        texts = te.load_texts_from_file(f)
        assert len(texts) == 1

    def test_skips_short_texts(self, tmp_path):
        d = tmp_path / "x"
        d.mkdir()
        f = d / "train.jsonl"
        f.write_text(json.dumps({"prompt": "hi"}) + "\n")
        texts = te.load_texts_from_file(f, min_len=20)
        assert len(texts) == 0


class TestBuildPairs:
    def test_mnrl_pairs_are_same_domain(self):
        domain_texts = {
            "spice": ["text A about SPICE", "text B about SPICE"],
            "emc": ["text C about EMC", "text D about EMC"],
        }
        pairs = te.build_mnrl_pairs(domain_texts)
        assert len(pairs) == 4
        for anchor, positive in pairs:
            assert isinstance(anchor, str)
            assert isinstance(positive, str)

    def test_hard_negative_triplets(self):
        domain_texts = {
            "embedded": ["firmware code for STM32"],
            "stm32": ["HAL driver for STM32F4"],
        }
        triplets = te.build_hard_negative_triplets(domain_texts)
        assert len(triplets) >= 1
        anchor, positive, negative = triplets[0]
        assert isinstance(anchor, str)
        assert isinstance(negative, str)


class TestOversample:
    def test_oversamples_small_domains(self):
        texts = ["one", "two", "three"]
        result = te.oversample(texts, target=10, seed=42)
        assert len(result) == 10
        assert "one" in result

    def test_no_change_when_already_enough(self):
        texts = ["a", "b", "c", "d", "e"]
        result = te.oversample(texts, target=3, seed=42)
        assert len(result) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/scripts/test_train_embeddings.py -v`
Expected: FAIL — functions not defined yet

- [ ] **Step 3: Rewrite train_embeddings.py**

Replace the entire content of `scripts/train_embeddings.py`:

```python
#!/usr/bin/env python3
"""Fine-tune domain-specific embedding model for Aeon memory recall.

Uses sentence-transformers with two loss functions:
1. MultipleNegativesRankingLoss (MNRL) — in-batch negatives
2. TripletLoss — hard negatives from confusing domain pairs

Usage:
    python3 scripts/train_embeddings.py --all
    python3 scripts/train_embeddings.py --dry-run
    python3 scripts/train_embeddings.py --domains spice emc power
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FINAL_DATA = REPO_ROOT / "data" / "final"
OUTPUT_DIR = REPO_ROOT / "models" / "niche-embeddings"

NICHE_DOMAINS = [
    "kicad-dsl", "spice", "emc", "stm32", "embedded",
    "freecad", "platformio", "power", "dsp", "electronics",
]

# Hard negative pairs: domains that are easily confused
HARD_NEGATIVE_PAIRS = [
    ("embedded", "stm32"),
    ("spice", "power"),
    ("kicad-dsl", "electronics"),
    ("embedded", "platformio"),
]

MIN_DOMAIN_EXAMPLES = 100  # oversample below this
MAX_DOMAIN_EXAMPLES = 2000  # cap above this


def load_texts_from_file(path: Path, min_len: int = 20) -> list[str]:
    """Extract text strings from a JSONL file."""
    texts: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "messages" in d:
                for m in d["messages"]:
                    c = m.get("content", "")
                    if len(c) >= min_len:
                        texts.append(c)
            elif "prompt" in d:
                p = d["prompt"]
                if len(p) >= min_len:
                    texts.append(p)
            elif "instruction" in d:
                p = d["instruction"]
                if len(p) >= min_len:
                    texts.append(p)
    return texts


def load_all_domain_texts(domains: list[str]) -> dict[str, list[str]]:
    """Load texts for each domain from data/final/."""
    domain_texts: dict[str, list[str]] = {}
    for domain in domains:
        train_file = FINAL_DATA / domain / "train.jsonl"
        if not train_file.exists():
            logger.warning("No data for %s at %s", domain, train_file)
            domain_texts[domain] = []
            continue
        texts = load_texts_from_file(train_file)
        # Cap
        if len(texts) > MAX_DOMAIN_EXAMPLES:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(texts), MAX_DOMAIN_EXAMPLES, replace=False)
            texts = [texts[i] for i in sorted(idx)]
        # Oversample
        if 0 < len(texts) < MIN_DOMAIN_EXAMPLES:
            texts = oversample(texts, MIN_DOMAIN_EXAMPLES, seed=42)
        domain_texts[domain] = texts
        logger.info("  %-15s %4d texts", domain, len(texts))
    return domain_texts


def oversample(texts: list[str], target: int, seed: int = 42) -> list[str]:
    """Repeat texts to reach target count."""
    if len(texts) >= target:
        return texts
    rng = np.random.default_rng(seed)
    extra = rng.choice(len(texts), target - len(texts), replace=True)
    return texts + [texts[i] for i in extra]


def build_mnrl_pairs(
    domain_texts: dict[str, list[str]],
) -> list[tuple[str, str]]:
    """Build (anchor, positive) pairs from same-domain texts for MNRL."""
    pairs: list[tuple[str, str]] = []
    for domain, texts in domain_texts.items():
        if len(texts) < 2:
            continue
        for i in range(0, len(texts) - 1, 2):
            pairs.append((texts[i], texts[i + 1]))
    return pairs


def build_hard_negative_triplets(
    domain_texts: dict[str, list[str]],
) -> list[tuple[str, str, str]]:
    """Build (anchor, positive, negative) triplets from confusing pairs."""
    triplets: list[tuple[str, str, str]] = []
    rng = np.random.default_rng(42)

    for domain_a, domain_b in HARD_NEGATIVE_PAIRS:
        texts_a = domain_texts.get(domain_a, [])
        texts_b = domain_texts.get(domain_b, [])
        if len(texts_a) < 2 or len(texts_b) < 1:
            continue

        # anchor + positive from domain_a, negative from domain_b
        for i in range(0, len(texts_a) - 1, 2):
            neg_idx = int(rng.integers(0, len(texts_b)))
            triplets.append((texts_a[i], texts_a[i + 1], texts_b[neg_idx]))

        # Reverse: anchor + positive from domain_b, negative from domain_a
        for i in range(0, len(texts_b) - 1, 2):
            neg_idx = int(rng.integers(0, len(texts_a)))
            triplets.append((texts_b[i], texts_b[i + 1], texts_a[neg_idx]))

    return triplets


def train(domains: list[str], args: argparse.Namespace) -> None:
    """Train embedding model with MNRL + TripletLoss."""
    try:
        from sentence_transformers import (
            SentenceTransformer,
            InputExample,
            losses,
        )
        from torch.utils.data import DataLoader
    except ImportError:
        logger.error(
            "sentence-transformers required. Install: "
            "pip install 'micro-kiki[embeddings]'"
        )
        sys.exit(1)

    logger.info("Loading domain texts...")
    domain_texts = load_all_domain_texts(domains)

    active = {d: t for d, t in domain_texts.items() if t}
    if not active:
        logger.error("No training data found.")
        return

    # Phase 1: MNRL pairs
    mnrl_pairs = build_mnrl_pairs(domain_texts)
    logger.info("MNRL pairs: %d", len(mnrl_pairs))

    mnrl_examples = [
        InputExample(texts=[a, p]) for a, p in mnrl_pairs
    ]

    # Phase 2: Hard negative triplets
    triplets = build_hard_negative_triplets(domain_texts)
    logger.info("Hard negative triplets: %d", len(triplets))

    triplet_examples = [
        InputExample(texts=[a, p, n]) for a, p, n in triplets
    ]

    # Load model
    logger.info("Loading all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # DataLoaders
    mnrl_loader = DataLoader(
        mnrl_examples, shuffle=True, batch_size=args.batch_size,
    )
    mnrl_loss = losses.MultipleNegativesRankingLoss(model)

    train_objectives = [(mnrl_loader, mnrl_loss)]

    if triplet_examples:
        triplet_loader = DataLoader(
            triplet_examples, shuffle=True, batch_size=args.batch_size,
        )
        triplet_loss = losses.TripletLoss(
            model=model, distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=args.margin,
        )
        train_objectives.append((triplet_loader, triplet_loss))

    # Train
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Training: %d epochs, batch=%d, margin=%.2f",
        args.epochs, args.batch_size, args.margin,
    )
    t0 = time.time()
    model.fit(
        train_objectives=train_objectives,
        epochs=args.epochs,
        warmup_steps=int(
            0.1 * len(mnrl_loader) * args.epochs
        ),
        output_path=str(OUTPUT_DIR),
        show_progress_bar=True,
    )
    elapsed = time.time() - t0
    logger.info("Training done in %.1fs", elapsed)
    logger.info("Model saved to %s", OUTPUT_DIR)

    # Quick eval
    _eval_separation(model, domain_texts)


def _eval_separation(model, domain_texts: dict[str, list[str]]) -> None:
    """Print intra-domain vs inter-domain cosine similarity."""
    from sentence_transformers import util

    domains_with_data = [d for d, t in domain_texts.items() if len(t) >= 4]
    if len(domains_with_data) < 2:
        logger.warning("Not enough domains for eval")
        return

    intra_scores: list[float] = []
    inter_scores: list[float] = []

    for domain in domains_with_data[:5]:
        texts = domain_texts[domain][:10]
        embs = model.encode(texts)
        sims = util.cos_sim(embs, embs).numpy()
        n = len(texts)
        for i in range(n):
            for j in range(i + 1, n):
                intra_scores.append(float(sims[i][j]))

    for i, d1 in enumerate(domains_with_data[:5]):
        for d2 in domains_with_data[i + 1: i + 3]:
            e1 = model.encode(domain_texts[d1][:5])
            e2 = model.encode(domain_texts[d2][:5])
            sims = util.cos_sim(e1, e2).numpy()
            inter_scores.extend(sims.flatten().tolist())

    avg_intra = np.mean(intra_scores) if intra_scores else 0
    avg_inter = np.mean(inter_scores) if inter_scores else 0
    logger.info(
        "Eval: intra-domain cosine=%.3f (target>0.7), "
        "inter-domain cosine=%.3f (target<0.3)",
        avg_intra, avg_inter,
    )


def dry_run(domains: list[str]) -> None:
    """Show what would be trained without loading models."""
    logger.info("DRY RUN")
    domain_texts = load_all_domain_texts(domains)
    mnrl_pairs = build_mnrl_pairs(domain_texts)
    triplets = build_hard_negative_triplets(domain_texts)
    logger.info("MNRL pairs: %d", len(mnrl_pairs))
    logger.info("Hard negative triplets: %d", len(triplets))
    logger.info("Output: %s", OUTPUT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train domain-specific embeddings for Aeon memory.",
    )
    parser.add_argument("--all", action="store_true",
                        help="Train on all niche domains")
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.3,
                        help="TripletLoss margin")
    args = parser.parse_args()

    domains = args.domains or (NICHE_DOMAINS if args.all else [])
    if not domains:
        parser.print_help()
        return

    if args.dry_run:
        dry_run(domains)
    else:
        train(domains, args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run data pipeline tests**

Run: `python3 -m pytest tests/scripts/test_train_embeddings.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run dry-run to verify data loading**

Run: `python3 scripts/train_embeddings.py --all --dry-run`
Expected: Shows per-domain counts and pair/triplet totals

- [ ] **Step 6: Commit**

```bash
git add scripts/train_embeddings.py tests/scripts/test_train_embeddings.py
git commit -m "feat(embeddings): MNRL + hard negatives training"
```

---

### Task 4: Update POC pipeline to use embed_fn

**Files:**
- Modify: `scripts/poc_pipeline_v2.py:113-116`

- [ ] **Step 1: Update MicroKikiPipeline to pass embed_fn**

In `scripts/poc_pipeline_v2.py`, find the `__init__` where `AeonPalace()` is constructed. Replace:

```python
        logger.info("[2/4] Initializing Aeon Memory Palace...")
        self.memory = AeonPalace()
        logger.info("  Memory: Atlas vector + Trace episodic graph")
```

with:

```python
        logger.info("[2/4] Initializing Aeon Memory Palace...")
        # Use hash-based embed_fn for POC (no sentence-transformers needed)
        import hashlib as _hl

        def _poc_embed(text: str) -> np.ndarray:
            h = _hl.sha256(text.encode()).digest()
            rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
            vec = rng.randn(384).astype(np.float32)
            return vec / (np.linalg.norm(vec) + 1e-8)

        # If trained model exists, prefer it
        _model_path = _REPO_ROOT / "models" / "niche-embeddings"
        if _model_path.exists() and (_model_path / "config.json").exists():
            try:
                self.memory = AeonPalace(model_path=str(_model_path))
                logger.info("  Memory: Atlas vector (trained embeddings, dim=%d)", self.memory._dim)
            except ImportError:
                self.memory = AeonPalace(dim=384, embed_fn=_poc_embed)
                logger.info("  Memory: Atlas vector (hash fallback, install sentence-transformers for real embeddings)")
        else:
            self.memory = AeonPalace(dim=384, embed_fn=_poc_embed)
            logger.info("  Memory: Atlas vector (hash embed, train model for semantic recall)")
```

Add `import numpy as np` at the top if not already present.

- [ ] **Step 2: Run POC smoke test**

Run: `python3 -c "import sys; sys.path.insert(0,'.'); exec(open('scripts/poc_pipeline_v2.py').read().split('class MicroKikiPipeline')[0]); print('imports OK')"`
Expected: `imports OK` (verifies no import errors)

- [ ] **Step 3: Commit**

```bash
git add scripts/poc_pipeline_v2.py
git commit -m "feat(poc): wire AeonPalace embed_fn in POC v2"
```

---

### Task 5: Run full test suite and fix regressions

**Files:**
- Possibly modify: any file with failing tests

- [ ] **Step 1: Run full test suite**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: 700+ passed, 0 failed

- [ ] **Step 2: Fix any regressions**

If tests fail, check for places that construct `AeonPalace()` without arguments. Every call site needs either `embed_fn=` or `model_path=`. Common locations:

- `scripts/poc_pipeline.py` (v1)
- `src/serving/aeon_hook.py`
- `scripts/eval_aeon.py`

For each, add the hash-based `embed_fn` as a stopgap (same pattern as Task 4).

- [ ] **Step 3: Run tests again**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "fix(aeon): update all AeonPalace call sites"
```

---

### Task 6: Train and validate embeddings (integration)

**Files:**
- No code changes — execution only

- [ ] **Step 1: Install sentence-transformers**

Run: `pip3 install --break-system-packages 'sentence-transformers>=3.0'`

- [ ] **Step 2: Run training**

Run: `python3 scripts/train_embeddings.py --all --epochs 5`
Expected: Model saved to `models/niche-embeddings/`, eval prints intra-domain > 0.7

- [ ] **Step 3: Verify model files exist**

Run: `ls models/niche-embeddings/`
Expected: `config.json`, `model.safetensors`, `tokenizer.json`, etc.

- [ ] **Step 4: Run POC v2 with trained model**

Run: `python3 scripts/poc_pipeline_v2.py --scenario multi-turn 2>&1 | grep -E 'Memory|trained|semantic'`
Expected: Log shows "trained embeddings" loaded, memory recall uses real embeddings

- [ ] **Step 5: Commit model config (not weights)**

```bash
echo "models/niche-embeddings/*.safetensors" >> .gitignore
echo "models/niche-embeddings/pytorch_model.bin" >> .gitignore
git add .gitignore
git commit -m "chore: gitignore embedding model weights"
```
