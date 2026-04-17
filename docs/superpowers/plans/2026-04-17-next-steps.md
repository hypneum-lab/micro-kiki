# Next Steps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire MiniLM embeddings into Aeon live pipeline, copy VQC weights from Studio, update AeonServingHook to use dynamic memory format, and re-run POC with real embeddings.

**Architecture:** The MiniLM model is already trained at `models/niche-embeddings/` on Studio. We copy it locally, wire it into AeonServingHook (the production-facing memory hook), update the memory format to match the proven POC pattern, and validate with tests + POC run.

**Tech Stack:** sentence-transformers, numpy, pytest, SSH (Studio)

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `src/serving/aeon_hook.py` | Production memory hook (pre/post inference) | Modify: dynamic budget + structured format |
| `tests/serving/test_aeon_hook.py` | Tests for AeonServingHook | Modify: update for new format |
| `scripts/poc_pipeline.py` | POC v1 memory wiring | Modify: same embed_fn pattern as v2 |

---

### Task 1: Copy VQC weights from Studio

**Files:**
- No code changes — file copy only

- [ ] **Step 1: Check VQC training status**

```bash
ssh studio "strings ~/micro-kiki/outputs/vqc-8q-training.log | grep -E 'Epoch|Training done|Early stopping' | tail -5"
```

If training is done, proceed. If still running, skip this task and come back later.

- [ ] **Step 2: Copy weights**

```bash
scp studio:~/micro-kiki/outputs/vqc-weights.npz outputs/vqc-8q-weights.npz
scp studio:~/micro-kiki/outputs/vqc-pca.npz outputs/vqc-8q-pca.npz
```

- [ ] **Step 3: Verify weights load**

```bash
python3 -c "
import numpy as np
d = np.load('outputs/vqc-8q-weights.npz')
print('weights:', d['weights'].shape)
print('linear_w:', d['linear_w'].shape)
print('linear_b:', d['linear_b'].shape)
"
```

Expected: `weights: (6, 8, 3)`, `linear_w: (8, 11)`, `linear_b: (11,)`

- [ ] **Step 4: Commit**

```bash
git add outputs/vqc-8q-weights.npz outputs/vqc-8q-pca.npz
git commit -m "feat(vqc): add 8-qubit trained weights"
```

---

### Task 2: Update AeonServingHook with dynamic memory format

**Files:**
- Modify: `src/serving/aeon_hook.py`
- Modify: `tests/serving/test_aeon_hook.py` (if exists)

- [ ] **Step 1: Write failing test**

Check if `tests/serving/test_aeon_hook.py` exists. If not, create it. Add:

```python
"""Tests for AeonServingHook dynamic memory format."""
from __future__ import annotations

import hashlib
import numpy as np
import pytest
from unittest.mock import MagicMock
from src.memory.aeon import AeonPalace
from src.memory.trace import Episode
from src.serving.aeon_hook import AeonServingHook
from datetime import datetime


def _mock_embed(dim: int = 64):
    def fn(text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec = rng.randn(dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)
    return fn


class TestAeonServingHookFormat:
    def test_pre_inference_uses_structured_format(self):
        palace = AeonPalace(dim=64, embed_fn=_mock_embed(64))
        palace.write("User: What is a buck converter?\nAssistant: A buck converter steps down voltage.",
                     domain="power")
        hook = AeonServingHook(palace)
        result = hook.pre_inference("Design a boost converter")
        assert "### Previous conversation context:" in result
        assert "### Current question:" in result
        assert "Design a boost converter" in result

    def test_pre_inference_no_memories_returns_original(self):
        palace = AeonPalace(dim=64, embed_fn=_mock_embed(64))
        hook = AeonServingHook(palace)
        result = hook.pre_inference("Hello world")
        assert result == "Hello world"

    def test_post_inference_stores_full_response(self):
        palace = AeonPalace(dim=64, embed_fn=_mock_embed(64))
        hook = AeonServingHook(palace)
        hook.post_inference(
            prompt="What is SPICE?",
            response="SPICE is a circuit simulator" * 100,
            domain="spice",
            turn_id="t1",
        )
        episodes = palace.recall("SPICE simulator", top_k=1)
        assert len(episodes) == 1
        assert "User: What is SPICE?" in episodes[0].content
        assert len(episodes[0].content) > 500
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/serving/test_aeon_hook.py::TestAeonServingHookFormat -v`
Expected: FAIL — `### Previous conversation context:` not in output

- [ ] **Step 3: Update aeon_hook.py**

Replace `src/serving/aeon_hook.py` content:

```python
"""Aeon integration hook for the serving pipeline.

Prepends recalled memories to prompts and writes new memories post-inference.
Uses dynamic memory budget and structured format matching POC v2.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.aeon import AeonPalace

logger = logging.getLogger(__name__)

# Total chars budget for memory context injection
MEMORY_BUDGET = 3000


class AeonServingHook:
    """Wraps AeonPalace for pre/post inference memory injection."""

    def __init__(self, palace: AeonPalace) -> None:
        self._palace = palace

    def pre_inference(self, prompt: str, top_k: int = 8) -> str:
        """Recall memories and prepend them to the prompt.

        Uses dynamic budget: MEMORY_BUDGET chars split evenly across
        recalled episodes.  Format uses markdown headers so the LLM
        recognises the memories as prior conversation context.
        """
        try:
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

    def post_inference(
        self,
        prompt: str,
        response: str,
        domain: str,
        turn_id: str,
    ) -> None:
        """Write the full interaction to Aeon memory."""
        content = f"User: {prompt}\nAssistant: {response}"
        try:
            self._palace.write(
                content=content,
                domain=domain,
                source=turn_id,
            )
        except Exception:
            logger.warning("Aeon write failed for turn %s", turn_id, exc_info=True)
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/serving/test_aeon_hook.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: 700+ passed

- [ ] **Step 6: Commit**

```bash
git add src/serving/aeon_hook.py tests/serving/test_aeon_hook.py
git commit -m "feat(aeon-hook): dynamic budget + full store"
```

---

### Task 3: Copy MiniLM model from Studio

**Files:**
- No code changes — file copy + gitignore

- [ ] **Step 1: Copy model from Studio**

```bash
mkdir -p models/niche-embeddings
scp -r studio:~/micro-kiki/models/niche-embeddings/ models/niche-embeddings/
```

- [ ] **Step 2: Verify model loads**

```bash
python3 -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('models/niche-embeddings')
print('dim:', m.get_sentence_embedding_dimension())
e = m.encode('buck converter SPICE netlist')
print('embedding shape:', e.shape)
"
```

Expected: `dim: 384`, `embedding shape: (384,)`

- [ ] **Step 3: Ensure weights are gitignored**

Check `.gitignore` for `models/niche-embeddings/*.safetensors`. Add if missing:

```bash
echo "models/niche-embeddings/model.safetensors" >> .gitignore
echo "models/niche-embeddings/pytorch_model.bin" >> .gitignore
git add .gitignore
git commit -m "chore: gitignore embedding weights"
```

---

### Task 4: Run POC v2 with real MiniLM embeddings on Studio

**Files:**
- No code changes — execution + validation

- [ ] **Step 1: Sync POC v2 to Studio**

```bash
scp scripts/poc_pipeline_v2.py studio:~/micro-kiki/scripts/poc_pipeline_v2.py
scp src/serving/aeon_hook.py studio:~/micro-kiki/src/serving/aeon_hook.py
scp src/memory/aeon.py studio:~/micro-kiki/src/memory/aeon.py
```

- [ ] **Step 2: Run multi-turn scenario on Studio**

```bash
ssh studio "cd ~/micro-kiki && nohup .venv/bin/python3 scripts/poc_pipeline_v2.py --scenario multi-turn > outputs/poc-v2-real-embeddings.log 2>&1 & echo PID=\$!"
```

- [ ] **Step 3: Wait and check results**

```bash
ssh studio "grep -E 'trained embeddings|Turn 4|inductor|Assistant:' ~/micro-kiki/outputs/poc-v2-real-embeddings.log"
```

Expected: `Memory: Atlas vector (trained embeddings, dim=384)` and Turn 4 successfully recalling inductor values.

- [ ] **Step 4: Compare hash vs trained embeddings**

Check if real embeddings improve recall quality vs hash (should see better episode ordering).
