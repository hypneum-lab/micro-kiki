#!/usr/bin/env python3
"""Generic domain distillation script.

Usage::

    uv run python scripts/distill_domain.py --domain embedded \
        --teacher-url http://kxkm-ai:8000 \
        --teacher-model devstral-v3

Reads seed prompts from ``data/prompts/<domain>.jsonl``, writes
completed examples to ``data/distilled/<domain>.jsonl``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.distill.generator import GeneratorConfig, generate_examples
from src.distill.teacher_client import GenerateParams, TeacherClient

logger = logging.getLogger(__name__)

DEFAULT_TEACHER_URL = "http://kxkm-ai:8000"
DEFAULT_TEACHER_MODEL = "Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf"
DEFAULT_MAX_EXAMPLES = 2000


class SyncTeacherAdapter:
    """Persistent event loop bridge for sync→async teacher calls."""

    def __init__(self, client: TeacherClient, model: str, params: GenerateParams) -> None:
        self._client = client
        self.model = model
        self._params = params
        self._loop = asyncio.new_event_loop()

    def complete(self, prompt: str, **_params: Any) -> str:
        return self._loop.run_until_complete(
            self._client.generate(prompt, self.model, params=self._params)
        )


def load_seed_prompts(path: Path) -> list[str]:
    if not path.exists():
        logger.error("Seed prompts not found: %s", path)
        sys.exit(1)
    prompts = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Bad JSON line %d in %s", lineno, path)
                continue
            prompt = entry.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                prompts.append(prompt)
    logger.info("Loaded %d seed prompts from %s", len(prompts), path)
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill domain dataset from a teacher LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--domain", required=True, help="Domain name (matches data/prompts/<domain>.jsonl)")
    parser.add_argument("--teacher-url", default=DEFAULT_TEACHER_URL)
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--prompts", type=Path, default=None, help="Override seed prompts path")
    parser.add_argument("--output", type=Path, default=None, help="Override output path")
    parser.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    prompts_path = args.prompts or Path(f"data/prompts/{args.domain}.jsonl")
    output_path = args.output or Path(f"data/distilled/{args.domain}.jsonl")

    prompts = load_seed_prompts(prompts_path)
    if not prompts:
        logger.error("No prompts found. Aborting.")
        sys.exit(1)

    n_per_prompt = max(1, -(-args.max_examples // len(prompts)))
    logger.info("Target %d: %d seeds x %d each = %d", args.max_examples, len(prompts), n_per_prompt, n_per_prompt * len(prompts))

    client = TeacherClient(endpoints={args.teacher_model: args.teacher_url})
    gen_params = GenerateParams(temperature=args.temperature, max_tokens=args.max_tokens, thinking=False)
    teacher = SyncTeacherAdapter(client, args.teacher_model, gen_params)

    config = GeneratorConfig(
        n_per_prompt=n_per_prompt, max_retries=3, retry_backoff_s=1.0,
        domain=args.domain, params=gen_params.to_dict(),
    )

    logger.info("Distilling %s -> %s", args.domain, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = generate_examples(prompts=prompts, teacher=teacher, output_path=output_path, config=config)
    logger.info("Done: generated=%d skipped=%d failed=%d", stats["generated"], stats["skipped"], stats["failed"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
