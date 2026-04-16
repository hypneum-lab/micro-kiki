#!/usr/bin/env python3
"""Distill all 32 domains using Qwen3-Coder-480B as local MLX teacher.

Usage:
    # Distill a single domain:
    uv run scripts/distill_with_local_teacher.py --domain chat-fr

    # Distill all domains:
    uv run scripts/distill_with_local_teacher.py --all

    # Dry-run (count prompts only):
    uv run scripts/distill_with_local_teacher.py --all --dry-run

Prerequisites:
    - Qwen3-Coder-480B-A35B MLX server running:
      mlx_lm.server --model ~/models/qwen3-coder/Qwen3-Coder-480B-A35B-Instruct-MLX-4bit --port 8200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from src.distill.teacher_client import TeacherClient
from src.distill.generator import generate_examples

logger = logging.getLogger(__name__)

TEACHER_URL = "http://localhost:8200"
TEACHER_MODEL = "qwen3-coder-480b"
KIKI_DATA = Path.home() / "KIKI-Mac_tunner" / "data" / "micro-kiki"

# Domains and their source data directories
DOMAINS = [
    "chat-fr", "reasoning", "python", "typescript", "cpp", "rust",
    "html-css", "shell", "sql", "yaml-json", "docker", "kicad-dsl",
    "spice", "lua-upy", "embedded", "stm32", "iot", "freecad",
    "platformio", "power", "emc", "dsp", "spice-sim", "electronics",
    "kicad-pcb", "web-frontend", "web-backend", "music-audio",
    "devops", "llm-orch", "math", "security",
]


def load_existing_prompts(domain: str) -> list[str]:
    """Load existing classified prompts from KIKI-Mac_tunner."""
    classified = KIKI_DATA / "classified" / f"{domain}.jsonl"
    if not classified.exists():
        logger.warning("No classified data for %s at %s", domain, classified)
        return []

    prompts = []
    for line in classified.read_text().strip().split("\n"):
        if not line:
            continue
        try:
            entry = json.loads(line)
            prompt = entry.get("prompt", entry.get("instruction", entry.get("input", "")))
            if prompt:
                prompts.append(prompt)
        except json.JSONDecodeError:
            continue

    logger.info("Loaded %d prompts for %s", len(prompts), domain)
    return prompts


async def distill_domain(domain: str, client: TeacherClient, dry_run: bool = False) -> dict:
    """Distill a single domain using the local teacher."""
    prompts = load_existing_prompts(domain)
    if not prompts:
        return {"domain": domain, "status": "no_prompts", "count": 0}

    output_path = Path(f"data/distilled/{domain}.jsonl")

    if dry_run:
        return {"domain": domain, "status": "dry_run", "prompts": len(prompts)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated = await generate_examples(
        prompts=prompts[:3000],  # cap at 3000 per domain
        teacher=client,
        model_name=TEACHER_MODEL,
        domain=domain,
        output_path=output_path,
        n_per_prompt=1,
    )
    return {"domain": domain, "status": "done", "generated": generated}


async def main(domains: list[str], dry_run: bool = False) -> None:
    client = TeacherClient(
        endpoints={TEACHER_MODEL: TEACHER_URL},
        cache_dir="data/teacher_cache",
    )

    results = []
    for domain in domains:
        logger.info("=== Distilling %s ===", domain)
        result = await distill_domain(domain, client, dry_run)
        results.append(result)
        logger.info("Result: %s", result)

    # Summary
    total = sum(r.get("generated", r.get("prompts", 0)) for r in results)
    logger.info("=== Summary: %d domains, %d total examples ===", len(results), total)
    for r in results:
        print(json.dumps(r))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Distill with local Qwen3-Coder-480B teacher")
    parser.add_argument("--domain", help="Single domain to distill")
    parser.add_argument("--all", action="store_true", help="Distill all 32 domains")
    parser.add_argument("--dry-run", action="store_true", help="Count prompts only")
    args = parser.parse_args()

    if args.domain:
        target_domains = [args.domain]
    elif args.all:
        target_domains = DOMAINS
    else:
        parser.print_help()
        sys.exit(1)

    asyncio.run(main(target_domains, args.dry_run))
