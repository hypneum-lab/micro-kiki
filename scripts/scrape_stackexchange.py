#!/usr/bin/env python3
"""Scrape Electronics StackExchange Q&A pairs for micro-kiki training data.

Uses the StackExchange API v2.3 (no auth for 300 req/day; pass --api-key for 10000/day).
Outputs JSONL training examples to data/stackexchange/<domain>/train.jsonl.

Usage::

    uv run python scripts/scrape_stackexchange.py --domain embedded
    uv run python scripts/scrape_stackexchange.py --all --max-pages 5
    uv run python scripts/scrape_stackexchange.py --domain stm32 --api-key YOUR_KEY
    uv run python scripts/scrape_stackexchange.py --domain electronics --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "data" / "stackexchange"

API_BASE = "https://api.stackexchange.com/2.3"
SITE = "electronics"

DOMAIN_TAGS: dict[str, list[str]] = {
    "kicad-dsl": ["kicad", "pcb-design", "schematic", "footprint", "pcb-layout"],
    "spice": ["spice", "ltspice", "ngspice", "simulation", "circuit-analysis"],
    "emc": ["emc", "emi", "shielding", "grounding", "esd", "compliance"],
    "stm32": ["stm32", "stm32f4", "stm32h7", "hal", "cubemx", "arm"],
    "embedded": [
        "embedded",
        "microcontroller",
        "firmware",
        "rtos",
        "freertos",
        "interrupt",
        "dma",
    ],
    "freecad": [],  # not on electronics.stackexchange
    "platformio": ["platformio"],
    "power": [
        "power-electronics",
        "smps",
        "buck-converter",
        "boost-converter",
        "mosfet",
        "dc-dc-converter",
    ],
    "dsp": ["dsp", "fft", "filter", "digital-signal-processing", "adc"],
    "electronics": [
        "operational-amplifier",
        "analog",
        "transistors",
        "circuit-design",
        "amplifier",
        "voltage-regulator",
    ],
}

ALL_DOMAINS = list(DOMAIN_TAGS.keys())


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------


class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.result: list[str] = []

    def handle_data(self, data: str) -> None:
        self.result.append(data)

    def get_text(self) -> str:
        return "".join(self.result)


def strip_html(html: str) -> str:
    """Strip HTML tags; convert <code>/<pre> to Markdown."""
    # Convert inline code and code blocks before stripping
    html = re.sub(r"<code>(.*?)</code>", r"`\1`", html, flags=re.DOTALL)
    html = re.sub(r"<pre>(.*?)</pre>", r"```\n\1\n```", html, flags=re.DOTALL)
    stripper = _HTMLStripper()
    stripper.feed(html)
    return stripper.get_text().strip()


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------


def _build_url(tag: str, page: int, api_key: str | None) -> str:
    params = (
        f"order=desc&sort=votes&tagged={tag}&site={SITE}"
        f"&filter=withbody&pagesize=100&page={page}"
    )
    if api_key:
        params += f"&key={api_key}"
    return f"{API_BASE}/questions?{params}"


def _fetch_json(
    client: httpx.Client,
    url: str,
    delay: float,
) -> dict[str, Any]:
    """Fetch a single URL with retry + 429 backoff."""
    backoff = delay * 2
    for attempt in range(3):
        try:
            resp = client.get(url, timeout=30)
            if resp.status_code == 429:
                wait = backoff * (2 ** attempt)
                logger.warning("Rate limited (429). Waiting %.1fs before retry.", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error("HTTP error on attempt %d: %s", attempt + 1, exc)
            if attempt == 2:
                raise
        except httpx.RequestError as exc:
            logger.error("Request error on attempt %d: %s", attempt + 1, exc)
            if attempt == 2:
                raise
    return {}


def _fetch_page(
    client: httpx.Client,
    tag: str,
    page: int,
    api_key: str | None,
    delay: float,
) -> dict[str, Any]:
    """Fetch one page of questions; handles 429 backoff."""
    url = _build_url(tag, page, api_key)
    return _fetch_json(client, url, delay)


def _fetch_answers(
    client: httpx.Client,
    question_ids: list[int],
    api_key: str | None,
    delay: float,
) -> dict[int, list[dict[str, Any]]]:
    """Fetch answers for a batch of question IDs (max 100 per call).

    Returns {question_id: [answer, ...]} with answer bodies included.
    """
    if not question_ids:
        return {}
    ids_str = ";".join(str(qid) for qid in question_ids)
    params = f"order=desc&sort=votes&site={SITE}&filter=withbody&pagesize=100&page=1"
    if api_key:
        params += f"&key={api_key}"
    url = f"{API_BASE}/questions/{ids_str}/answers?{params}"
    data = _fetch_json(client, url, delay)

    result: dict[int, list[dict[str, Any]]] = {}
    for ans in data.get("items", []):
        qid = ans.get("question_id")
        if qid is not None:
            result.setdefault(qid, []).append(ans)
    return result


def _best_answer(answers: list[dict[str, Any]], min_score: int) -> dict[str, Any] | None:
    """Return the accepted answer or the highest-scoring answer above min_score."""
    accepted = [a for a in answers if a.get("is_accepted")]
    if accepted:
        return accepted[0]
    high = [a for a in answers if a.get("score", 0) >= min_score]
    if high:
        return max(high, key=lambda a: a.get("score", 0))
    return None


def _to_training_example(
    question: dict[str, Any],
    answer: dict[str, Any],
    domain: str,
) -> dict[str, Any]:
    title = question.get("title", "").strip()
    q_body = strip_html(question.get("body", ""))
    a_body = strip_html(answer.get("body", ""))

    user_content = f"{title}\n\n{q_body}" if q_body else title

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": a_body},
        ],
        "domain": domain,
        "source": "electronics.stackexchange.com",
        "se_score": question.get("score", 0),
    }


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


def scrape_domain(
    domain: str,
    *,
    max_pages: int = 10,
    min_score: int = 5,
    api_key: str | None = None,
    delay: float = 1.0,
    dry_run: bool = False,
) -> int:
    """Scrape a single domain; returns number of examples written."""
    tags = DOMAIN_TAGS.get(domain, [])
    if not tags:
        logger.warning("Domain '%s' has no StackExchange tags — skipping.", domain)
        return 0

    out_dir = OUTPUT_ROOT / domain
    out_file = out_dir / "train.jsonl"

    # Resume: skip if file already has content
    if out_file.exists() and out_file.stat().st_size > 0:
        existing = sum(1 for _ in out_file.open())
        if existing > 0:
            logger.info(
                "Domain '%s': output already has %d examples, skipping.", domain, existing
            )
            return existing

    if dry_run:
        logger.info("[dry-run] Would scrape domain '%s' with tags: %s", domain, tags)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    seen_ids: set[int] = set()
    examples: list[dict[str, Any]] = []

    with httpx.Client(headers={"Accept-Encoding": "gzip"}) as client:
        for tag in tags:
            logger.info("  tag=%s ...", tag)
            for page in range(1, max_pages + 1):
                data = _fetch_page(client, tag, page, api_key, delay)
                items = data.get("items", [])
                if not items:
                    break

                # Filter eligible questions (unseen, high score, has answers)
                eligible: list[dict[str, Any]] = []
                for q in items:
                    qid = q.get("question_id")
                    if qid in seen_ids:
                        continue
                    if q.get("score", 0) < min_score:
                        continue
                    if q.get("answer_count", 0) == 0:
                        continue
                    eligible.append(q)

                # Fetch answers in batch (up to 100 IDs per API call)
                if eligible:
                    eligible_ids = [q["question_id"] for q in eligible]
                    time.sleep(delay)
                    answers_map = _fetch_answers(client, eligible_ids, api_key, delay)

                    for q in eligible:
                        qid = q["question_id"]
                        answers = answers_map.get(qid, [])
                        best = _best_answer(answers, min_score)
                        if best is None:
                            continue
                        seen_ids.add(qid)
                        examples.append(_to_training_example(q, best, domain))

                has_more = data.get("has_more", False)
                logger.debug(
                    "    page=%d items=%d collected=%d has_more=%s",
                    page, len(items), len(examples), has_more,
                )

                time.sleep(delay)
                if not has_more:
                    break

    logger.info("Domain '%s': %d examples collected.", domain, len(examples))
    if examples:
        with out_file.open("w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info("  Written → %s", out_file)

    return len(examples)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scrape Electronics StackExchange for micro-kiki training data."
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--domain",
        choices=ALL_DOMAINS,
        help="Single domain to scrape.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Scrape all domains.",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=10,
        metavar="N",
        help="Max pages per tag (default: 10, 100 questions/page).",
    )
    p.add_argument(
        "--min-score",
        type=int,
        default=5,
        metavar="N",
        help="Minimum question score (default: 5).",
    )
    p.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help="StackExchange API key (raises limit from 300 to 10000 req/day).",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=1.0,
        metavar="SECS",
        help="Seconds between requests (default: 1.0).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be scraped without making HTTP calls.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    domains = ALL_DOMAINS if args.all else [args.domain]

    total = 0
    for domain in domains:
        count = scrape_domain(
            domain,
            max_pages=args.max_pages,
            min_score=args.min_score,
            api_key=args.api_key,
            delay=args.delay,
            dry_run=args.dry_run,
        )
        total += count

    logger.info("Done. Total examples: %d", total)


if __name__ == "__main__":
    main()
