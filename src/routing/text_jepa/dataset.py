"""Domain corpus loader — reads data/final/<domain>/train.jsonl into (text, domain) pairs."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DomainSample:
    text: str
    domain: str


def load_domain_corpus(
    data_dir: Path | str,
    domains: list[str],
    max_per_domain: int,
) -> list[DomainSample]:
    """Load user-content messages from per-domain JSONL files.

    Args:
        data_dir: Root containing `<domain>/train.jsonl` subdirs.
        domains: Domain names to include.
        max_per_domain: Cap on samples per domain (first N lines).

    Returns:
        List of DomainSample, possibly empty if no files found.
    """
    data_dir = Path(data_dir)
    out: list[DomainSample] = []
    for domain in domains:
        path = data_dir / domain / "train.jsonl"
        if not path.exists():
            logger.warning("missing %s", path)
            continue
        with path.open(encoding="utf-8") as f:
            count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = rec.get("messages", [])
                user = next(
                    (m["content"] for m in msgs if m.get("role") == "user"), None
                )
                if not user:
                    continue
                out.append(DomainSample(text=user, domain=domain))
                count += 1
                if count >= max_per_domain:
                    break
    return out
