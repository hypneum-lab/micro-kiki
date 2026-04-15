"""Cross-domain deduplication via MinHash + LSH.

Given a directory of per-domain JSONL files (one ``{domain}.jsonl`` per
file), this module identifies near-duplicate examples that appear across
domains and emits a disjoint partition: each offending example is
re-assigned to the single domain with the highest affinity (measured as
the maximum Jaccard similarity between that example and any other
example in each candidate domain), and removed from the others.

The goal is to enforce the project guardrail that per-domain datasets
stay disjoint (see CLAUDE.md) before per-stack training.

Implementation notes
--------------------
* Pure-Python MinHash (``num_perm = 128`` default) using ``hashlib.blake2b``
  seeded on the permutation index — no ``datasketch`` dependency required,
  which keeps the data pipeline self-contained and deterministic across
  Python builds.
* The LSH banding groups the 128 hashes into ``bands = 32`` bands of
  ``rows = 4`` each, which tunes the detection to Jaccard ~0.74 (the
  classic s ~ (1/b)^(1/r) threshold) — a reasonable default for spotting
  near-duplicate prompts/completions that differ by formatting or a few
  tokens but are semantically identical.
* Shingles are character 5-grams of the canonicalized prompt — we use the
  prompt (not the completion) because the teacher will produce distinct
  completions even for identical prompts, and the project treats a prompt
  as the unique identity of a training example.
* Ties on max-affinity are broken by the domain name (lexicographic) to
  guarantee a deterministic partition regardless of file iteration order.

CLI
---
::

    python -m src.distill.dedup --input data/raw/ --output data/dedup/

writes ``{domain}.jsonl`` into ``--output`` with cross-domain duplicates
removed, plus ``_dedup_report.json`` summarizing how many rows moved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MinHash
# ---------------------------------------------------------------------------


DEFAULT_NUM_PERM = 128
DEFAULT_BANDS = 32
DEFAULT_ROWS = 4  # bands * rows == num_perm
DEFAULT_SHINGLE_SIZE = 5

_MAX_HASH = (1 << 64) - 1


def _canonicalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip — for shingle stability."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def shingles(text: str, k: int = DEFAULT_SHINGLE_SIZE) -> set[str]:
    """Return the set of character k-shingles of ``text``.

    Uses canonicalized text so minor whitespace/casing differences don't
    affect the signature. If the text is shorter than ``k`` we fall back
    to returning the whole canonicalized string as a single shingle so
    we never emit an empty set (which would make Jaccard undefined).
    """

    s = _canonicalize(text)
    if len(s) < k:
        return {s} if s else set()
    return {s[i : i + k] for i in range(len(s) - k + 1)}


def _hash_shingle(shingle: str, seed: int) -> int:
    """64-bit hash of ``shingle`` keyed by ``seed``.

    Blake2b gives us seeded hashes without pulling ``mmh3``; we slice the
    first 8 bytes as a uint64.
    """

    key = seed.to_bytes(8, "little", signed=False)
    h = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8, key=key)
    (val,) = struct.unpack("<Q", h.digest())
    return val


def minhash(
    shingle_set: Iterable[str], num_perm: int = DEFAULT_NUM_PERM
) -> list[int]:
    """Return a MinHash signature of length ``num_perm``.

    Empty input yields an all-``_MAX_HASH`` signature which will have
    zero Jaccard overlap with any real signature.
    """

    sig = [_MAX_HASH] * num_perm
    shingle_list = list(shingle_set)
    if not shingle_list:
        return sig
    for i in range(num_perm):
        sig[i] = min(_hash_shingle(sh, i) for sh in shingle_list)
    return sig


def jaccard_estimate(sig_a: list[int], sig_b: list[int]) -> float:
    """Jaccard similarity estimate from two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signature length mismatch")
    if not sig_a:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


# ---------------------------------------------------------------------------
# LSH index
# ---------------------------------------------------------------------------


@dataclass
class LSHIndex:
    """Banding-based LSH over MinHash signatures.

    Candidate pairs are any two keys that collide in at least one band.
    """

    bands: int = DEFAULT_BANDS
    rows: int = DEFAULT_ROWS
    _buckets: list[dict[bytes, list[str]]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._buckets is None:
            self._buckets = [defaultdict(list) for _ in range(self.bands)]

    @property
    def num_perm(self) -> int:
        return self.bands * self.rows

    def insert(self, key: str, signature: list[int]) -> None:
        if len(signature) != self.num_perm:
            raise ValueError(
                f"signature length {len(signature)} != bands*rows "
                f"{self.num_perm}"
            )
        for b in range(self.bands):
            chunk = signature[b * self.rows : (b + 1) * self.rows]
            band_key = b"".join(h.to_bytes(8, "little") for h in chunk)
            self._buckets[b][band_key].append(key)

    def candidate_pairs(self) -> set[tuple[str, str]]:
        """Return unique (key_a, key_b) pairs that collided in any band."""
        pairs: set[tuple[str, str]] = set()
        for band in self._buckets:
            for bucket_keys in band.values():
                if len(bucket_keys) < 2:
                    continue
                # Sort for deterministic pair ordering (a < b).
                keys = sorted(set(bucket_keys))
                for i in range(len(keys)):
                    for j in range(i + 1, len(keys)):
                        pairs.add((keys[i], keys[j]))
        return pairs


# ---------------------------------------------------------------------------
# Dedup driver
# ---------------------------------------------------------------------------


@dataclass
class DedupConfig:
    num_perm: int = DEFAULT_NUM_PERM
    bands: int = DEFAULT_BANDS
    rows: int = DEFAULT_ROWS
    shingle_size: int = DEFAULT_SHINGLE_SIZE
    # Minimum estimated Jaccard to consider two rows duplicates.
    similarity_threshold: float = 0.7

    def __post_init__(self) -> None:
        if self.bands * self.rows != self.num_perm:
            raise ValueError(
                "bands * rows must equal num_perm "
                f"(got {self.bands}*{self.rows} != {self.num_perm})"
            )


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("skipping malformed line in %s", path)


def _row_text(row: dict) -> str:
    """The text we shingle for dedup. Currently: the prompt field."""
    return str(row.get("prompt", ""))


def dedup_domains(
    rows_by_domain: dict[str, list[dict]],
    config: DedupConfig | None = None,
) -> tuple[dict[str, list[dict]], dict]:
    """Produce a cross-domain disjoint partition.

    Parameters
    ----------
    rows_by_domain:
        Mapping ``domain_name -> list_of_row_dicts``. Each row must have a
        ``prompt`` field; other fields are preserved.
    config:
        Optional :class:`DedupConfig`. Defaults to the module defaults.

    Returns
    -------
    (partitioned, report)
        ``partitioned`` is the same shape as ``rows_by_domain`` but with
        cross-domain duplicates kept only in their highest-affinity
        domain. ``report`` is a dict with per-domain counts and the list
        of resolved duplicate groups (for auditing).
    """

    cfg = config or DedupConfig()

    # ------------------------------------------------------------------
    # 1. Build signatures and LSH index over (domain, idx) keys.
    # ------------------------------------------------------------------
    index = LSHIndex(bands=cfg.bands, rows=cfg.rows)
    sigs: dict[str, list[int]] = {}
    shingle_sets: dict[str, set[str]] = {}
    key_to_meta: dict[str, tuple[str, int]] = {}

    for domain, rows in rows_by_domain.items():
        for i, row in enumerate(rows):
            key = f"{domain}::{i}"
            sh = shingles(_row_text(row), cfg.shingle_size)
            shingle_sets[key] = sh
            sig = minhash(sh, cfg.num_perm)
            sigs[key] = sig
            key_to_meta[key] = (domain, i)
            index.insert(key, sig)

    # ------------------------------------------------------------------
    # 2. Collect *cross-domain* duplicate groups.
    # ------------------------------------------------------------------
    # Union-find over keys; union pairs that pass the jaccard threshold.
    parent: dict[str, str] = {k: k for k in sigs}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in index.candidate_pairs():
        da, _ = key_to_meta[a]
        db, _ = key_to_meta[b]
        if da == db:
            # Same-domain dedup is out of scope here (each generator already
            # avoids re-submitting a prompt via its SHA hash).
            continue
        if jaccard_estimate(sigs[a], sigs[b]) >= cfg.similarity_threshold:
            union(a, b)

    groups: dict[str, list[str]] = defaultdict(list)
    for k in sigs:
        groups[find(k)].append(k)
    # Only groups touching multiple domains are cross-domain duplicates.
    cross_groups = [
        sorted(members)
        for members in groups.values()
        if len({key_to_meta[m][0] for m in members}) > 1
    ]

    # ------------------------------------------------------------------
    # 3. Resolve each cross-group by highest-affinity domain.
    # ------------------------------------------------------------------
    # Affinity of a member to a candidate domain = max Jaccard between the
    # member's shingle set and any *same-domain* member in that group.
    # Since the members already form a similarity cluster, we can pick the
    # domain whose member has the globally highest incoming affinity sum
    # (ties broken lexicographically on domain).
    keys_to_drop: set[str] = set()
    resolutions: list[dict] = []

    for group in cross_groups:
        per_domain: dict[str, list[str]] = defaultdict(list)
        for k in group:
            per_domain[key_to_meta[k][0]].append(k)

        # Score each domain by total pairwise Jaccard of its members with
        # the rest of the group — larger = stronger affinity.
        def domain_score(domain_keys: list[str]) -> float:
            total = 0.0
            for mk in domain_keys:
                mk_sh = shingle_sets[mk]
                for other in group:
                    if other == mk:
                        continue
                    other_sh = shingle_sets[other]
                    inter = len(mk_sh & other_sh)
                    union_size = len(mk_sh | other_sh) or 1
                    total += inter / union_size
            return total

        scored = sorted(
            per_domain.items(),
            key=lambda kv: (-domain_score(kv[1]), kv[0]),
        )
        winner = scored[0][0]
        # Drop every member not in the winning domain.
        for domain, keys in per_domain.items():
            if domain != winner:
                keys_to_drop.update(keys)
        resolutions.append(
            {
                "winner": winner,
                "members": {
                    d: [key_to_meta[k][1] for k in ks]
                    for d, ks in per_domain.items()
                },
            }
        )

    # ------------------------------------------------------------------
    # 4. Materialize the partitioned output.
    # ------------------------------------------------------------------
    partitioned: dict[str, list[dict]] = {d: [] for d in rows_by_domain}
    for domain, rows in rows_by_domain.items():
        for i, row in enumerate(rows):
            key = f"{domain}::{i}"
            if key in keys_to_drop:
                continue
            partitioned[domain].append(row)

    report = {
        "threshold": cfg.similarity_threshold,
        "num_perm": cfg.num_perm,
        "bands": cfg.bands,
        "rows": cfg.rows,
        "domains": {
            d: {"input": len(rows), "output": len(partitioned[d])}
            for d, rows in rows_by_domain.items()
        },
        "cross_groups": len(cross_groups),
        "dropped": len(keys_to_drop),
        "resolutions": resolutions,
    }
    return partitioned, report


def dedup_directory(
    input_dir: Path, output_dir: Path, config: DedupConfig | None = None
) -> dict:
    """Run :func:`dedup_domains` over every ``*.jsonl`` file in ``input_dir``.

    Writes the disjoint partition to ``output_dir`` and a
    ``_dedup_report.json`` summary. Returns the report dict.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_domain: dict[str, list[dict]] = {}
    for p in sorted(input_dir.glob("*.jsonl")):
        domain = p.stem
        rows_by_domain[domain] = list(_iter_jsonl(p))

    partitioned, report = dedup_domains(rows_by_domain, config)

    for domain, rows in partitioned.items():
        out = output_dir / f"{domain}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    report_path = output_dir / "_dedup_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-domain dedup via MinHash + LSH."
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="dir of per-domain jsonl files"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="dir to write partitioned jsonls + report",
    )
    parser.add_argument("--num-perm", type=int, default=DEFAULT_NUM_PERM)
    parser.add_argument("--bands", type=int, default=DEFAULT_BANDS)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--shingle-size", type=int, default=DEFAULT_SHINGLE_SIZE)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    cfg = DedupConfig(
        num_perm=args.num_perm,
        bands=args.bands,
        rows=args.rows,
        shingle_size=args.shingle_size,
        similarity_threshold=args.threshold,
    )
    report = dedup_directory(args.input, args.output, cfg)
    print(
        json.dumps(
            {
                "dropped": report["dropped"],
                "cross_groups": report["cross_groups"],
                "domains": report["domains"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
