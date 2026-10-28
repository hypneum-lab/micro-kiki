"""Corpus preparation and the frozen train / val / test split for NH-81.

Corpus: French Project Gutenberg books listed (id + SHA-256) in
``corpus_manifest.json``. Each book is stripped of the Gutenberg licence
header/footer, the books are concatenated in manifest order and tokenised
with the BSK BPE (``bsk_bpe.json``, 16 000 tokens, SHA-256 in the prereg).

Split rule (preregistered): the token stream is cut into blocks of 4096
tokens; block ``i`` goes to val if ``i % 20 == 18``, to test if
``i % 20 == 19``, to train otherwise (10 % held out, 5 % + 5 %).
"""

from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np

BLOCK = 4096
_START = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
_END = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.I)


def strip_gutenberg(text: str) -> str:
    m = _START.search(text)
    if m:
        text = text[m.end():]
    m = _END.search(text)
    if m:
        text = text[:m.start()]
    return text.strip()


def split_of_block(i: int) -> str:
    r = i % 20
    return "val" if r == 18 else "test" if r == 19 else "train"


def build(corpus_dir: str, bpe_path: str, out_dir: str) -> dict:
    from tokenizers import Tokenizer

    man = json.load(open(os.path.join(corpus_dir, "manifest.json")))
    tk = Tokenizer.from_file(bpe_path)
    ids: list[np.ndarray] = []
    for b in man["books"]:
        raw = open(os.path.join(corpus_dir, "raw", f"{b['id']}.txt"), "rb").read()
        assert hashlib.sha256(raw).hexdigest() == b["sha256"], b["id"]
        text = strip_gutenberg(raw.decode("utf-8", errors="ignore"))
        paras = [p for p in text.split("\n") if p.strip()]
        for e in tk.encode_batch([p + "\n" for p in paras]):
            ids.append(np.asarray(e.ids, dtype=np.int32))
    stream = np.concatenate(ids)
    n_blocks = len(stream) // BLOCK
    parts = {"train": [], "val": [], "test": []}
    for i in range(n_blocks):
        parts[split_of_block(i)].append(stream[i * BLOCK:(i + 1) * BLOCK])
    os.makedirs(out_dir, exist_ok=True)
    info = {"tokens_total": int(n_blocks * BLOCK), "vocab": tk.get_vocab_size()}
    for k, v in parts.items():
        arr = np.concatenate(v)
        arr.tofile(os.path.join(out_dir, f"{k}.i32"))
        info[f"tokens_{k}"] = int(arr.size)
        info[f"sha256_{k}"] = hashlib.sha256(arr.tobytes()).hexdigest()
    json.dump(info, open(os.path.join(out_dir, "split_info.json"), "w"), indent=1)
    return info


def eval_windows(n_tokens: int, seq: int, count: int, seed: int) -> np.ndarray:
    """Fixed window start offsets for held-out evaluation."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_tokens - seq - 1, size=count)


if __name__ == "__main__":
    import sys

    print(json.dumps(build(sys.argv[1], sys.argv[2], sys.argv[3]), indent=1))
