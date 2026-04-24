#!/usr/bin/env python3
"""MultiPL-E multilingual HumanEval port for V4 SOTA adapters.

Loads `nuprl/MultiPL-E` `humaneval-{lang}` split and generates completions
with the requested LoRA adapter. For `py`, we reuse the HumanEval Python
sandbox and compute real pass@1. For any other language we capture
completions verbatim and mark `passed=None` with `status=no_sandbox` so a
Docker-based scorer can re-grade later — we don't shell out to gcc / rustc /
tsc inline because that path is too fragile.

Usage (on Studio):
    python scripts/eval_multipl_e_v4.py \
        --base-model /Users/clems/KIKI-Mac_tunner/models/Qwen3.6-35B-A3B \
        --adapter /Users/clems/KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b-v4-sota/python \
        --lang py \
        --output /tmp/multipl-e-py.json \
        --label python \
        --n 20
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import signal
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("multipl_e_v4")

SUPPORTED_LANGS = (
    "cpp", "rs", "ts", "go", "java", "js", "lua",
    "php", "py", "rb", "sh", "swift", "scala",
)

DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "micro-kiki-evals"

# ---------------------------------------------------------------------------
# Fixture materialization
# ---------------------------------------------------------------------------


def _ensure_fixture(fixture: Path | None, lang: str) -> Path:
    if fixture is not None:
        target = fixture
    else:
        target = DEFAULT_CACHE_ROOT / f"multipl_e_humaneval_{lang}.jsonl"

    if target.exists() and target.stat().st_size > 0:
        logger.info("reusing fixture at %s", target)
        return target

    logger.info("materializing MultiPL-E humaneval-%s fixture at %s", lang, target)
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            f"pip install datasets (needed to build MultiPL-E fixture): {e}"
        ) from e

    target.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("nuprl/MultiPL-E", f"humaneval-{lang}", split="test")
    with target.open("w", encoding="utf-8") as f:
        for row in ds:
            rec = {
                "task_id": row.get("name") or row.get("task_id") or f"{lang}/?",
                "language": row.get("language", lang),
                "prompt": row["prompt"],
                "tests": row.get("tests", ""),
                "stop_tokens": list(row.get("stop_tokens", []) or []),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("wrote %d problems to %s", len(ds), target)
    return target


# ---------------------------------------------------------------------------
# Completion extraction
# ---------------------------------------------------------------------------


def _extract_completion(raw: str, prompt: str, stop_tokens: list[str]) -> str:
    text = raw or ""
    if text.startswith(prompt):
        text = text[len(prompt):]
    if text.lstrip().startswith("```"):
        tail = text.split("\n", 1)[1] if "\n" in text else ""
        text = tail
    # Drop language tag line
    first_nl = text.find("\n")
    if first_nl != -1:
        head = text[:first_nl].strip().lower()
        if head.isalpha() and len(head) <= 12:
            text = text[first_nl + 1:]
    cut = len(text)
    for s in stop_tokens:
        if not s:
            continue
        i = text.find(s)
        if i != -1 and i < cut:
            cut = i
    # Always stop at closing fence to be safe
    for s in ("\n```", "```"):
        i = text.find(s)
        if i != -1 and i < cut:
            cut = i
    return text[:cut]


# ---------------------------------------------------------------------------
# Python sandbox (only used for lang == "py")
# ---------------------------------------------------------------------------


def _py_worker(prompt: str, completion: str, tests: str, q) -> None:
    try:
        signal.alarm(8)
        program = prompt + completion + "\n" + tests + "\n"
        ns: dict[str, Any] = {"__name__": "__test__"}
        exec(compile(program, "<multipl_e_py>", "exec"), ns)
        q.put(("ok", ""))
    except BaseException as e:  # noqa: BLE001
        q.put(("fail", f"{type(e).__name__}: {e}"))
    finally:
        try:
            signal.alarm(0)
        except Exception:
            pass


def _py_run_tests(prompt: str, completion: str, tests: str) -> tuple[bool, str]:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_py_worker, args=(prompt, completion, tests, q))
    p.start()
    p.join(timeout=15)
    if p.is_alive():
        p.terminate()
        p.join(1)
        return False, "timeout"
    if q.empty():
        return False, "no_result"
    status, err = q.get()
    return status == "ok", err


# ---------------------------------------------------------------------------
# MLX loader (lazy)
# ---------------------------------------------------------------------------


def _load_mlx(base_model: Path, adapter: Path | None, max_tokens: int):
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    logger.info("loading base model from %s", base_model)
    t0 = time.monotonic()
    model, tokenizer = load(str(base_model))
    logger.info("base loaded in %.1fs", time.monotonic() - t0)

    if adapter is not None:
        from mlx_lm.tuner.utils import load_adapters

        t1 = time.monotonic()
        logger.info("applying adapter: %s", adapter)
        load_adapters(model, str(adapter))
        logger.info("adapter applied in %.1fs", time.monotonic() - t1)

    sampler = make_sampler(temp=0.0)

    def _gen(prompt: str) -> str:
        return generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )

    return _gen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True, type=Path)
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument(
        "--lang",
        required=True,
        choices=SUPPORTED_LANGS,
        help="Target language (MultiPL-E humaneval-{lang} config).",
    )
    p.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="JSONL fixture. If omitted, cached per-lang under "
        f"{DEFAULT_CACHE_ROOT}.",
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--label", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    fixture = _ensure_fixture(args.fixture, args.lang)

    problems: list[dict[str, Any]] = []
    with fixture.open() as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    problems = problems[: args.n]
    logger.info(
        "loaded %d problems from %s (lang=%s)", len(problems), fixture, args.lang
    )

    has_sandbox = args.lang == "py"
    out: dict[str, Any] = {
        "label": args.label,
        "language": args.lang,
        "base_model": str(args.base_model),
        "adapter": str(args.adapter) if args.adapter else None,
        "n_problems": len(problems),
        "max_tokens": args.max_tokens,
        "completions_only": not has_sandbox,
        "per_problem": [],
    }

    if args.dry_run:
        logger.warning("dry run — skipping mlx_lm")
        out["pass@1"] = None
        out["status"] = "dry_run"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        logger.info("dry-run wrote %s (N=%d)", args.output, len(problems))
        return 0

    try:
        gen = _load_mlx(args.base_model, args.adapter, args.max_tokens)
    except ImportError as e:
        logger.error("mlx_lm unavailable: %s", e)
        out["status"] = "mlx_unavailable"
        out["error"] = str(e)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        return 2

    passed = 0
    total_gen_s = 0.0
    for i, prob in enumerate(problems, 1):
        prompt = prob["prompt"]
        stop_tokens = prob.get("stop_tokens") or []
        t_g = time.monotonic()
        try:
            raw = gen(prompt)
        except Exception as e:  # noqa: BLE001
            logger.warning("gen failed for %s: %s", prob["task_id"], e)
            raw = ""
        gen_s = time.monotonic() - t_g
        total_gen_s += gen_s
        completion = _extract_completion(raw, prompt, stop_tokens)

        if has_sandbox:
            ok, err = _py_run_tests(prompt, completion, prob.get("tests", ""))
            status = "ok" if ok else "fail"
            if ok:
                passed += 1
            logger.info(
                "[%d/%d] %s: %s (%.1fs)",
                i,
                len(problems),
                prob["task_id"],
                "PASS" if ok else "FAIL",
                gen_s,
            )
            out["per_problem"].append(
                {
                    "task_id": prob["task_id"],
                    "passed": ok,
                    "status": status,
                    "gen_s": round(gen_s, 2),
                    "completion": completion[:1200],
                    "error": err[:200] if err else "",
                }
            )
        else:
            logger.info(
                "[%d/%d] %s: CAPTURED (%.1fs, lang=%s)",
                i,
                len(problems),
                prob["task_id"],
                gen_s,
                args.lang,
            )
            out["per_problem"].append(
                {
                    "task_id": prob["task_id"],
                    "passed": None,
                    "status": "no_sandbox",
                    "gen_s": round(gen_s, 2),
                    "completion": completion[:1200],
                    "tests": prob.get("tests", "")[:400],
                }
            )

    if has_sandbox:
        out["pass@1"] = passed / len(problems) if problems else 0.0
    else:
        out["pass@1"] = None
    out["total_gen_s"] = round(total_gen_s, 1)
    out["status"] = "ok"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info(
        "DONE label=%s lang=%s pass@1=%s, total_gen=%.1fs, wrote %s",
        args.label,
        args.lang,
        f"{out['pass@1']:.3f}" if out["pass@1"] is not None else "n/a",
        total_gen_s,
        args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
