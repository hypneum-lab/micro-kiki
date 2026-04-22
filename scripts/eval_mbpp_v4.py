#!/usr/bin/env python3
"""MBPP Python coding eval for V4 SOTA adapters.

Parallel benchmark to HumanEval. Loads `google-research-datasets/mbpp` test
split, prompts the model with the problem text plus the three assertion tests,
then executes the completion in an isolated subprocess with SIGALRM timeout.

Usage (on Studio):
    python scripts/eval_mbpp_v4.py \
        --base-model /Users/clems/KIKI-Mac_tunner/models/Qwen3.6-35B-A3B \
        --adapter /Users/clems/KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b-v4-sota/python \
        --output /tmp/mbpp-python.json \
        --label python \
        --n 50
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
logger = logging.getLogger("mbpp_v4")

DEFAULT_CACHE = Path.home() / ".cache" / "micro-kiki-evals" / "mbpp_test.jsonl"

# ---------------------------------------------------------------------------
# Fixture materialization
# ---------------------------------------------------------------------------


def _ensure_fixture(fixture: Path | None) -> Path:
    target = fixture if fixture is not None else DEFAULT_CACHE
    if target.exists() and target.stat().st_size > 0:
        logger.info("reusing fixture at %s", target)
        return target

    logger.info("materializing MBPP fixture at %s", target)
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            f"pip install datasets (needed to build MBPP fixture): {e}"
        ) from e

    target.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("google-research-datasets/mbpp", split="test")
    with target.open("w", encoding="utf-8") as f:
        for row in ds:
            rec = {
                "task_id": f"mbpp/{row['task_id']}",
                "text": row["text"],
                "code": row["code"],
                "test_list": list(row["test_list"]),
                "test_setup_code": row.get("test_setup_code", "") or "",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("wrote %d problems to %s", len(ds), target)
    return target


# ---------------------------------------------------------------------------
# Prompt & completion handling
# ---------------------------------------------------------------------------

STOP_SEQUENCES = (
    "\ndef test_",
    "\nassert ",
    "\nclass Test",
    "\nif __name__",
    "\nprint(",
    "\n```",
    "```",
)


def _build_prompt(problem: dict[str, Any]) -> str:
    tests = "\n".join(problem["test_list"][:3])
    return (
        f'"""{problem["text"]}\n'
        f"Your code should pass these tests:\n"
        f"{tests}\n"
        f'"""\n'
    )


def _extract_completion(raw: str, prompt: str) -> str:
    text = raw or ""
    if text.startswith(prompt):
        text = text[len(prompt):]
    if text.lstrip().startswith("```"):
        tail = text.split("\n", 1)[1] if "\n" in text else ""
        text = tail
        if text.lstrip().startswith("python\n"):
            text = text.lstrip()[len("python\n"):]
    cut = len(text)
    for s in STOP_SEQUENCES:
        i = text.find(s)
        if i != -1 and i < cut:
            cut = i
    return text[:cut]


# ---------------------------------------------------------------------------
# Sandbox runner
# ---------------------------------------------------------------------------


def _worker(program: str, q) -> None:
    try:
        signal.alarm(8)
        ns: dict[str, Any] = {"__name__": "__test__"}
        exec(compile(program, "<mbpp>", "exec"), ns)
        q.put(("ok", ""))
    except BaseException as e:  # noqa: BLE001 - sandbox
        q.put(("fail", f"{type(e).__name__}: {e}"))
    finally:
        try:
            signal.alarm(0)
        except Exception:
            pass


def _run_tests(
    setup: str, completion: str, test_list: list[str]
) -> tuple[bool, str]:
    program_parts: list[str] = []
    if setup:
        program_parts.append(setup)
    program_parts.append(completion)
    program_parts.extend(test_list)
    program = "\n".join(program_parts) + "\n"

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_worker, args=(program, q))
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
        "--fixture",
        type=Path,
        default=None,
        help=f"JSONL fixture. If omitted, auto-downloaded to {DEFAULT_CACHE}.",
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--label", required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    fixture = _ensure_fixture(args.fixture)

    problems: list[dict[str, Any]] = []
    with fixture.open() as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    problems = problems[: args.n]
    logger.info("loaded %d problems from %s", len(problems), fixture)

    out: dict[str, Any] = {
        "label": args.label,
        "base_model": str(args.base_model),
        "adapter": str(args.adapter) if args.adapter else None,
        "n_problems": len(problems),
        "max_tokens": args.max_tokens,
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
        prompt = _build_prompt(prob)
        t_g = time.monotonic()
        try:
            raw = gen(prompt)
        except Exception as e:  # noqa: BLE001
            logger.warning("gen failed for %s: %s", prob["task_id"], e)
            raw = ""
        gen_s = time.monotonic() - t_g
        total_gen_s += gen_s
        completion = _extract_completion(raw, prompt)
        ok, err = _run_tests(
            prob.get("test_setup_code", ""), completion, prob["test_list"]
        )
        logger.info(
            "[%d/%d] %s: %s (%.1fs, err=%s)",
            i,
            len(problems),
            prob["task_id"],
            "PASS" if ok else "FAIL",
            gen_s,
            (err[:60] + "...") if err and len(err) > 60 else err,
        )
        if ok:
            passed += 1
        out["per_problem"].append(
            {
                "task_id": prob["task_id"],
                "passed": ok,
                "gen_s": round(gen_s, 2),
                "completion": completion[:800],
                "error": err[:200] if err else "",
            }
        )

    out["pass@1"] = passed / len(problems) if problems else 0.0
    out["total_gen_s"] = round(total_gen_s, 1)
    out["status"] = "ok"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info(
        "DONE label=%s pass@1=%.3f (%d/%d), total_gen=%.1fs, wrote %s",
        args.label,
        out["pass@1"],
        passed,
        len(problems),
        total_gen_s,
        args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
