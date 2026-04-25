#!/usr/bin/env python3
"""
Filter stub/placeholder examples from python training data.

Only removes clear stubs — conservative approach to preserve
legitimate uses of 'pass', 'TODO', etc. in real code.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def is_stub(assistant_text: str) -> tuple[bool, str]:
    """
    Return (is_stub, reason) for the given assistant response.
    Conservative: only flag unambiguous stubs.
    """
    text = assistant_text.strip()

    # Too short to be useful
    if len(text) < 50:
        return True, "too_short"

    # Contains "Your code here" — always a stub marker
    if "your code here" in text.lower():
        return True, "your_code_here"

    # "placeholder" repeated many times — spam filler
    if text.lower().count("placeholder") >= 3:
        return True, "placeholder_spam"

    # Primary content is NotImplementedError / bare pass with no real logic
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    code_lines = [
        l
        for l in lines
        if not l.startswith("#") and not l.startswith('"""') and not l.startswith("'''")
    ]
    if code_lines and all(
        "NotImplementedError" in l
        or l == "pass"
        or l.startswith("def ")
        or l.startswith("class ")
        or l.startswith("@")
        or l.startswith("return")
        for l in code_lines
    ):
        # Only flag if there are no real implementation lines
        has_real_code = any(
            l not in ("pass",)
            and "NotImplementedError" not in l
            and not l.startswith("def ")
            and not l.startswith("class ")
            and not l.startswith("@")
            and not l.startswith("return")
            for l in code_lines
        )
        if not has_real_code and any(
            "NotImplementedError" in l or l == "pass" for l in code_lines
        ):
            return True, "not_implemented"

    return False, ""


def get_assistant_content(example: dict) -> str | None:
    """Extract assistant message content from a chat-format example."""
    messages = example.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            # Handle list-of-parts format
            if isinstance(content, list):
                return " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
    # Fallback for completion-style format
    return example.get("output") or example.get("response")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter stub/placeholder examples from python training JSONL"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSONL file (cleaned)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report stats without writing output",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.output == args.input:
        print("ERROR: output must differ from input (won't overwrite original)", file=sys.stderr)
        sys.exit(1)

    total = 0
    kept = 0
    reasons: Counter = Counter()

    output_lines: list[str] = []

    print(f"Reading {args.input} …")
    with args.input.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            total += 1
            try:
                example = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"  WARN line {lineno}: JSON parse error — {exc}", file=sys.stderr)
                reasons["json_error"] += 1
                continue

            assistant = get_assistant_content(example)
            if assistant is None:
                # No assistant turn — keep it, could be valid prompt-only
                kept += 1
                output_lines.append(raw)
                continue

            stub, reason = is_stub(assistant)
            if stub:
                reasons[reason] += 1
            else:
                kept += 1
                output_lines.append(raw)

    filtered = total - kept
    print(f"\n{'='*55}")
    print(f"  Total examples : {total:>10,}")
    print(f"  Kept           : {kept:>10,}")
    print(f"  Filtered       : {filtered:>10,}  ({100 * filtered / max(total, 1):.2f}%)")
    print(f"{'='*55}")
    print("  Breakdown by reason:")
    for reason, count in reasons.most_common():
        print(f"    {reason:<25} {count:>8,}")
    print(f"{'='*55}\n")

    if args.dry_run:
        print("Dry-run mode — no output written.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {kept:,} examples to {args.output} …")
    with args.output.open("w", encoding="utf-8") as fh:
        for line in output_lines:
            fh.write(line + "\n")
    print("Done.")


if __name__ == "__main__":
    main()
