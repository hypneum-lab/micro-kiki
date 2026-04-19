"""LLM-as-judge rubric + response parser for C2 downstream eval.

The judge scores a candidate answer 0-5 given question + reference domain.
We build a prompt with a strict rubric and parse an integer score from the response.
"""
from __future__ import annotations

import re

_RUBRIC_TEMPLATE = """You are evaluating an answer to a technical question in the domain of {domain}.

Question: {question}

Answer: {answer}

Rubric (choose one integer 0 to 5):
- 0: completely wrong, irrelevant, or refuses to answer
- 1: fundamentally incorrect but on-topic
- 2: mostly wrong with one correct element
- 3: partially correct, misses key points
- 4: mostly correct, minor omissions
- 5: fully correct, complete, technically precise

Think briefly, then end your response with exactly "Score: <N>" where N is the integer 0-5. Do not wrap N in markdown or quotes."""


def build_rubric_prompt(question: str, answer: str, domain: str) -> str:
    """Return a prompt asking the judge to score the answer 0-5."""
    return _RUBRIC_TEMPLATE.format(
        question=question.strip(),
        answer=answer.strip(),
        domain=domain.strip(),
    )


def parse_score(response: str) -> int | None:
    """Extract the integer score from the judge's response.

    Strategy:
    1. Look for explicit "Score: N" pattern.
    2. Look for "N out of M" pattern (take N).
    3. Grab the last integer in the response.
    Clamps result to [0, 5]. Returns None if no integer is found.
    """
    if not response:
        return None
    # 1. Explicit "Score: N" or "score: N"
    m = re.search(r"[Ss]core\s*:\s*(-?\d+)", response)
    if m:
        return max(0, min(5, int(m.group(1))))
    # 2. "N out of M" pattern
    m = re.search(r"(-?\d+)\s+out\s+of\s+\d+", response)
    if m:
        return max(0, min(5, int(m.group(1))))
    # 3. Last integer fallback
    nums = re.findall(r"-?\d+", response)
    if not nums:
        return None
    return max(0, min(5, int(nums[-1])))
