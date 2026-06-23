"""Loose letter scoring for held-out multiple-choice evaluation."""

from __future__ import annotations

import re
from typing import Any

from .boxed_math import extract_boxed_answer


_RELAXED_PATTERNS = [
    re.compile(r"(?:final\s+)?(?:answer|choice|option)\s*(?:is|:|=|would be|should be)?\s*\(?\s*([A-H])\s*\)?", re.I),
    re.compile(r"\b(?:choose|select|pick)\s*(?:option\s*)?\(?\s*([A-H])\s*\)?", re.I),
    re.compile(r"\bso\s+(?:the\s+)?(?:answer|choice)\s*(?:is|:)\s*\(?\s*([A-H])\s*\)?", re.I),
]


def _allowed_labels(extra_info: Any) -> set[str]:
    del extra_info
    return set("ABCDEFGH")


def _first_allowed_letter(text: Any, allowed: set[str]) -> str | None:
    for char in str(text).upper():
        if char in allowed:
            return char
    return None


def relaxed_letter(response: str, extra_info: Any = None) -> str | None:
    """Extract the paper-facing MCQ letter from a model response.

    Use the final boxed answer when present; otherwise recover the final
    answer/choice/option letter from the response tail.
    """
    allowed = _allowed_labels(extra_info)
    boxed = extract_boxed_answer(response)
    if boxed:
        pred = _first_allowed_letter(boxed, allowed)
        if pred is not None:
            return pred

    tail = str(response)[-1200:]
    matches: list[tuple[int, str]] = []
    for pattern in _RELAXED_PATTERNS:
        matches.extend(
            (match.start(), match.group(1).upper())
            for match in pattern.finditer(tail)
            if match.group(1).upper() in allowed
        )
    if matches:
        return sorted(matches)[-1][1]

    standalone = list(re.finditer(r"(?<![A-Za-z])([A-H])(?![A-Za-z])", str(response)[-200:]))
    for match in reversed(standalone):
        letter = match.group(1).upper()
        if letter in allowed:
            return letter
    return None


def compute_score(solution_str: str, ground_truth: Any, extra_info: Any = None) -> float:
    allowed = _allowed_labels(extra_info)
    gold = _first_allowed_letter(ground_truth, allowed)
    pred = relaxed_letter(solution_str, extra_info=extra_info)
    return 1.0 if gold is not None and pred == gold else 0.0
