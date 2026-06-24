"""Scoring helpers for multiple-choice QA evals."""

from __future__ import annotations

import re
from typing import Any


_BOXED_RE = re.compile(r"\\boxed\s*(?:\{([^{}]+)\}|([A-Za-z0-9]))")
_TEXT_RE = re.compile(r"\\text\{([^{}]+)\}")
_PAPER_LOOSE_PATTERNS = [
    re.compile(r"(?:final\s+)?(?:answer|choice|option)\s*(?:is|:|=|would be|should be)?\s*\(?\s*([A-H])\s*\)?", re.I),
    re.compile(r"\b(?:choose|select|pick)\s*(?:option\s*)?\(?\s*([A-H])\s*\)?", re.I),
    re.compile(r"\bso\s+(?:the\s+)?(?:answer|choice)\s*(?:is|:)\s*\(?\s*([A-H])\s*\)?", re.I),
]


def _allowed_labels(extra_info: Any) -> list[str]:
    if isinstance(extra_info, dict):
        labels = extra_info.get("choice_labels", [])
        if labels is None:
            labels = []
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        return [str(label).strip().upper() for label in labels if str(label).strip()]
    return []


def _normalize_label(candidate: str | None, allowed: list[str]) -> str | None:
    if candidate is None:
        return None
    candidate = _TEXT_RE.sub(r"\1", candidate)
    candidate = candidate.strip().upper()
    if len(candidate) >= 2 and candidate[0] == "(" and candidate[-1] == ")":
        candidate = candidate[1:-1].strip()
    if not candidate:
        return None
    if allowed and candidate in allowed:
        return candidate
    if allowed and candidate not in allowed:
        return None
    return candidate


def _normalize_gold(ground_truth: Any, extra_info: Any = None) -> str | None:
    allowed = _allowed_labels(extra_info)
    return _extract_choice_label(str(ground_truth), allowed=allowed)


def _extract_choice_label(text: str, allowed: list[str]) -> str | None:
    boxed_matches = list(_BOXED_RE.finditer(text))
    if boxed_matches:
        match = boxed_matches[-1]
        return _normalize_label(match.group(1) or match.group(2), allowed=allowed)
    direct = _normalize_label(text, allowed=allowed)
    if direct is not None:
        return direct
    return None


def extract_boxed_choice(solution_str: str, extra_info: Any = None) -> str | None:
    allowed = _allowed_labels(extra_info)
    boxed_matches = list(_BOXED_RE.finditer(solution_str))
    if not boxed_matches:
        return None
    match = boxed_matches[-1]
    return _normalize_label(match.group(1) or match.group(2), allowed=allowed)


def _extract_boxed_text(solution_str: str) -> str | None:
    solution_str = str(solution_str)
    idx = solution_str.rfind("boxed")
    if idx < 0:
        return None

    answer = solution_str[idx + len("boxed") :]
    if not answer:
        return None

    if answer[0] == "{":
        depth = 1
        chars = []
        for ch in answer[1:]:
            if ch == "{":
                depth += 1
                chars.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
                chars.append(ch)
            else:
                chars.append(ch)
        if depth != 0:
            return None
        extracted = "".join(chars)
    else:
        extracted = answer.split("$")[0]

    extracted = extracted.strip().rstrip(".").rstrip("/")
    return extracted or None


def _first_letter_a_to_h(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"[A-H]", str(text).upper())
    return match.group(0) if match else None


def extract_loose_choice(solution_str: str, extra_info: Any = None) -> str | None:
    del extra_info

    boxed = _extract_boxed_text(solution_str)
    boxed_letter = _first_letter_a_to_h(boxed)
    if boxed_letter is not None:
        return boxed_letter

    tail = str(solution_str)[-1200:]
    matches: list[tuple[int, str]] = []
    for pattern in _PAPER_LOOSE_PATTERNS:
        matches.extend((match.start(), match.group(1).upper()) for match in pattern.finditer(tail))
    if matches:
        return sorted(matches)[-1][1]

    standalone = list(re.finditer(r"(?<![A-Za-z])([A-H])(?![A-Za-z])", str(solution_str)[-200:]))
    if standalone:
        return standalone[-1].group(1).upper()

    return None


def compute_strict_score(solution_str: str, ground_truth: Any, extra_info: Any = None) -> float:
    gold = _normalize_gold(ground_truth, extra_info=extra_info)
    pred = extract_boxed_choice(solution_str, extra_info=extra_info)
    return 1.0 if gold is not None and pred == gold else 0.0


def compute_loose_score(solution_str: str, ground_truth: Any, extra_info: Any = None) -> float:
    del extra_info
    gold = _first_letter_a_to_h(str(ground_truth).strip())
    pred = extract_loose_choice(solution_str)
    return 1.0 if gold is not None and pred == gold else 0.0


def compute_score(solution_str: str, ground_truth: Any, extra_info: Any = None) -> float:
    return compute_loose_score(solution_str, ground_truth, extra_info=extra_info)
