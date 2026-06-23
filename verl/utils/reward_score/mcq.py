"""Scoring helpers for multiple-choice QA evals."""

from __future__ import annotations

import re
from typing import Any


_BOXED_RE = re.compile(r"\\boxed\s*(?:\{([^{}]+)\}|([A-Za-z0-9]))")
_TEXT_RE = re.compile(r"\\text\{([^{}]+)\}")


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


def extract_loose_choice(solution_str: str, extra_info: Any = None) -> str | None:
    allowed = _allowed_labels(extra_info)
    strict = extract_boxed_choice(solution_str, extra_info=extra_info)
    if strict is not None:
        return strict

    tail = solution_str[-800:]
    upper_tail = _TEXT_RE.sub(r"\1", tail).upper()
    allowed_pat = "".join(allowed) if allowed else "A-J"
    patterns = [
        rf"FINAL ANSWER[^A-Z0-9]{{0,24}}([{allowed_pat}])\b",
        rf"ANSWER[^A-Z0-9]{{0,24}}(?:IS|:)?[^A-Z0-9]{{0,12}}([{allowed_pat}])\b",
        rf"(?:OPTION|CHOICE)[^A-Z0-9]{{0,12}}([{allowed_pat}])\b",
        rf"\(([{allowed_pat}])\)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, upper_tail)
        if matches:
            candidate = _normalize_label(matches[-1], allowed=allowed)
            if candidate is not None:
                return candidate

    standalone = re.findall(rf"\b([{allowed_pat}])\b", upper_tail)
    if standalone:
        candidate = _normalize_label(standalone[-1], allowed=allowed)
        if candidate is not None:
            return candidate

    if isinstance(extra_info, dict):
        choices = extra_info.get("choice_text_by_label") or {}
        normalized_tail = " ".join(upper_tail.split())
        for label, choice_text in choices.items():
            choice_text_norm = " ".join(str(choice_text).upper().split())
            if choice_text_norm and choice_text_norm in normalized_tail:
                return _normalize_label(str(label), allowed=allowed)

    return None


def compute_strict_score(solution_str: str, ground_truth: Any, extra_info: Any = None) -> float:
    gold = _normalize_gold(ground_truth, extra_info=extra_info)
    pred = extract_boxed_choice(solution_str, extra_info=extra_info)
    return 1.0 if gold is not None and pred == gold else 0.0


def compute_loose_score(solution_str: str, ground_truth: Any, extra_info: Any = None) -> float:
    gold = _normalize_gold(ground_truth, extra_info=extra_info)
    pred = extract_loose_choice(solution_str, extra_info=extra_info)
    return 1.0 if gold is not None and pred == gold else 0.0


def compute_score(solution_str: str, ground_truth: Any, extra_info: Any = None) -> float:
    return compute_loose_score(solution_str, ground_truth, extra_info=extra_info)
