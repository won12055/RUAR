"""Scoring helpers for multiple-choice QA evals."""

from __future__ import annotations

import re
from typing import Any


_BOXED_RE = re.compile(r"\\boxed\s*(?:\{([^{}]+)\}|([A-Za-z0-9]))")
_TEXT_RE = re.compile(r"\\text\{([^{}]+)\}")
_SPECIAL_TOKEN_RE = re.compile(r"<[|｜][^|｜>]+[|｜]>")
_DEFAULT_LOOSE_LABELS = list("ABCDEFGH")


def _allowed_labels(extra_info: Any) -> list[str]:
    if isinstance(extra_info, dict):
        labels = extra_info.get("choice_labels", [])
        if labels is None:
            labels = []
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        return [str(label).strip().upper() for label in labels if str(label).strip()]
    return []


def _active_labels(extra_info: Any) -> list[str]:
    return _allowed_labels(extra_info) or list(_DEFAULT_LOOSE_LABELS)


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


def _label_pattern(labels: list[str]) -> str:
    return "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))


def _paper_loose_patterns(labels: list[str]) -> list[re.Pattern]:
    label_pat = _label_pattern(labels)
    return [
        re.compile(
            rf"(?:final\s+)?(?:answer|choice|option)\s*(?:is|:|=|would be|should be)?\s*"
            rf"(?:option\s*)?\(?\s*({label_pat})\s*\)?",
            re.I,
        ),
        re.compile(rf"\b(?:choose|select|pick)\s*(?:option\s*)?\(?\s*({label_pat})\s*\)?", re.I),
        re.compile(
            rf"\bso\s+(?:the\s+)?(?:answer|choice)\s*(?:is|:)\s*(?:option\s*)?\(?\s*({label_pat})\s*\)?",
            re.I,
        ),
    ]


def _strip_special_tokens(text: str) -> str:
    return _SPECIAL_TOKEN_RE.sub("", text)


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
    return _normalize_label(_extract_boxed_text(solution_str), allowed=allowed)


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
    text = _TEXT_RE.sub(r"\1", str(text))
    match = re.search(r"[A-H]", str(text).upper())
    return match.group(0) if match else None


def extract_loose_choice(solution_str: str, extra_info: Any = None) -> str | None:
    allowed = _allowed_labels(extra_info)
    labels = _active_labels(extra_info)
    text = _strip_special_tokens(str(solution_str))

    boxed = extract_boxed_choice(text, extra_info=extra_info)
    if boxed is not None:
        return boxed

    tail = text[-1200:]
    matches: list[tuple[int, str]] = []
    for pattern in _paper_loose_patterns(labels):
        for match in pattern.finditer(tail):
            candidate = _normalize_label(match.group(1), allowed=allowed)
            if candidate is not None:
                matches.append((match.start(), candidate))
    if matches:
        return sorted(matches)[-1][1]

    label_pat = _label_pattern(labels)
    standalone = list(re.finditer(rf"(?<![A-Za-z0-9])({label_pat})(?![A-Za-z0-9])", text[-200:], re.I))
    if standalone:
        candidate = _normalize_label(standalone[-1].group(1), allowed=allowed)
        if candidate is not None:
            return candidate

    if isinstance(extra_info, dict):
        choices = extra_info.get("choice_text_by_label") or {}
        normalized_tail = " ".join(tail.upper().split())
        for label, choice_text in choices.items():
            choice_text_norm = " ".join(str(choice_text).upper().split())
            if choice_text_norm and choice_text_norm in normalized_tail:
                candidate = _normalize_label(str(label), allowed=allowed)
                if candidate is not None:
                    return candidate

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
