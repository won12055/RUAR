"""Reflective reasoning step extraction utilities for RUAR.

Detects exact reflection cues (Wait, Alternatively) and returns the
reflective reasoning steps used by Utility-Guided Advantage Rescaling.

Conventions:
    cue_j     : token range of the cue word itself (start_char, end_char)
    span_j    : [cue_j, cue_{j+1}) or [cue_j, </think>)
    before_j  : prefix [0, cue_j_start)
    after_j   : prefix [0, span_j_end)

Cue typing:
    WAIT = 0
    ALT  = 1
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

WAIT = 0
ALT = 1
CUE_NAMES = {WAIT: "WAIT", ALT: "ALT"}
CUE_ALIASES = {
    "wait": WAIT,
    "WAIT": WAIT,
    "Wait": WAIT,
    "alt": ALT,
    "ALT": ALT,
    "alternative": ALT,
    "alternatively": ALT,
    "Alternatively": ALT,
}

# Case-sensitive exact-word cues. Punctuation is intentionally ignored: "Wait,"
# and "Wait!" both match the same cue, while "wait" and "Waiter" do not.
_WAIT_PATTERN = re.compile(r"\b(Wait)\b")
_ALT_PATTERN = re.compile(r"\b(Alternatively)\b")

THINK_START = "<think>"
THINK_END = "</think>"
FINAL_MARKERS = (
    "\n**Final Answer**",
    "**Final Answer**",
    "\nFinal Answer",
    "Final Answer:",
    "\nThe final answer",
    "The final answer",
)
FINAL_ANSWER_PROTECTION_MARKERS = (
    "\n**Final Answer**",
    "**Final Answer**",
)
FINAL_ANSWER_BOXED_FALLBACK_MIN_FRAC = 0.5


@dataclass(frozen=True)
class Cue:
    cue_type: int
    cue_start: int
    cue_end: int


@dataclass(frozen=True)
class ReflectiveStep:
    cue_type: int
    cue_start: int
    cue_end: int
    span_end: int

    @property
    def before_end(self) -> int:
        return self.cue_start

    @property
    def after_end(self) -> int:
        return self.span_end


def _think_block_bounds(text: str) -> Tuple[int, int]:
    """Return (start, end) of the <think>...</think> body.

    If <think> is absent, the body starts at 0 (some templates inject the tag
    only in the prompt). If </think> is absent, the body ends at len(text).
    """
    s = text.find(THINK_START)
    block_start = s + len(THINK_START) if s >= 0 else 0
    e = text.find(THINK_END, block_start)
    if e >= 0:
        block_end = e
    else:
        marker_positions = [text.find(marker, block_start) for marker in FINAL_MARKERS]
        marker_positions = [pos for pos in marker_positions if pos >= 0]
        block_end = min(marker_positions) if marker_positions else len(text)
    return block_start, block_end


def find_final_answer_start(text: str) -> Optional[int]:
    """Return the earliest character where the final answer region starts.

    This intentionally uses only high-precision markers. Generic phrases such
    as "The final answer" often appear inside problem restatements or reasoning,
    so using them here can stop scaling too early.
    """
    candidates: List[int] = []
    block_start, _ = _think_block_bounds(text)
    for marker in FINAL_ANSWER_PROTECTION_MARKERS:
        pos = text.find(marker, block_start)
        if pos >= 0:
            candidates.append(pos)
    think_end = text.find(THINK_END, block_start)
    if think_end >= 0:
        candidates.append(think_end)
    boxed_pos = text.rfind("\\boxed")
    if boxed_pos >= 0 and boxed_pos >= int(len(text) * FINAL_ANSWER_BOXED_FALLBACK_MIN_FRAC):
        candidates.append(boxed_pos)
    return min(candidates) if candidates else None


def _find_cues(text: str, start_limit: int, end_limit: int) -> List[Cue]:
    cues: List[Cue] = []
    for pat, ctype in ((_WAIT_PATTERN, WAIT), (_ALT_PATTERN, ALT)):
        for m in pat.finditer(text, start_limit, end_limit):
            cue_start = m.start(1)
            cues.append(Cue(cue_type=ctype, cue_start=cue_start, cue_end=m.end(1)))
    cues.sort(key=lambda c: c.cue_start)
    return cues


def extract_reflective_steps(
    text: str,
    max_spans: Optional[int] = None,
    max_span_chars: Optional[int] = None,
    cue_types: Optional[Iterable[int | str]] = None,
    span_selection: str = "first",
) -> List[ReflectiveStep]:
    """Extract reflective reasoning steps within the <think> block.

    Args:
        text: Full response text. The <think>...</think> block is auto-detected;
            cues outside the block are ignored.
        max_spans: If set, return only N spans according to span_selection.
        max_span_chars: If set, truncate any span longer than this many chars.
        cue_types: Optional cue allow-list. Accepts ids or strings such as
            "wait", "alternative", or "all".
        span_selection: Which spans to keep when max_spans is set.

    Returns:
        Reflective steps ordered by cue_start.
    """
    start_limit, end_limit = _think_block_bounds(text)
    cues = _find_cues(text, start_limit, end_limit)
    allowed = None
    if cue_types is not None:
        allowed = set()
        for cue_type in cue_types:
            if isinstance(cue_type, str):
                if cue_type.lower() == "all":
                    allowed = {WAIT, ALT}
                    break
                if cue_type not in CUE_ALIASES:
                    raise ValueError(f"Unknown reflection cue type: {cue_type}")
                allowed.add(CUE_ALIASES[cue_type])
            else:
                allowed.add(int(cue_type))
    if not cues:
        return []

    spans: List[ReflectiveStep] = []
    for i, cue in enumerate(cues):
        next_start = cues[i + 1].cue_start if i + 1 < len(cues) else end_limit
        span_end = next_start
        if max_span_chars is not None:
            span_end = min(span_end, cue.cue_start + max_span_chars)
        spans.append(
            ReflectiveStep(
                cue_type=cue.cue_type,
                cue_start=cue.cue_start,
                cue_end=cue.cue_end,
                span_end=span_end,
            )
        )

    if allowed is not None:
        spans = [span for span in spans if span.cue_type in allowed]

    if max_spans is not None and len(spans) > max_spans:
        selection = str(span_selection or "first").lower()
        if selection == "first":
            spans = spans[:max_spans]
        elif selection in ("middle", "median"):
            start = max(0, (len(spans) - max_spans) // 2)
            spans = spans[start : start + max_spans]
        elif selection in ("first_last", "first+last", "front_back"):
            front = (max_spans + 1) // 2
            back = max_spans - front
            spans = spans[:front] + spans[-back:] if back > 0 else spans[:front]
        else:
            raise ValueError(f"Unknown reflective step span_selection: {span_selection}")
    return spans


# Prefix appended to before/after reflective-step slices so the verifier can score
# whether that partial reasoning state can already produce a correct answer.
# It is used only for training-time answer-forcing probes, not for inference early exit.
ANSWER_FORCING_SUFFIX = "\n**Final Answer**\n\\boxed"


def build_answer_forcing_prefixes(
    text: str,
    span: ReflectiveStep,
    suffix: str = ANSWER_FORCING_SUFFIX,
) -> Tuple[str, str]:
    """Construct before/after prefixes ready for answer-forced scoring."""
    before = text[: span.before_end].rstrip() + suffix
    after = text[: span.after_end].rstrip() + suffix
    return before, after


def char_span_to_token_span(
    text: str,
    char_start: int,
    char_end: int,
    tokenizer,
) -> Tuple[int, int]:
    """Map a character-level span to a token-level [start, end) range.

    Uses offset_mapping from the full-text tokenization + bisect_right so that
    boundary whitespace tokens (e.g. 'Ġ' preceding 'ĠWait') are attributed to
    the token that actually contains the character, not the preceding token.
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    token_ends = [e for _, e in enc["offset_mapping"]]
    tok_start = bisect.bisect_right(token_ends, char_start)
    tok_end = bisect.bisect_right(token_ends, char_end)
    return tok_start, tok_end
