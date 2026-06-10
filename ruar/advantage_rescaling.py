"""Utility-Guided Advantage Rescaling for RUAR.

This module contains the method-level rescaling logic from
When Reflection Helps. It is intentionally trainer-independent so the equations
can be tested and inspected without launching distributed training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class ReflectiveStepRegion:
    """Token-level region for one extracted reflective reasoning step."""

    start_token: int
    end_token: int
    utility: float
    p_before: float


@dataclass(frozen=True)
class RUARGammaResult:
    """Positive/negative advantage multipliers for one response."""

    positive_gamma: list[float]
    negative_gamma: list[float]
    answer_ready_step_index: Optional[int]
    answer_ready_token: Optional[int]


def utility_to_sign_gammas(utility: float) -> tuple[float, float]:
    """Return Eq. 5 multipliers for positive and negative advantages."""

    utility = float(utility)
    return max(0.0, 1.0 + utility), max(0.0, 1.0 - utility)


def find_answer_ready_step(
    p_before_values: Sequence[float],
    ready_threshold: float = 0.75,
    consecutive_required: int = 3,
) -> Optional[int]:
    """Return the first index in a c-step near-answer-ready streak.

    This implements Eq. 6 from the paper. If no streak exists, return None.
    """

    consecutive_required = max(1, int(consecutive_required))
    streak = 0
    streak_start: Optional[int] = None
    for idx, p_before in enumerate(p_before_values):
        if float(p_before) >= float(ready_threshold):
            if streak == 0:
                streak_start = idx
            streak += 1
            if streak >= consecutive_required:
                return streak_start
        else:
            streak = 0
            streak_start = None
    return None


def compute_ruar_gammas(
    response_length: int,
    step_regions: Sequence[ReflectiveStepRegion],
    ready_threshold: float = 0.75,
    consecutive_required: int = 3,
    post_ready_positive_multiplier: float = 0.25,
    post_ready_negative_multiplier: float = 1.25,
    final_answer_start_token: Optional[int] = None,
) -> RUARGammaResult:
    """Compute RUAR token multipliers for one response.

    Pre-ready reflective steps receive utility-guided multipliers. From the
    answer-ready step onward, reasoning tokens receive fixed post-ready
    multipliers. If ``final_answer_start_token`` is supplied, the answer segment
    is excluded from post-ready rescaling.
    """

    response_length = max(0, int(response_length))
    pos_gamma = [1.0] * response_length
    neg_gamma = [1.0] * response_length
    ordered_steps = sorted(step_regions, key=lambda s: s.start_token)
    ready_idx = find_answer_ready_step(
        [s.p_before for s in ordered_steps],
        ready_threshold=ready_threshold,
        consecutive_required=consecutive_required,
    )

    pre_ready_end = ready_idx if ready_idx is not None else len(ordered_steps)
    for step in ordered_steps[:pre_ready_end]:
        start = max(0, min(response_length, int(step.start_token)))
        end = max(start, min(response_length, int(step.end_token)))
        step_pos_gamma, step_neg_gamma = utility_to_sign_gammas(step.utility)
        for token_idx in range(start, end):
            pos_gamma[token_idx] = step_pos_gamma
            neg_gamma[token_idx] = step_neg_gamma

    ready_token = None
    if ready_idx is not None and ready_idx < len(ordered_steps):
        ready_token = max(0, min(response_length, int(ordered_steps[ready_idx].start_token)))
        post_ready_end = response_length
        if final_answer_start_token is not None:
            post_ready_end = max(ready_token, min(response_length, int(final_answer_start_token)))
        for token_idx in range(ready_token, post_ready_end):
            pos_gamma[token_idx] = float(post_ready_positive_multiplier)
            neg_gamma[token_idx] = float(post_ready_negative_multiplier)

    return RUARGammaResult(
        positive_gamma=pos_gamma,
        negative_gamma=neg_gamma,
        answer_ready_step_index=ready_idx,
        answer_ready_token=ready_token,
    )


def rescale_advantages(advantages, positive_gamma: Sequence[float], negative_gamma: Sequence[float]):
    """Apply sign-dependent RUAR multipliers to a vector of advantages.

    Works with Python sequences and with torch tensors. Torch is imported only
    when a tensor-like input is passed.
    """

    if hasattr(advantages, "device") and hasattr(advantages, "dtype"):
        import torch

        pos = torch.as_tensor(positive_gamma, device=advantages.device, dtype=advantages.dtype)
        neg = torch.as_tensor(negative_gamma, device=advantages.device, dtype=advantages.dtype)
        return advantages * torch.where(advantages >= 0, pos, neg)

    return [
        float(adv) * (float(pos) if float(adv) >= 0.0 else float(neg))
        for adv, pos, neg in zip(advantages, positive_gamma, negative_gamma)
    ]
