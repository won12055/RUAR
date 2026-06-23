# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Portions adapted from DEER (Dynamic Early Exit in Reasoning Models),
# MIT License, Copyright (c) 2025 chenxuYang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Boxed final-answer scoring for RUAR evaluation."""

from __future__ import annotations

from .math_answer import grade_answer
from .math_answer.grader import math_equal


def extract_boxed_answer(solution_str: str) -> str | None:
    """Extract the answer after the final boxed marker.

    The evaluation prompts ask the model to place the final answer in
    ``\\boxed{}``.  Some answer-forcing prompts continue from ``\\boxed``
    directly, so this accepts both ``\\boxed{42}`` and ``\\boxed 42``.
    """
    text = str(solution_str)
    idx = text.rfind("boxed")
    if idx < 0:
        return None

    answer = text[idx + len("boxed") :]
    if not answer:
        return None

    if answer[0] == "{":
        depth = 1
        chars: list[str] = []
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


def compute_score(solution_str: str, ground_truth: str, data_name: str | None = None) -> float:
    """Return 1.0 if the final boxed answer matches the reference."""
    del data_name
    extracted = extract_boxed_answer(solution_str)
    if extracted is None:
        return 0.0

    reference = str(ground_truth)
    if grade_answer(extracted, reference):
        return 1.0

    try:
        return 1.0 if math_equal(extracted, reference, timeout=True) else 0.0
    except Exception:
        return 0.0
