"""Reporting metrics for RUAR evaluation tables.

The Accuracy-Efficiency Score (AES) formula follows the evaluation metric used
by O1-Pruner, with alpha=1, beta=3, and gamma=5. It is computed relative to a
base row and is provided only for reporting paper-style evaluation summaries;
it is not part of the RUAR training method.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AccuracyEfficiency:
    accuracy: float
    tokens: float
    token_reduction: float
    relative_accuracy_delta: float
    aes: float


def accuracy_efficiency_score(
    base_accuracy: float,
    base_tokens: float,
    model_accuracy: float,
    model_tokens: float,
) -> float:
    """Compute AES relative to the base row.

    Delta_len = (Len_base - Len_model) / Len_base.
    Delta_acc = (Acc_model - Acc_base) / Acc_base.
    AES = Delta_len + 3|Delta_acc| if accuracy does not drop, otherwise
    AES = Delta_len - 5|Delta_acc|.
    """

    if base_tokens <= 0:
        raise ValueError(f"base_tokens must be positive, got {base_tokens}")
    if base_accuracy <= 0:
        raise ValueError(f"base_accuracy must be positive, got {base_accuracy}")
    delta_len = (float(base_tokens) - float(model_tokens)) / float(base_tokens)
    delta_acc = (float(model_accuracy) - float(base_accuracy)) / float(base_accuracy)
    if delta_acc >= 0:
        return delta_len + 3.0 * abs(delta_acc)
    return delta_len - 5.0 * abs(delta_acc)


def summarize_accuracy_efficiency(
    base_accuracy: float,
    base_tokens: float,
    model_accuracy: float,
    model_tokens: float,
) -> AccuracyEfficiency:
    """Return accuracy, token reduction, relative accuracy delta, and AES."""

    token_reduction = (float(base_tokens) - float(model_tokens)) / float(base_tokens)
    relative_accuracy_delta = (float(model_accuracy) - float(base_accuracy)) / float(base_accuracy)
    return AccuracyEfficiency(
        accuracy=float(model_accuracy),
        tokens=float(model_tokens),
        token_reduction=token_reduction,
        relative_accuracy_delta=relative_accuracy_delta,
        aes=accuracy_efficiency_score(
            base_accuracy=base_accuracy,
            base_tokens=base_tokens,
            model_accuracy=model_accuracy,
            model_tokens=model_tokens,
        ),
    )
