"""RUAR: Utility-Guided Advantage Rescaling for Efficient Reasoning."""

from .advantage_rescaling import (
    RUARGammaResult,
    ReflectiveStepRegion,
    compute_ruar_gammas,
    find_answer_ready_step,
    rescale_advantages,
    utility_to_sign_gammas,
)
from .answer_forcing_utility_estimation import (
    ANSWER_FORCING_SUFFIX,
    AnswerForcingProbeBatch,
    AnswerForcingProbeEvaluation,
    AnswerForcingProbePoint,
    ReflectiveStepUtility,
    estimate_answer_forcing_utilities,
)
from .reflective_step_extraction import (
    ALT,
    CUE_NAMES,
    WAIT,
    ReflectiveStep,
    build_answer_forcing_prefixes,
    char_span_to_token_span,
    extract_reflective_steps,
    find_final_answer_start,
)

__all__ = [
    "ALT",
    "ANSWER_FORCING_SUFFIX",
    "AnswerForcingProbeBatch",
    "AnswerForcingProbeEvaluation",
    "AnswerForcingProbePoint",
    "CUE_NAMES",
    "RUARGammaResult",
    "ReflectiveStep",
    "ReflectiveStepRegion",
    "ReflectiveStepUtility",
    "WAIT",
    "build_answer_forcing_prefixes",
    "char_span_to_token_span",
    "compute_ruar_gammas",
    "estimate_answer_forcing_utilities",
    "extract_reflective_steps",
    "find_answer_ready_step",
    "find_final_answer_start",
    "rescale_advantages",
    "utility_to_sign_gammas",
]
