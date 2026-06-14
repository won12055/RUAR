"""Answer-forcing utility estimation for RUAR reflective steps.

Implements the before/after utility estimator used in
When Reflection Helps: Utility-Guided Advantage Rescaling.

    p_before_j = (#correct out of K answer-forced rollouts from before_j) / K
    p_after_j  = (#correct out of K answer-forced rollouts from after_j ) / K
    u_j        = p_after_j - p_before_j   in [-1, +1]

Key efficiency property:
    span_j = [cue_j, cue_{j+1})  =>  after_j == before_{j+1}
So with N reflective steps per sample we only need (N + 1) unique probe points,
not 2N. We dedup probes within a sample before batching to the rollout engine.

Design:
    The generation and verification engines are injected as callables. This
    keeps the module pure-Python and unit-testable. Training integrations can
    plug these callables into any rollout generator and answer verifier.

Inputs:
    samples: list of dicts with keys
        - "prompt":        str, original formatted prompt used to generate response
        - "response":      str, the full generated trace (incl. <think>...</think>)
        - "ground_truth":  any, opaque payload passed through to verify_fn
        - "data_source":   any, same
        - "extra_info":    any, same  (optional)

Outputs:
    labels: list[ReflectiveStepUtility], one per (sample_idx, span_idx) actually probed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .reflective_step_extraction import (
    ANSWER_FORCING_SUFFIX,
    ReflectiveStep,
    extract_reflective_steps,
)


# A generate_fn takes a list of prompt strings and returns, for each prompt,
# K continuation strings. Shape: List[List[str]] with len == len(prompts) and
# each inner list of length K.
GenerateFn = Callable[[List[str], int], List[List[str]]]

# A verify_fn takes parallel lists (completion_text, ground_truth, data_source,
# extra_info) and returns a parallel list of binary correctness floats in {0, 1}.
VerifyFn = Callable[
    [List[str], List[Any], List[Any], List[Any]],
    List[float],
]


@dataclass(frozen=True)
class AnswerForcingProbePoint:
    """A single (sample, char position) at which to answer-force.

    `char_pos` is a position in `samples[sample_idx]["response"]`. The probe
    prompt is `response[:char_pos].rstrip() + ANSWER_FORCING_SUFFIX`.
    """

    sample_idx: int
    char_pos: int


@dataclass
class ReflectiveStepUtility:
    sample_idx: int
    span_idx: int
    cue_type: int
    cue_start_char: int
    cue_end_char: int
    span_end_char: int
    p_before: float
    p_after: float
    utility: float  # p_after - p_before
    raw_utility: float  # retained for logging parity with the trainer
    post_answer: bool = False
    post_answer_score: float = 0.0
    post_answer_penalty: float = 0.0
    span_class: int = 0


@dataclass
class AnswerForcingProbeBatch:
    """Intermediate dedup'd probe batch, exposed for inspection / logging."""

    points: List[AnswerForcingProbePoint] = field(default_factory=list)
    prompts: List[str] = field(default_factory=list)
    ground_truths: List[Any] = field(default_factory=list)
    data_sources: List[Any] = field(default_factory=list)
    extra_infos: List[Any] = field(default_factory=list)
    # Assistant-side prefix parallel to prompts. Used to reconstruct a complete
    # assistant response for verification after generate_fn returns continuation.
    response_prefixes: List[str] = field(default_factory=list)
    # index lookup: (sample_idx, char_pos) -> position in `points`
    index: Dict[Tuple[int, int], int] = field(default_factory=dict)


@dataclass
class AnswerForcingProbeEvaluation:
    """Generated answer-forcing continuations and verifier scores for one probe."""

    sample_idx: int
    char_pos: int
    response_prefix: str
    continuations: List[str]
    scores: List[float]
    probability: float


def _build_probe_batch(
    samples: Sequence[Dict[str, Any]],
    spans_per_sample: Sequence[List[ReflectiveStep]],
    suffix: str,
) -> AnswerForcingProbeBatch:
    """Collect unique probe points across all samples.

    For each sample with N spans we add N + 1 unique char positions:
        cue_0_start, cue_1_start, ..., cue_{N-1}_start, span_{N-1}_end
    The span_end of span_j equals cue_{j+1}_start for j < N-1, so the dedup
    inside one sample collapses adjacent spans for free.
    """
    batch = AnswerForcingProbeBatch()

    for sidx, (sample, spans) in enumerate(zip(samples, spans_per_sample)):
        if not spans:
            continue
        prompt = sample["prompt"]
        response = sample["response"]
        gt = sample["ground_truth"]
        ds = sample.get("data_source")
        ei = sample.get("extra_info")

        # Ordered, dedup'd char positions for this sample.
        positions: List[int] = []
        seen: set = set()
        for sp in spans:
            for pos in (sp.cue_start, sp.span_end):
                if pos in seen:
                    continue
                seen.add(pos)
                positions.append(pos)

        for pos in positions:
            key = (sidx, pos)
            if key in batch.index:
                continue
            response_prefix = response[:pos].rstrip() + suffix
            batch.index[key] = len(batch.points)
            batch.points.append(AnswerForcingProbePoint(sample_idx=sidx, char_pos=pos))
            batch.prompts.append(prompt + response_prefix)
            batch.response_prefixes.append(response_prefix)
            batch.ground_truths.append(gt)
            batch.data_sources.append(ds)
            batch.extra_infos.append(ei)

    return batch


def _ordered_probe_positions(spans: Sequence[ReflectiveStep]) -> List[int]:
    """Return the ordered, deduplicated probe positions for a sample."""
    positions: List[int] = []
    seen: set = set()
    for sp in spans:
        for pos in (sp.cue_start, sp.span_end):
            if pos in seen:
                continue
            seen.add(pos)
            positions.append(pos)
    return positions


def _estimate_solve_probabilities(
    batch: AnswerForcingProbeBatch,
    K: int,
    generate_fn: GenerateFn,
    verify_fn: VerifyFn,
) -> Tuple[List[float], List[AnswerForcingProbeEvaluation]]:
    """Run K answer-forced rollouts per probe and return p = (#correct)/K."""
    if not batch.prompts:
        return [], []

    completions: List[List[str]] = generate_fn(batch.prompts, K)
    assert len(completions) == len(batch.prompts), (
        f"generate_fn returned {len(completions)} groups for {len(batch.prompts)} prompts"
    )
    assert all(len(c) == K for c in completions), (
        f"generate_fn must return exactly K={K} continuations per prompt"
    )

    # Flatten for a single verifier call: K continuations per probe, in order.
    flat_completions: List[str] = []
    flat_gt: List[Any] = []
    flat_ds: List[Any] = []
    flat_ei: List[Any] = []
    for i, group in enumerate(completions):
        flat_completions.extend(batch.response_prefixes[i] + continuation for continuation in group)
        flat_gt.extend([batch.ground_truths[i]] * K)
        flat_ds.extend([batch.data_sources[i]] * K)
        flat_ei.extend([batch.extra_infos[i]] * K)

    flat_scores = verify_fn(flat_completions, flat_gt, flat_ds, flat_ei)
    assert len(flat_scores) == len(flat_completions), (
        f"verify_fn returned {len(flat_scores)} scores for {len(flat_completions)} items"
    )

    probabilities: List[float] = []
    evaluations: List[AnswerForcingProbeEvaluation] = []
    for i in range(len(batch.prompts)):
        chunk = flat_scores[i * K : (i + 1) * K]
        probability = sum(chunk) / K
        probabilities.append(probability)
        point = batch.points[i]
        evaluations.append(
            AnswerForcingProbeEvaluation(
                sample_idx=point.sample_idx,
                char_pos=point.char_pos,
                response_prefix=batch.response_prefixes[i],
                continuations=completions[i],
                scores=[float(score) for score in chunk],
                probability=probability,
            )
        )
    return probabilities, evaluations


def classify_step(p_before: float, p_after: float, class_threshold: float) -> int:
    """Classify a reflective step for logging/debugging.

    The actor update uses the continuous utility value; this class id is kept so
    dumps can expose the same diagnostic categories as the training run.
    """
    before_ok = p_before >= class_threshold
    after_ok = p_after >= class_threshold
    if not before_ok and after_ok:
        return 1
    if before_ok and after_ok:
        return 2
    if before_ok and not after_ok:
        return 3
    return 0


def estimate_answer_forcing_utilities(
    samples: Sequence[Dict[str, Any]],
    generate_fn: GenerateFn,
    verify_fn: VerifyFn,
    K: int = 4,
    max_spans_per_sample: Optional[int] = 2,
    max_span_chars: Optional[int] = None,
    span_selection: str = "first",
    cue_types=None,
    answer_forcing_suffix: str = ANSWER_FORCING_SUFFIX,
    post_answer_mode: str = "off",
    post_answer_penalty: float = 0.0,
    post_answer_threshold: float = 1.0,
    class_threshold: float = 0.75,
    ready_threshold: float = 0.75,
    consecutive_required: int = 3,
    stop_after_ready: bool = False,
    return_probe_evaluations: bool = False,
):
    """Estimate RUAR answer-forcing utilities for a batch of samples.

    Args:
        samples: see module docstring.
        generate_fn: produces K answer continuations per probe prompt.
        verify_fn: rule-based scorer, returns 1.0 for correct else 0.0.
        K: number of MC rollouts per probe point.
        max_spans_per_sample: cap to keep compute bounded.
        max_span_chars: optional truncation passed to extract_reflective_steps.
        span_selection: selection policy when max_spans_per_sample is set.
        cue_types: optional cue allow-list passed to extract_reflective_steps.
        answer_forcing_suffix: appended to each probe prefix.

    Returns:
        labels: utility labels, in `(sample_idx, span_idx)` order.
        batch:  the dedup'd probe batch (for logging / debugging).
        probs:  parallel to batch.points, p = (#correct)/K at each probe.
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if max_spans_per_sample is not None and max_spans_per_sample <= 0:
        max_spans_per_sample = None
    post_answer_mode = str(post_answer_mode).lower()
    if post_answer_mode not in ("off", "none", "disabled", "answer_ready"):
        raise ValueError(f"Unknown post_answer_mode: {post_answer_mode}")
    post_answer_penalty = float(post_answer_penalty)
    post_answer_threshold = float(post_answer_threshold)
    class_threshold = float(class_threshold)

    spans_per_sample = [
        extract_reflective_steps(
            sample["response"],
            max_spans=max_spans_per_sample,
            max_span_chars=max_span_chars,
            cue_types=cue_types,
            span_selection=span_selection,
        )
        for sample in samples
    ]
    ready_threshold = float(ready_threshold)
    consecutive_required = max(1, int(consecutive_required))
    ready_boundary_chars = [-1 for _ in samples]

    def _make_label(
        sidx: int,
        span_idx: int,
        sp: ReflectiveStep,
        p_before: float,
        p_after: float,
    ) -> ReflectiveStepUtility:
        raw_utility = p_after - p_before
        post_answer = False
        post_answer_score = 0.0
        applied_post_answer_penalty = 0.0
        utility = raw_utility
        if post_answer_mode == "answer_ready":
            post_answer_score = p_before
            post_answer = post_answer_score >= post_answer_threshold
            if post_answer:
                applied_post_answer_penalty = post_answer_penalty
                utility = min(utility, 0.0) - applied_post_answer_penalty
        return ReflectiveStepUtility(
            sample_idx=sidx,
            span_idx=span_idx,
            cue_type=sp.cue_type,
            cue_start_char=sp.cue_start,
            cue_end_char=sp.cue_end,
            span_end_char=sp.span_end,
            p_before=p_before,
            p_after=p_after,
            utility=utility,
            raw_utility=raw_utility,
            post_answer=post_answer,
            post_answer_score=post_answer_score,
            post_answer_penalty=applied_post_answer_penalty,
            span_class=classify_step(p_before, p_after, class_threshold),
        )

    if not stop_after_ready:
        batch = _build_probe_batch(samples, spans_per_sample, answer_forcing_suffix)
        probabilities, probe_evaluations = _estimate_solve_probabilities(batch, K, generate_fn, verify_fn)

        labels: List[ReflectiveStepUtility] = []
        for sidx, spans in enumerate(spans_per_sample):
            for span_idx, sp in enumerate(spans):
                i_before = batch.index.get((sidx, sp.cue_start))
                i_after = batch.index.get((sidx, sp.span_end))
                if i_before is None or i_after is None:
                    continue
                labels.append(_make_label(
                    sidx,
                    span_idx,
                    sp,
                    probabilities[i_before],
                    probabilities[i_after],
                ))

        if return_probe_evaluations:
            return labels, batch, probabilities, probe_evaluations, ready_boundary_chars
        return labels, batch, probabilities, ready_boundary_chars

    # Ready-aware probing path: stop after the first stable ready boundary and
    # avoid probing post-ready spans, which will receive default post-ready
    # scaling downstream.
    batch = AnswerForcingProbeBatch()
    probabilities: List[float] = []
    probe_evaluations: List[AnswerForcingProbeEvaluation] = []
    labels_by_sample: List[List[ReflectiveStepUtility]] = [[] for _ in samples]
    positions_per_sample = [_ordered_probe_positions(spans) for spans in spans_per_sample]
    probs_by_sample: List[List[float]] = [[] for _ in samples]
    next_pos_idx = [0 for _ in samples]
    ready_streak = [0 for _ in samples]
    ready_streak_start = [-1 for _ in samples]
    active = {sidx for sidx, spans in enumerate(spans_per_sample) if spans}

    while active:
        round_batch = AnswerForcingProbeBatch()
        round_sample_order: List[int] = []
        round_pos_order: List[int] = []
        for sidx in sorted(active):
            pos_idx = next_pos_idx[sidx]
            positions = positions_per_sample[sidx]
            if pos_idx >= len(positions):
                continue
            pos = positions[pos_idx]
            sample = samples[sidx]
            response_prefix = sample["response"][:pos].rstrip() + answer_forcing_suffix
            round_batch.index[(sidx, pos)] = len(round_batch.points)
            round_batch.points.append(AnswerForcingProbePoint(sample_idx=sidx, char_pos=pos))
            round_batch.prompts.append(sample["prompt"] + response_prefix)
            round_batch.response_prefixes.append(response_prefix)
            round_batch.ground_truths.append(sample["ground_truth"])
            round_batch.data_sources.append(sample.get("data_source"))
            round_batch.extra_infos.append(sample.get("extra_info"))
            round_sample_order.append(sidx)
            round_pos_order.append(pos)

        if not round_batch.prompts:
            break

        round_probabilities, round_evaluations = _estimate_solve_probabilities(
            round_batch, K, generate_fn, verify_fn
        )

        for point, prompt, prefix, gt, ds, ei, prob, evaluation in zip(
            round_batch.points,
            round_batch.prompts,
            round_batch.response_prefixes,
            round_batch.ground_truths,
            round_batch.data_sources,
            round_batch.extra_infos,
            round_probabilities,
            round_evaluations,
        ):
            batch.index[(point.sample_idx, point.char_pos)] = len(batch.points)
            batch.points.append(point)
            batch.prompts.append(prompt)
            batch.response_prefixes.append(prefix)
            batch.ground_truths.append(gt)
            batch.data_sources.append(ds)
            batch.extra_infos.append(ei)
            probabilities.append(prob)
            probe_evaluations.append(evaluation)

        finished_samples = []
        for sidx, pos in zip(round_sample_order, round_pos_order):
            pos_idx = next_pos_idx[sidx]
            spans = spans_per_sample[sidx]
            current_prob = round_probabilities[round_batch.index[(sidx, pos)]]
            probs_by_sample[sidx].append(current_prob)

            # Finalize the previous span once we know its p_after.
            prev_span_idx = pos_idx - 1
            if prev_span_idx >= 0 and prev_span_idx < len(spans):
                prev_span = spans[prev_span_idx]
                labels_by_sample[sidx].append(
                    _make_label(
                        sidx,
                        prev_span_idx,
                        prev_span,
                        probs_by_sample[sidx][prev_span_idx],
                        probs_by_sample[sidx][prev_span_idx + 1],
                    )
                )

            # Update ready streak on p_before for the current span start.
            if pos_idx < len(spans):
                if current_prob >= ready_threshold:
                    if ready_streak[sidx] == 0:
                        ready_streak_start[sidx] = pos_idx
                    ready_streak[sidx] += 1
                    if ready_streak[sidx] >= consecutive_required:
                        boundary_span_idx = ready_streak_start[sidx]
                        ready_boundary_chars[sidx] = spans[boundary_span_idx].cue_start
                        labels_by_sample[sidx] = [
                            label
                            for label in labels_by_sample[sidx]
                            if label.span_idx < boundary_span_idx
                        ]
                        finished_samples.append(sidx)
                        next_pos_idx[sidx] += 1
                        continue
                else:
                    ready_streak[sidx] = 0
                    ready_streak_start[sidx] = -1

            next_pos_idx[sidx] += 1
            if next_pos_idx[sidx] >= len(positions_per_sample[sidx]):
                finished_samples.append(sidx)

        for sidx in finished_samples:
            active.discard(sidx)

    labels = [label for sample_labels in labels_by_sample for label in sample_labels]

    if return_probe_evaluations:
        return labels, batch, probabilities, probe_evaluations, ready_boundary_chars
    return labels, batch, probabilities, ready_boundary_chars
