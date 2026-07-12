from ruar.answer_forcing_utility_estimation import estimate_answer_forcing_utilities
from ruar.reflective_step_extraction import (
    ANSWER_FORCING_SUFFIX,
    WAIT,
    extract_reflective_steps,
)


def _sample(response):
    return {
        "prompt": "PROMPT\n",
        "response": response,
        "ground_truth": "1",
        "data_source": "test",
    }


def _probe_functions(sample, probability_by_position, k=4):
    prompt_to_scores = {}
    for char_pos, probability in probability_by_position.items():
        prompt = (
            sample["prompt"]
            + sample["response"][:char_pos].rstrip()
            + ANSWER_FORCING_SUFFIX
        )
        correct = round(probability * k)
        prompt_to_scores[prompt] = ["1"] * correct + ["0"] * (k - correct)

    def generate_fn(prompts, requested_k):
        assert requested_k == k
        return [prompt_to_scores[prompt] for prompt in prompts]

    def verify_fn(completions, ground_truths, data_sources, extra_infos):
        del ground_truths, data_sources, extra_infos
        return [float(completion.endswith("1")) for completion in completions]

    return generate_fn, verify_fn


def _run_ready_aware(response, probability_by_position, **kwargs):
    sample = _sample(response)
    generate_fn, verify_fn = _probe_functions(sample, probability_by_position)
    return estimate_answer_forcing_utilities(
        [sample],
        generate_fn,
        verify_fn,
        K=4,
        max_spans_per_sample=None,
        cue_types=[WAIT],
        stop_after_ready=True,
        consecutive_required=kwargs.pop("consecutive_required", 99),
        **kwargs,
    )


def test_ready_aware_reuses_shared_wait_boundary():
    response = "<think>Wait first. Wait second.</think>"
    spans = extract_reflective_steps(response, cue_types=[WAIT])
    shared = spans[0].span_end
    assert shared == spans[1].cue_start

    labels, batch, probabilities, ready_boundaries = _run_ready_aware(
        response,
        {
            spans[0].cue_start: 0.0,
            shared: 1.0,
            spans[1].span_end: 0.0,
        },
    )

    assert [point.char_pos for point in batch.points] == [
        spans[0].cue_start,
        shared,
        spans[1].span_end,
    ]
    assert probabilities == [0.0, 1.0, 0.0]
    assert [(label.span_idx, label.utility) for label in labels] == [(0, 1.0), (1, -1.0)]
    assert ready_boundaries == [-1]


def test_ready_aware_uses_exact_boundaries_across_alternatively_gap():
    response = "<think>Wait first. Alternatively reconsider. Wait second.</think>"
    spans = extract_reflective_steps(response, cue_types=[WAIT])
    assert spans[0].span_end < spans[1].cue_start

    labels, batch, probabilities, ready_boundaries = _run_ready_aware(
        response,
        {
            spans[0].cue_start: 0.0,
            spans[0].span_end: 1.0,
            spans[1].cue_start: 0.0,
            spans[1].span_end: 1.0,
        },
    )

    assert [point.char_pos for point in batch.points] == [
        spans[0].cue_start,
        spans[0].span_end,
        spans[1].cue_start,
        spans[1].span_end,
    ]
    assert probabilities == [0.0, 1.0, 0.0, 1.0]
    assert [(label.span_idx, label.utility) for label in labels] == [(0, 1.0), (1, 1.0)]
    assert ready_boundaries == [-1]


def test_ready_streak_ignores_unselected_alternatively_endpoint():
    response = "<think>Wait first. Alternatively reconsider. Wait second.</think>"
    spans = extract_reflective_steps(response, cue_types=[WAIT])

    labels, batch, _, ready_boundaries = _run_ready_aware(
        response,
        {
            spans[0].cue_start: 1.0,
            spans[0].span_end: 1.0,
            spans[1].cue_start: 1.0,
            spans[1].span_end: 0.0,
        },
        consecutive_required=2,
        ready_threshold=0.75,
    )

    # The Alternatively endpoint must not count as the second ready estimate;
    # probing continues until the second selected Wait start.
    assert [point.char_pos for point in batch.points] == [
        spans[0].cue_start,
        spans[0].span_end,
        spans[1].cue_start,
    ]
    assert labels == []
    assert ready_boundaries == [spans[0].cue_start]
