from ruar.advantage_rescaling import (
    ReflectiveStepRegion,
    compute_ruar_gammas,
    find_answer_ready_step,
    rescale_advantages,
    utility_to_sign_gammas,
)
from ruar_eval.metrics import accuracy_efficiency_score


def test_utility_to_sign_gammas():
    assert utility_to_sign_gammas(0.5) == (1.5, 0.5)
    assert utility_to_sign_gammas(-0.5) == (0.5, 1.5)


def test_find_answer_ready_step():
    assert find_answer_ready_step([0.0, 0.75, 0.8, 1.0], 0.75, 3) == 1
    assert find_answer_ready_step([0.0, 0.75, 0.5, 1.0], 0.75, 2) is None


def test_compute_ruar_gammas():
    result = compute_ruar_gammas(
        response_length=8,
        step_regions=[
            ReflectiveStepRegion(1, 3, 0.5, 0.0),
            ReflectiveStepRegion(3, 5, 0.0, 0.75),
            ReflectiveStepRegion(5, 7, -0.5, 0.8),
        ],
        ready_threshold=0.75,
        consecutive_required=2,
        final_answer_start_token=7,
    )
    assert result.answer_ready_step_index == 1
    assert result.positive_gamma == [1.0, 1.5, 1.5, 0.25, 0.25, 0.25, 0.25, 1.0]
    assert result.negative_gamma == [1.0, 0.5, 0.5, 1.25, 1.25, 1.25, 1.25, 1.0]
    assert rescale_advantages([1, -1], [1.5, 0.25], [0.5, 1.25]) == [1.5, -1.25]


def test_accuracy_efficiency_score():
    assert round(accuracy_efficiency_score(0.934, 5048, 0.928, 3069), 3) == 0.360
