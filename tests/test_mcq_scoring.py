from verl.utils.reward_score import mcq


def test_numeric_choice_labels_are_preserved():
    extra_info = {"choice_labels": ["1", "2", "3", "4"]}

    assert mcq.compute_strict_score(r"\boxed{2}", "2", extra_info=extra_info) == 1.0
    assert mcq.compute_strict_score(r"\boxed{(2)}", "2", extra_info=extra_info) == 1.0
    assert mcq.compute_loose_score("The correct answer is (2).", "2", extra_info=extra_info) == 1.0


def test_numeric_answer_does_not_match_letter_labels():
    extra_info = {"choice_labels": ["A", "B", "C", "D"]}

    assert mcq.compute_strict_score(r"\boxed{2}", "B", extra_info=extra_info) == 0.0
    assert mcq.compute_loose_score("The correct answer is (B).", "B", extra_info=extra_info) == 1.0
