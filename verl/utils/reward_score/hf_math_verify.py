import numpy as np
import torch
from verl import DataProto
from math_verify import parse, verify


def math_verify_reward_function(solution_str, ground_truth):
    ground_truth = [ground_truth] if isinstance(ground_truth, str) else ground_truth

    # We always take the final solution
    if "</think>" in solution_str:
        solution_str = solution_str.split("</think>")[1]

    # if we can find a boxed representation, always use that
    from verl.utils.reward_score.math_answer import _last_boxed_only_string
    solution_str = _last_boxed_only_string(solution_str) or solution_str

    # to make this more stable, wrap with box
    solution_str = f"\\boxed{{{solution_str}}}"

    # 0 in case parsing cannot be completed
    try:
        math_verify_parsed = parse(solution_str, parsing_timeout=5)
    except Exception:
        return 0.0

    # 0 if parsing is problematic
    if len(math_verify_parsed) < 2:
        return 0.0

    # We perform a quick string match first
    if math_verify_parsed[1] in ground_truth:
        return 1.0

    # We now fallback to semantic verification
    for gt in ground_truth:
        try:
            if verify(
                    parse(f"\\boxed{{{gt}}}", parsing_timeout=5),
                    math_verify_parsed,
                    timeout_seconds=5,
            ):
                return 1.0
        except Exception:
            continue

    # Very unlikely to be correct after the above matches
    return 0.0


def compute_score(solution_str, ground_truth):
    if isinstance(ground_truth, (str, float, int)):
        ground_truth = [ground_truth]
    elif isinstance(ground_truth, list) and isinstance(ground_truth[0], np.ndarray):
        ground_truth = ground_truth[0].tolist()
    score = math_verify_reward_function( solution_str, ground_truth)
    return float(score)

if __name__ == "__main__":
    # print(compute_score("""\\begin{bmatrix}\n -7 & 6 & -8 \\\\\n 11 & -9 & 12 \\\\\n 15 & -16 & 19 \n \\end{bmatrix}""", """\\begin{pmatrix}\n -7 & 6 & -8 \\\\\n 11 & -9 & 12 \\\\\n 15 & -16 & 19\n \\end{pmatrix}"""))
    print(compute_score("""<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.""","""$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"""))
