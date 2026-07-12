import asyncio
from concurrent.futures import ThreadPoolExecutor

from verl.workers.reward_manager.ruar import parallel_compute_score_async


def _exact_match_score(data_source, solution_str, ground_truth, extra_info):
    del data_source, extra_info
    return float(solution_str == ground_truth)


def test_parallel_score_reuses_caller_owned_executor():
    executor = ThreadPoolExecutor(max_workers=2)
    try:
        scores = asyncio.run(
            parallel_compute_score_async(
                _exact_match_score,
                ["1", "0"],
                ["1", "1"],
                ["test", "test"],
                num_processes=2,
                timeout=1.0,
                executor=executor,
            )
        )
        assert scores == [1.0, 0.0]
        assert executor.submit(lambda: 7).result() == 7
    finally:
        executor.shutdown(wait=True)
