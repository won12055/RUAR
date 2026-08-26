from __future__ import annotations

import asyncio
from concurrent.futures import Executor, ProcessPoolExecutor
from functools import partial
from typing import Any, Callable

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


def _score_one(
    evaluation_func: Callable,
    completion: str,
    reference: Any,
    data_source: str,
    extra_info: Any,
) -> float:
    score = evaluation_func(
        data_source=data_source,
        solution_str=completion,
        ground_truth=reference,
        extra_info=extra_info,
    )
    if isinstance(score, (int, float, bool)):
        return float(score)
    return float(score[0])


async def parallel_compute_score_async(
    evaluation_func,
    completions,
    references,
    tasks,
    extra_info=None,
    num_processes=64,
    timeout=300.0,
    executor: Executor | None = None,
    batch_size: int | None = None,
):
    if extra_info is None:
        extra_info = [None] * len(tasks)

    loop = asyncio.get_running_loop()
    scores = []
    owns_executor = executor is None
    if executor is None:
        executor = ProcessPoolExecutor(max_workers=num_processes)
    if batch_size is None or batch_size <= 0:
        batch_size = len(completions) or 1

    try:
        for start in range(0, len(completions), batch_size):
            stop = min(start + batch_size, len(completions))
            futures = [
                loop.run_in_executor(
                    executor,
                    partial(_score_one, evaluation_func, completion, reference, data_source, task_extra_info),
                )
                for completion, reference, data_source, task_extra_info in zip(
                    completions[start:stop],
                    references[start:stop],
                    tasks[start:stop],
                    extra_info[start:stop],
                )
            ]
            results = await asyncio.gather(
                *(asyncio.wait_for(future, timeout=timeout) for future in futures),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    scores.append(0.0)
                else:
                    scores.append(float(result))
    finally:
        if owns_executor:
            executor.shutdown(wait=True)
    return scores


class RUARRewardManager:
    """Rule-based terminal reward manager for RUAR training."""

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score

    def _decode_responses(self, data: DataProto) -> list[str]:
        return self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)

    def _ground_truths(self, data: DataProto) -> list[Any]:
        return [item.non_tensor_batch["reward_info"]["ground_truth"] for item in data]

    def verify(self, data: DataProto, n_samples=None):
        del n_samples
        responses = self._decode_responses(data)
        data_sources = data.non_tensor_batch["data_source"]
        extra_info = data.non_tensor_batch.get("extra_info", [None] * len(data_sources))
        try:
            scores = asyncio.run(
                parallel_compute_score_async(
                    self.compute_score,
                    responses,
                    self._ground_truths(data),
                    data_sources,
                    extra_info=extra_info,
                    num_processes=64,
                )
            )
        except Exception as exc:
            print(f"Unexpected error in reward computation. Setting all scores to 0: {exc}")
            scores = [0.0 for _ in responses]

        data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=data.batch["responses"].device)
        return scores

    def __call__(self, data: DataProto):
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        prompt_length = data.batch["prompts"].shape[-1]
        response_lengths = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1).long()
        terminal_indices = torch.clamp(response_lengths, min=1) - 1
        scores = self.verify(data)

        rows = torch.arange(len(data), device=reward_tensor.device)
        reward_tensor[rows, terminal_indices] = torch.tensor(
            scores,
            dtype=torch.float32,
            device=reward_tensor.device,
        )
        return reward_tensor
