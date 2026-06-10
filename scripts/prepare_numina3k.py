#!/usr/bin/env python3
"""Prepare the Numina-3K RUAR training split."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SOURCES = {"math", "cn_k12", "olympiads", "aops_forum", "amc_aime"}
TRAIN_SIZE_DEFAULT = 3200
VAL_SIZE_DEFAULT = 512
MAX_ANSWER_LEN = 100
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.input_jsonl is not None:
        return load_jsonl(args.input_jsonl)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install the training dependencies, or pass --input-jsonl with a local NuminaMath-CoT export."
        ) from exc
    kwargs = {"split": args.hf_split}
    if args.hf_revision:
        kwargs["revision"] = args.hf_revision
    return load_dataset(args.hf_repo, **kwargs)


def extract_boxed(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", str(text))
    if not matches:
        return None
    answer = matches[-1].strip()
    return answer or None


def is_parseable_answer(answer: str) -> bool:
    if len(answer) > MAX_ANSWER_LEN:
        return False
    prose_words = {"prove", "show", "hence", "therefore", "thus", "since", "which"}
    lower_tokens = set(answer.lower().split())
    return not any(word in lower_tokens for word in prose_words)


def to_ruar_row(problem: str, answer: str, source: str, index: int, data_source: str) -> dict[str, Any]:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem.strip()},
        ],
        "data_source": data_source,
        "ability": "math",
        "reward_info": {"ground_truth": answer},
        "extra_info": {"index": index, "source": f"numina_{source}"},
    }


def build_candidates(rows: Iterable[dict[str, Any]], source_key: str, problem_key: str, solution_key: str) -> list[dict]:
    candidates = []
    skipped_source = 0
    skipped_answer = 0
    for row in rows:
        source = str(row.get(source_key, "")).strip()
        if source not in SOURCES:
            skipped_source += 1
            continue

        problem = row.get(problem_key)
        solution = row.get(solution_key)
        if problem is None or solution is None:
            skipped_answer += 1
            continue

        answer = extract_boxed(str(solution))
        if answer is None or not is_parseable_answer(answer):
            skipped_answer += 1
            continue
        candidates.append({"problem": str(problem), "answer": answer, "source": source})

    print(
        f"Candidates after filtering: {len(candidates)} "
        f"(skipped_source={skipped_source}, skipped_answer={skipped_answer})",
        flush=True,
    )
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/numina_3k"))
    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE_DEFAULT)
    parser.add_argument("--val-size", type=int, default=VAL_SIZE_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-source", default="numina_3k")
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--hf-repo", default="AI-MO/NuminaMath-CoT")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-revision", default=None)
    parser.add_argument("--source-key", default="source")
    parser.add_argument("--problem-key", default="problem")
    parser.add_argument("--solution-key", default="solution")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args)
    candidates = build_candidates(
        rows,
        source_key=args.source_key,
        problem_key=args.problem_key,
        solution_key=args.solution_key,
    )
    total_needed = args.train_size + args.val_size
    if len(candidates) < total_needed:
        print(f"ERROR: only {len(candidates)} candidates, need {total_needed}", file=sys.stderr)
        raise SystemExit(1)

    random.Random(args.seed).shuffle(candidates)
    train_rows = candidates[: args.train_size]
    val_rows = candidates[args.train_size : total_needed]

    train = [
        to_ruar_row(row["problem"], row["answer"], row["source"], index, args.data_source)
        for index, row in enumerate(train_rows)
    ]
    val = [
        to_ruar_row(row["problem"], row["answer"], row["source"], args.train_size + index, args.data_source)
        for index, row in enumerate(val_rows)
    ]

    source_counts: dict[str, int] = {}
    for row in train_rows:
        source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
    print(f"Train source distribution: {source_counts}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train).to_parquet(args.out_dir / "train.parquet", index=False)
    pd.DataFrame(val).to_parquet(args.out_dir / "val.parquet", index=False)
    print(f"Wrote train={len(train)} val={len(val)} to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
