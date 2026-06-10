#!/usr/bin/env python3
"""Prepare RUAR evaluation parquet files from local benchmark JSONL files."""

# Benchmark prompt formatting follows the DEER Qwen boxed-answer convention.
# DEER is released under the MIT License; see RUAR/NOTICE.

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
ANSWER_FORCING_SUFFIX = "\n**Final Answer**\n\\boxed"
ALIASES = {
    "math500": ("math500", "math"),
    "aime2024": ("aime2024", "aime"),
    "aime2025": ("aime2025", "aime25"),
    "gsm8k": ("gsm8k",),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_source_jsonl(dataset: str, data_dir: Path, split: str) -> Path:
    candidates = [data_dir / alias / f"{split}.jsonl" for alias in ALIASES.get(dataset, (dataset,))]
    candidates.append(data_dir / f"{dataset}.jsonl")
    for path in candidates:
        if path.is_file():
            return path
    rendered = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not find JSONL for dataset={dataset!r}. Looked for:\n{rendered}\n"
        "Set --input-jsonl explicitly or place the benchmark under --data-dir."
    )


def question_text(row: dict[str, Any]) -> str:
    for key in ("problem", "question", "Question", "input"):
        value = row.get(key)
        if value is not None:
            return str(value).strip()
    raise ValueError(f"Cannot find question text in row keys: {sorted(row)}")


def answer_text(row: dict[str, Any]) -> str:
    answer = row.get("answer", row.get("final_answer"))
    if answer is None:
        raise ValueError(f"Cannot find answer in row keys: {sorted(row)}")
    return str(answer)


def chat_prompt(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def to_rows(rows: list[dict[str, Any]], dataset: str, data_source: str, start_index: int = 0) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for offset, row in enumerate(rows):
        idx = row.get("idx", row.get("id", start_index + offset))
        prepared.append(
            {
                "prompt": chat_prompt(question_text(row)),
                "data_source": data_source,
                "ability": "math",
                "reward_info": {"ground_truth": answer_text(row)},
                "extra_info": {
                    "index": idx,
                    "benchmark": dataset,
                    "answer_forcing_suffix": ANSWER_FORCING_SUFFIX,
                },
            }
        )
    return prepared


def select_rows(
    rows: list[dict[str, Any]],
    train_size: int,
    val_size: int,
    shuffle: bool,
    seed: int,
    exclude_train_from_val: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(rows)
    if shuffle:
        random.Random(seed).shuffle(rows)

    if train_size < 1:
        raise ValueError("--train-size must be at least 1 because the eval trainer still builds a train loader")
    if train_size > len(rows):
        raise ValueError(f"--train-size={train_size} exceeds dataset size={len(rows)}")

    train_rows = rows[:train_size]
    val_pool = rows[train_size:] if exclude_train_from_val else rows
    if val_size < 0:
        val_rows = val_pool
    else:
        if val_size > len(val_pool):
            raise ValueError(f"--val-size={val_size} exceeds available validation rows={len(val_pool)}")
        val_rows = val_pool[:val_size]
    if not val_rows:
        raise ValueError("Validation parquet would be empty")
    return train_rows, val_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset key, e.g. math500, aime2024, aime2025, gsm8k.")
    parser.add_argument("--data-source", default=None, help="Value written to the parquet data_source column.")
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "benchmarks")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=1)
    parser.add_argument("--val-size", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--exclude-train-from-val", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_jsonl = args.input_jsonl or find_source_jsonl(args.dataset, args.data_dir, args.split)
    rows = load_jsonl(input_jsonl)
    train_rows, val_rows = select_rows(
        rows,
        train_size=args.train_size,
        val_size=args.val_size,
        shuffle=args.shuffle,
        seed=args.seed,
        exclude_train_from_val=args.exclude_train_from_val,
    )

    data_source = args.data_source or args.dataset
    train = to_rows(train_rows, dataset=args.dataset, data_source=data_source)
    val = to_rows(val_rows, dataset=args.dataset, data_source=data_source)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train).to_parquet(args.out_dir / "train.parquet", index=False)
    pd.DataFrame(val).to_parquet(args.out_dir / "test.parquet", index=False)
    print(
        f"wrote dataset={args.dataset} data_source={data_source} "
        f"train={len(train)} val={len(val)} out_dir={args.out_dir}"
    )


if __name__ == "__main__":
    main()
