#!/usr/bin/env python3
"""Prepare RUAR evaluation parquet files from benchmark sources."""

# Held-out MCQ evaluation uses the boxed option-letter prompt used by the
# Qwen3-8B reproduction runs.

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


SYSTEM_PROMPT_MATH = "Please reason step by step, and put your final answer within \\boxed{}."
SYSTEM_PROMPT_MCQ = "Please reason step by step, and put only the final option letter within \\boxed{}."
ANSWER_FORCING_SUFFIX = "\n**Final Answer**\n\\boxed"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

ALIASES = {
    "math": ("math", "math500"),
    "math500": ("math500", "math"),
    "aime": ("aime", "aime2024"),
    "aime2024": ("aime2024", "aime"),
    "gsm8k": ("gsm8k",),
    "hmmt25": ("hmmt25",),
    "gpqa_diamond": ("gpqa_diamond", "gpqa"),
    "arc_challenge": ("arc_challenge",),
    "commonsenseqa": ("commonsenseqa",),
}
MCQ_DATASETS = {"gpqa_diamond", "arc_challenge", "commonsenseqa"}
DEFAULT_DATA_SOURCES = {
    "gsm8k": "boxed_math/gsm8k",
    "math500": "boxed_math/math500",
    "aime2024": "boxed_math/aime2024",
    "hmmt25": "boxed_math/hmmt25",
    "gpqa_diamond": "mcq/gpqa_diamond",
    "arc_challenge": "mcq/arc_challenge",
    "commonsenseqa": "mcq/commonsenseqa",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def candidate_jsonl_paths(dataset: str, data_dir: Path, split: str) -> list[Path]:
    candidates = [data_dir / alias / f"{split}.jsonl" for alias in ALIASES.get(dataset, (dataset,))]
    candidates.append(data_dir / f"{dataset}.jsonl")
    return candidates


def find_source_jsonl(dataset: str, data_dir: Path, split: str) -> Path | None:
    for path in candidate_jsonl_paths(dataset, data_dir, split):
        if path.is_file():
            return path
    return None


def question_text(row: dict[str, Any]) -> str:
    for key in ("problem", "question", "Question", "input"):
        value = row.get(key)
        if value is not None:
            return str(value).strip()
    raise ValueError(f"Cannot find question text in row keys: {sorted(row)}")


def math_answer_text(row: dict[str, Any]) -> str:
    answer = row.get("answer", row.get("final_answer"))
    if answer is None:
        raise ValueError(f"Cannot find answer in row keys: {sorted(row)}")
    return str(answer)


def _answer_to_letter(answer: Any) -> str:
    if isinstance(answer, int):
        return LETTERS[answer]
    text = str(answer).strip()
    if len(text) == 1 and text.upper() in LETTERS:
        return text.upper()
    if text.isdigit():
        return LETTERS[int(text)]
    return text


def _arc_answer_to_letter(answer: Any) -> str:
    text = str(answer).strip()
    if len(text) == 1 and text.upper() in LETTERS:
        return text.upper()
    if text.isdigit():
        value = int(text)
        if 1 <= value <= len(LETTERS):
            return LETTERS[value - 1]
    return _answer_to_letter(answer)


def _choice_texts(row: dict[str, Any]) -> list[str] | None:
    choices = row.get("choices", row.get("options"))
    if choices is None:
        return None
    if isinstance(choices, dict):
        if "text" in choices:
            return [str(choice).strip() for choice in choices["text"]]
        ordered = [choices[key] for key in sorted(choices) if len(str(key)) == 1]
        return [str(choice).strip() for choice in ordered]
    return [str(choice).strip() for choice in choices]


def _format_choices(choices: list[str]) -> str:
    return "\n".join(f"{LETTERS[i]}. {choice}" for i, choice in enumerate(choices))


def mcq_answer_text(row: dict[str, Any], dataset: str) -> str:
    if dataset == "arc_challenge" and "answerKey" in row:
        return _arc_answer_to_letter(row["answerKey"])
    answer = row.get("answer", row.get("answerKey", row.get("label", row.get("final_answer"))))
    if answer is None:
        raise ValueError(f"Cannot find MCQ answer in row keys: {sorted(row)}")
    return _answer_to_letter(answer)


def mcq_question_text(row: dict[str, Any], dataset: str) -> str:
    if row.get("problem") is not None and _choice_texts(row) is None:
        return str(row["problem"]).strip()

    choices = _choice_texts(row)
    question = question_text(row)
    if choices:
        return (
            f"{question}\n\n"
            f"{_format_choices(choices)}\n\n"
            "Choose the single best answer. Put only the final option letter inside \\boxed{}."
        )
    return (
        f"{question}\n\n"
        "Choose the single best answer. Put only the final option letter inside \\boxed{}."
    )


def choice_labels(row: dict[str, Any]) -> list[str]:
    choices = _choice_texts(row)
    if choices:
        return list(LETTERS[: len(choices)])
    return list("ABCDEFGH")


def chat_prompt(system: str, user: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def load_hf_rows(dataset: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional `datasets` dependency or provide --input-jsonl/local benchmark JSONL."
        ) from exc

    if dataset == "arc_challenge":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        return [
            {
                "idx": row.get("id", i),
                "question": row["question"],
                "choices": list(row["choices"]["text"]),
                "answer": _arc_answer_to_letter(row["answerKey"]),
            }
            for i, row in enumerate(ds)
        ]
    if dataset == "commonsenseqa":
        ds = load_dataset("tau/commonsense_qa", split="validation")
        return [
            {
                "idx": row.get("id", i),
                "question": row["question"],
                "choices": list(row["choices"]["text"]),
                "answer": row["answerKey"],
            }
            for i, row in enumerate(ds)
        ]
    raise FileNotFoundError


def load_source_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    input_jsonl = args.input_jsonl or find_source_jsonl(args.dataset, args.data_dir, args.split)
    if input_jsonl is not None:
        return load_jsonl(input_jsonl)
    if args.dataset in {"arc_challenge", "commonsenseqa"}:
        return load_hf_rows(args.dataset)

    rendered = "\n".join(f"  - {path}" for path in candidate_jsonl_paths(args.dataset, args.data_dir, args.split))
    raise FileNotFoundError(
        f"Could not find JSONL for dataset={args.dataset!r}. Looked for:\n{rendered}\n"
        "Set --input-jsonl explicitly or place the benchmark under --data-dir. "
        "For GPQA-Diamond, provide the gpqa_diamond/test.jsonl export used by the evaluation protocol."
    )


def to_rows(rows: list[dict[str, Any]], dataset: str, data_source: str, start_index: int = 0) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    is_mcq = dataset in MCQ_DATASETS
    for offset, row in enumerate(rows):
        idx = row.get("idx", row.get("id", start_index + offset))
        if is_mcq:
            labels = choice_labels(row)
            prepared.append(
                {
                    "prompt": chat_prompt(SYSTEM_PROMPT_MCQ, mcq_question_text(row, dataset)),
                    "data_source": data_source,
                    "ability": "multiple_choice",
                    "reward_info": {"style": "rule", "ground_truth": mcq_answer_text(row, dataset)},
                    "extra_info": {
                        "index": idx,
                        "benchmark": dataset,
                        "choice_labels": labels,
                        "primary_metric": "loose_letter_accuracy",
                        "secondary_metric": "strict_boxed_accuracy",
                        "answer_forcing_suffix": ANSWER_FORCING_SUFFIX,
                    },
                }
            )
        else:
            prepared.append(
                {
                    "prompt": chat_prompt(SYSTEM_PROMPT_MATH, question_text(row)),
                    "data_source": data_source,
                    "ability": "math",
                    "reward_info": {"style": "rule", "ground_truth": math_answer_text(row)},
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
    parser.add_argument("--dataset", required=True, help="Dataset key, e.g. math500, aime2024, gpqa_diamond, arc_challenge.")
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
    parser.add_argument(
        "--write-train-placeholder",
        action="store_true",
        help="Also write a one-row train.parquet for older eval launchers that require data.train_files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_source_rows(args)
    train_rows, val_rows = select_rows(
        rows,
        train_size=args.train_size,
        val_size=args.val_size,
        shuffle=args.shuffle,
        seed=args.seed,
        exclude_train_from_val=args.exclude_train_from_val,
    )

    data_source = args.data_source or DEFAULT_DATA_SOURCES.get(args.dataset, args.dataset)
    train = to_rows(train_rows, dataset=args.dataset, data_source=data_source)
    val = to_rows(val_rows, dataset=args.dataset, data_source=data_source)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.write_train_placeholder:
        pd.DataFrame(train).to_parquet(args.out_dir / "train.parquet", index=False)
    pd.DataFrame(val).to_parquet(args.out_dir / "test.parquet", index=False)
    train_msg = f" train={len(train)}" if args.write_train_placeholder else ""
    print(
        f"wrote dataset={args.dataset} data_source={data_source}"
        f"{train_msg} test={len(val)} out_dir={args.out_dir}"
    )


if __name__ == "__main__":
    main()
