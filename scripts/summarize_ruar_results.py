#!/usr/bin/env python3
"""Summarize RUAR evaluation CSV/COT dumps with accuracy and token counts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--format", choices=["md", "tsv"], default="md")
    return parser.parse_args()


def response_length(record: dict[str, Any]) -> int | None:
    for key in ("response_length", "response_len", "length", "token_count"):
        value = record.get(key)
        if value is not None:
            return int(value)
    ids = record.get("response_token_ids")
    if isinstance(ids, list):
        return len(ids)
    return None


def summarize_cot_dump(path: Path) -> tuple[float | None, float | None, int]:
    scores: list[float] = []
    lengths: list[int] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            score = record.get("score_raw", record.get("acc", record.get("score")))
            if score is not None:
                scores.append(float(score))
            length = response_length(record)
            if length is not None:
                lengths.append(length)
    acc = sum(scores) / len(scores) if scores else None
    tok = sum(lengths) / len(lengths) if lengths else None
    return acc, tok, max(len(scores), len(lengths))


def load_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        cot_path = Path(row["cot_dump"])
        acc, tok, n = summarize_cot_dump(cot_path)
        if acc is None and row.get("score"):
            acc = float(row["score"])
        out.append({
            "model": row["model"],
            "dataset": row["dataset"],
            "acc": acc,
            "tok": tok,
            "n": n or int(row.get("n") or 0),
            "cot_dump": str(cot_path),
        })
    return out


def add_weighted_averages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["dataset"] != "weighted_avg":
            grouped.setdefault(row["model"], []).append(row)

    result = list(rows)
    for model, model_rows in grouped.items():
        valid = [
            row
            for row in model_rows
            if row["acc"] is not None and row["tok"] is not None and row["n"] > 0
        ]
        if not valid:
            continue
        total_n = sum(row["n"] for row in valid)
        result.append({
            "model": model,
            "dataset": "weighted_avg",
            "acc": sum(row["acc"] * row["n"] for row in valid) / total_n,
            "tok": sum(row["tok"] * row["n"] for row in valid) / total_n,
            "n": total_n,
            "cot_dump": "",
        })
    return result


def row_sort_key(row: dict[str, Any]) -> tuple[bool, str, str]:
    return row["dataset"] == "weighted_avg", row["dataset"], row["model"]


def render(rows: list[dict[str, Any]], fmt: str) -> str:
    if fmt == "tsv":
        lines = ["model\tdataset\tacc\ttok\tn"]
        for row in sorted(rows, key=row_sort_key):
            lines.append(
                "\t".join([
                    row["model"],
                    row["dataset"],
                    "" if row["acc"] is None else f"{row['acc']:.6f}",
                    "" if row["tok"] is None else f"{row['tok']:.0f}",
                    str(row["n"]),
                ])
            )
        return "\n".join(lines)

    def fmt_float(value, digits: int) -> str:
        return "" if value is None else f"{value:.{digits}f}"

    lines = [
        "| model | dataset | acc | tok | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=row_sort_key):
        lines.append(
            f"| {row['model']} | {row['dataset']} | "
            f"{fmt_float(row['acc'], 3)} | "
            f"{fmt_float(row['tok'], 0)} | "
            f"{row['n']} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = add_weighted_averages(load_rows(args.results_csv))
    print(render(rows, fmt=args.format))


if __name__ == "__main__":
    main()
