#!/usr/bin/env python3
"""Summarize RUAR eval CSV/COT dumps with token counts and AES."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ruar_eval.metrics import accuracy_efficiency_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--base-label", default="base")
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


def render(rows: list[dict[str, Any]], base_label: str, fmt: str) -> str:
    bases = {row["dataset"]: row for row in rows if row["model"] == base_label}
    enriched = []
    for row in rows:
        base = bases.get(row["dataset"])
        aes = None
        reduction = None
        if base and row["acc"] is not None and row["tok"] is not None and base["acc"] and base["tok"]:
            reduction = (base["tok"] - row["tok"]) / base["tok"]
            aes = accuracy_efficiency_score(base["acc"], base["tok"], row["acc"], row["tok"])
        enriched.append({**row, "token_reduction": reduction, "aes": aes})

    if fmt == "tsv":
        lines = ["model\tdataset\tacc\ttok\ttoken_reduction\tAES\tn"]
        for row in sorted(enriched, key=lambda r: (r["dataset"], r["model"])):
            lines.append(
                "\t".join([
                    row["model"],
                    row["dataset"],
                    "" if row["acc"] is None else f"{row['acc']:.6f}",
                    "" if row["tok"] is None else f"{row['tok']:.0f}",
                    "" if row["token_reduction"] is None else f"{row['token_reduction']:.3f}",
                    "" if row["aes"] is None else f"{row['aes']:.3f}",
                    str(row["n"]),
                ])
            )
        return "\n".join(lines)

    def fmt_float(value, digits: int) -> str:
        return "" if value is None else f"{value:.{digits}f}"

    lines = [
        "| model | dataset | acc | tok | token reduction | AES | n |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(enriched, key=lambda r: (r["dataset"], r["model"])):
        lines.append(
            f"| {row['model']} | {row['dataset']} | "
            f"{fmt_float(row['acc'], 3)} | "
            f"{fmt_float(row['tok'], 0)} | "
            f"{fmt_float(row['token_reduction'], 3)} | "
            f"{fmt_float(row['aes'], 3)} | {row['n']} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.results_csv)
    print(render(rows, base_label=args.base_label, fmt=args.format))


if __name__ == "__main__":
    main()
