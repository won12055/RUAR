#!/usr/bin/env bash
# Evaluate a RUAR checkpoint on the paper-style seven benchmark suite.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUAR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export RUAR_ROOT

export DATASET_FILTER="${DATASET_FILTER:-gsm8k,math500,aime2024,hmmt25,gpqa_diamond,arc_challenge,commonsenseqa}"
export DATASET_SPECS="${DATASET_SPECS:-gsm8k|gsm8k|boxed_math/gsm8k,math500|math500|boxed_math/math500,aime2024|aime2024|boxed_math/aime2024,hmmt25|hmmt25|boxed_math/hmmt25,gpqa_diamond|gpqa_diamond|loose_mcq/gpqa_diamond,arc_challenge|arc_challenge|loose_mcq/arc_challenge,commonsenseqa|commonsenseqa|loose_mcq/commonsenseqa}"
export EVAL_ROOT="${EVAL_ROOT:-$RUAR_ROOT/cot_dumps/eval_7bench}"
export DATA_ROOT="${DATA_ROOT:-$RUAR_ROOT/data/paper_eval}"
export CKPT_EVAL_ROOT="${CKPT_EVAL_ROOT:-$RUAR_ROOT/checkpoints/eval/7bench}"
export RESULT_CSV="${RESULT_CSV:-$RUAR_ROOT/reports/ruar_7bench_results.csv}"
export REPORT_MD="${REPORT_MD:-$RUAR_ROOT/reports/ruar_7bench_report.md}"
export REPORT_TITLE="${REPORT_TITLE:-RUAR Seven-Benchmark Evaluation}"
export BASE_MODEL_NOTE="${BASE_MODEL_NOTE:-RUAR is evaluated against the base HF model using the same paper-style benchmark prompts.}"

bash "$SCRIPT_DIR/eval_ruar_math_aime.sh"
