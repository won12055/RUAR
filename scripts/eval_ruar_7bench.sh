#!/usr/bin/env bash
# Evaluate a RUAR checkpoint on the paper-style seven benchmark suite.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUAR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export RUAR_ROOT

export DATASET_FILTER="${DATASET_FILTER:-gsm8k,math500,aime2024,hmmt25,gpqa_diamond,arc_challenge,commonsenseqa}"
export DATASET_SPECS="${DATASET_SPECS:-gsm8k|gsm8k|boxed_math/gsm8k,math500|math500|boxed_math/math500,aime2024|aime2024|boxed_math/aime2024,hmmt25|hmmt25|boxed_math/hmmt25,gpqa_diamond|gpqa_diamond|mcq/gpqa_diamond,arc_challenge|arc_challenge|mcq/arc_challenge,commonsenseqa|commonsenseqa|mcq/commonsenseqa}"
export EVAL_ROOT="${EVAL_ROOT:-$RUAR_ROOT/cot_dumps/eval_7bench}"
export DATA_ROOT="${DATA_ROOT:-$RUAR_ROOT/data/paper_eval}"
export CKPT_EVAL_ROOT="${CKPT_EVAL_ROOT:-$RUAR_ROOT/checkpoints/eval/7bench}"
export RESULT_CSV="${RESULT_CSV:-$RUAR_ROOT/reports/ruar_7bench_results.csv}"
export REPORT_MD="${REPORT_MD:-$RUAR_ROOT/reports/ruar_7bench_report.md}"
export REPORT_TITLE="${REPORT_TITLE:-RUAR Seven-Benchmark Evaluation}"
export EVAL_JOB_NAME="${EVAL_JOB_NAME:-eval_ruar_7bench}"
export BASE_MODEL_NOTE="${BASE_MODEL_NOTE:-RUAR is evaluated against the base HF model using the same paper-style benchmark prompts.}"
export RUAR_STEP="${RUAR_STEP:-30}"
export BASE_MODEL="${BASE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-18432}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-18432}"
export TOKEN_BUDGET="${TOKEN_BUDGET:-32768}"

bash "$SCRIPT_DIR/eval_ruar_math_aime.sh"
