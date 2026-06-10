#!/usr/bin/env bash
# Evaluate a RUAR checkpoint on MATH500 and AIME2024.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUAR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${RUAR_LOG_DIR:-$RUAR_ROOT/logs}"
mkdir -p "$LOG_DIR" "$RUAR_ROOT/reports" "$RUAR_ROOT/cot_dumps" "$RUAR_ROOT/checkpoints/eval"

export RUN_TAG="${RUN_TAG:-ruar_qwen3_8b_numina_3k_50step}"
export CKPT_COND="${CKPT_COND:-ruar_wait_rloo}"
export CKPT_ROOT="${CKPT_ROOT:-$RUAR_ROOT/checkpoints/ruar/numina_3k/$RUN_TAG/$CKPT_COND}"
export RUAR_STEP="${RUAR_STEP:-50}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
export RUAR_MODEL_LABEL="${RUAR_MODEL_LABEL:-ruar}"
export INCLUDE_BASE_MODEL="${INCLUDE_BASE_MODEL:-1}"
export INCLUDE_RUAR_MODEL="${INCLUDE_RUAR_MODEL:-1}"
export DATASET_FILTER="${DATASET_FILTER:-math500,aime2024}"
export MODEL_FILTER="${MODEL_FILTER:-base,ruar}"
export EVAL_COND_PREFIX="${EVAL_COND_PREFIX:-ruar}"
export EVAL_PROJECT_NAME="${EVAL_PROJECT_NAME:-ruar_eval}"
export EVAL_ROOT="${EVAL_ROOT:-$RUAR_ROOT/cot_dumps/eval_math_aime}"
export DATA_ROOT="${DATA_ROOT:-$RUAR_ROOT/data/eval_ruar_grid}"
export BENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR:-$RUAR_ROOT/data/benchmarks}"
export CKPT_EVAL_ROOT="${CKPT_EVAL_ROOT:-$RUAR_ROOT/checkpoints/eval/math_aime}"
export RESULT_CSV="${RESULT_CSV:-$RUAR_ROOT/reports/ruar_math_aime_results.csv}"
export REPORT_MD="${REPORT_MD:-$RUAR_ROOT/reports/ruar_math_aime_report.md}"
export REPORT_TITLE="${REPORT_TITLE:-RUAR MATH500/AIME2024 Evaluation}"
export BASE_MODEL_NOTE="${BASE_MODEL_NOTE:-RUAR is evaluated against the base HF model using greedy decoding.}"
export WRITE_PROGRESS_REPORT="${WRITE_PROGRESS_REPORT:-1}"
export ROLLOUT_N="${ROLLOUT_N:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-32768}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-34816}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-34816}"
export TOKEN_BUDGET="${TOKEN_BUDGET:-65536}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
export VALIDATE_SAMPLE="${VALIDATE_SAMPLE:-False}"

SBATCH_ARGS=(
    --parsable
    --job-name=eval_ruar_math_aime
    --output="$LOG_DIR/eval_ruar_math_aime_%j.out"
    --error="$LOG_DIR/eval_ruar_math_aime_%j.err"
)
if [[ -n "${SBATCH_PARTITION:-}" ]]; then
    SBATCH_ARGS+=(--partition="$SBATCH_PARTITION")
fi
if [[ -n "${SBATCH_NODELIST:-}" ]]; then
    SBATCH_ARGS+=(--nodelist="$SBATCH_NODELIST")
fi
if [[ -n "${SBATCH_TIME:-}" ]]; then
    SBATCH_ARGS+=(--time="$SBATCH_TIME")
fi
if [[ -n "${SBATCH_DEPENDENCY:-}" ]]; then
    SBATCH_ARGS+=(--dependency="$SBATCH_DEPENDENCY")
fi

job_id="$(sbatch "${SBATCH_ARGS[@]}" "$RUAR_ROOT/slurm/eval_ruar_grid.sbatch")"
echo "Submitted RUAR eval job=$job_id"
echo "Report: $REPORT_MD"
echo "Monitor: squeue -j ${job_id%%_*}"
