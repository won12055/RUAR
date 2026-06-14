#!/usr/bin/env bash
# Submit the Qwen3-8B RUAR training run from the paper configuration.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$ROOT/configs/ruar_qwen3_8b.env}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-$ROOT/slurm/train_ruar_qwen3_8b.sbatch}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"

if [[ -f "$CONFIG_FILE" ]]; then
    declare -A RUAR_ENV_BEFORE_CONFIG=()
    while IFS='=' read -r name _; do
        if [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
            RUAR_ENV_BEFORE_CONFIG["$name"]="${!name}"
        fi
    done < <(env)

    set -a
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
    set +a

    # Keep values explicitly passed through the environment above config defaults.
    for name in "${!RUAR_ENV_BEFORE_CONFIG[@]}"; do
        export "$name=${RUAR_ENV_BEFORE_CONFIG[$name]}"
    done
    unset RUAR_ENV_BEFORE_CONFIG
fi

: "${MODEL_PATH:?Set MODEL_PATH to the local Qwen3-8B checkpoint or model id.}"
export TOTAL_STEPS="${TOTAL_STEPS:-50}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export TEST_FREQ="${TEST_FREQ:--1}"
export RUN_FINAL_VALIDATION="${RUN_FINAL_VALIDATION:-False}"
export REFLECTION_STOP_AFTER_READY="${REFLECTION_STOP_AFTER_READY:-True}"
export RUN_SUFFIX="${RUN_SUFFIX:-qwen3_8b_2gpu_train8_step${TOTAL_STEPS}_waitstart_answerready_lp02_m18k_r16k_mem030}"
export RUAR_ROOT="$ROOT"

mkdir -p "$LOG_DIR"

DEFAULT_DATA_DIR="$ROOT/data/${DATASET_TAG:-numina_3k}"
TRAIN_FILE="${TRAIN_FILE:-$DEFAULT_DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$DEFAULT_DATA_DIR/val.parquet}"
export TRAIN_FILE VAL_FILE

if [[ "${PREPARE_TRAIN_DATA:-0}" == "1" && ( ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ) ]]; then
    PREPARE_ARGS=(
        "$ROOT/scripts/prepare_numina3k.py"
        --out-dir "$DEFAULT_DATA_DIR"
        --train-size "${NUMINA_TRAIN_SIZE:-3200}"
        --val-size "${NUMINA_VAL_SIZE:-512}"
        --seed "${NUMINA_SEED:-42}"
    )
    if [[ -n "${NUMINA_INPUT_JSONL:-}" ]]; then
        PREPARE_ARGS+=(--input-jsonl "$NUMINA_INPUT_JSONL")
    fi
    if [[ -n "${NUMINA_HF_REPO:-}" ]]; then
        PREPARE_ARGS+=(--hf-repo "$NUMINA_HF_REPO")
    fi
    if [[ -n "${NUMINA_HF_REVISION:-}" ]]; then
        PREPARE_ARGS+=(--hf-revision "$NUMINA_HF_REVISION")
    fi
    python "${PREPARE_ARGS[@]}"
fi

SBATCH_ARGS=(
    --parsable
    --array=0
    --gres=gpu:"${N_GPUS:-2}"
    --cpus-per-task="${CPUS_PER_TASK:-16}"
    --mem="${MEM:-256G}"
    --job-name="ruar_qwen3_8b_${RUN_SUFFIX}"
    --output="$LOG_DIR/ruar_qwen3_8b_${RUN_SUFFIX}_%j.out"
    --error="$LOG_DIR/ruar_qwen3_8b_${RUN_SUFFIX}_%j.err"
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

job_id=$(sbatch "${SBATCH_ARGS[@]}" "$SBATCH_SCRIPT")

echo "Submitted RUAR Qwen3-8B job=$job_id"
echo "Run suffix: $RUN_SUFFIX"
echo "Monitor:"
echo "  squeue -j ${job_id%%_*}"
echo "  tail -f $LOG_DIR/ruar_qwen3_8b_${RUN_SUFFIX}_${job_id%%_*}.out"
