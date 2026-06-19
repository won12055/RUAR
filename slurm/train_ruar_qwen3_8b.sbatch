#!/bin/bash
#SBATCH --job-name=ruar_qwen3_8b
#SBATCH --output=logs/ruar_qwen3_8b_%A_%a.out
#SBATCH --error=logs/ruar_qwen3_8b_%A_%a.err
#SBATCH --time=30-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --array=0

set -euo pipefail

RUAR_ROOT="${RUAR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$RUAR_ROOT"

if [[ -n "${RUAR_VENV:-}" ]]; then
    # shellcheck disable=SC1090
    source "$RUAR_VENV/bin/activate"
elif [[ -f "$RUAR_ROOT/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$RUAR_ROOT/.venv/bin/activate"
fi

export PYTHONPATH="$RUAR_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export VERL_DISABLE_TORCH_COMPILE="${VERL_DISABLE_TORCH_COMPILE:-1}"
export VERL_DISABLE_FLASH_ATTN_CE="${VERL_DISABLE_FLASH_ATTN_CE:-1}"
export VERL_ATTENTION_IMPLEMENTATION="${VERL_ATTENTION_IMPLEMENTATION:-sdpa}"

DATASET_TAG="${DATASET_TAG:-numina_3k}"
RUN_CONDITION="${RUN_CONDITION:-ruar_wait_rloo}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local Qwen3-8B checkpoint or model id.}"
TRAIN_FILE="${TRAIN_FILE:-$RUAR_ROOT/data/$DATASET_TAG/train.parquet}"
VAL_FILE="${VAL_FILE:-$RUAR_ROOT/data/$DATASET_TAG/val.parquet}"

if [[ "$MODEL_PATH" == /* || "$MODEL_PATH" == ./* || "$MODEL_PATH" == ../* ]]; then
    if [[ ! -d "$MODEL_PATH" ]]; then
        echo "Missing local MODEL_PATH directory: $MODEL_PATH" >&2
        exit 2
    fi
fi
if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "Missing TRAIN_FILE=$TRAIN_FILE" >&2
    exit 2
fi
if [[ ! -f "$VAL_FILE" ]]; then
    echo "Missing VAL_FILE=$VAL_FILE" >&2
    exit 2
fi

TOTAL_STEPS="${TOTAL_STEPS:-50}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:--1}"
RUN_FINAL_VALIDATION="${RUN_FINAL_VALIDATION:-False}"
RUN_SUFFIX="${RUN_SUFFIX:-qwen3_8b_2gpu_train8_step${TOTAL_STEPS}_waitstart_answerready_lp02_m18k_r16k_mem030}"
RUN_TAG="${RUN_TAG:-ruar_qwen3_8b_${DATASET_TAG}_${RUN_CONDITION}_${RUN_SUFFIX}}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-18432}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-18432}"
TOKEN_BUDGET="${TOKEN_BUDGET:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.30}"
RUAR_VLLM_ENABLE_SLEEP_MODE="${RUAR_VLLM_ENABLE_SLEEP_MODE:-True}"
RUAR_VLLM_INITIAL_SLEEP="${RUAR_VLLM_INITIAL_SLEEP:-True}"
N_GPUS="${N_GPUS:-${SLURM_GPUS_ON_NODE:-2}}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-$N_GPUS}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"

if [[ "$USE_REMOVE_PADDING" == "True" ]]; then
    export VERL_DISABLE_FLASH_ATTN_CE=0
    export VERL_ATTENTION_IMPLEMENTATION=flash_attention_2
fi

ROLLOUT_N="${ROLLOUT_N:-16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
ACTOR_LR="${ACTOR_LR:-2e-6}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"

LENGTH_PENALTY_ENABLE="${LENGTH_PENALTY_ENABLE:-True}"
LENGTH_PENALTY_COEF="${LENGTH_PENALTY_COEF:-0.2}"
LENGTH_PENALTY_TARGET="${LENGTH_PENALTY_TARGET:-8192}"
LENGTH_PENALTY_MODE="${LENGTH_PENALTY_MODE:-correct_group_sigmoid}"
LENGTH_PENALTY_CAP="${LENGTH_PENALTY_CAP:-1.0}"
LENGTH_PENALTY_STD_FLOOR="${LENGTH_PENALTY_STD_FLOOR:-0.0}"
LENGTH_PENALTY_ONLY_CORRECT="${LENGTH_PENALTY_ONLY_CORRECT:-True}"

ADVANTAGE_SCALING_ENABLE="${ADVANTAGE_SCALING_ENABLE:-True}"
ADVANTAGE_SCALING_ONLY_CORRECT="${ADVANTAGE_SCALING_ONLY_CORRECT:-False}"
ADVANTAGE_SCALING_READY_THRESHOLD="${ADVANTAGE_SCALING_READY_THRESHOLD:-0.75}"
ADVANTAGE_SCALING_CONSECUTIVE_REQUIRED="${ADVANTAGE_SCALING_CONSECUTIVE_REQUIRED:-3}"
ADVANTAGE_SCALING_SCOPE="${ADVANTAGE_SCALING_SCOPE:-all_post_ready}"
ADVANTAGE_SCALING_EXCLUDE_FINAL_ANSWER="${ADVANTAGE_SCALING_EXCLUDE_FINAL_ANSWER:-True}"
ADVANTAGE_SCALING_POST_READY_DEFAULT_GAMMA_POS="${ADVANTAGE_SCALING_POST_READY_DEFAULT_GAMMA_POS:-0.25}"
ADVANTAGE_SCALING_POST_READY_DEFAULT_GAMMA_NEG="${ADVANTAGE_SCALING_POST_READY_DEFAULT_GAMMA_NEG:-1.25}"
ADVANTAGE_SCALING_POST_READY_POSITIVE_GAMMA="${ADVANTAGE_SCALING_POST_READY_POSITIVE_GAMMA:-0.2}"
ADVANTAGE_SCALING_POST_READY_NEGATIVE_GAMMA="${ADVANTAGE_SCALING_POST_READY_NEGATIVE_GAMMA:-1.0}"
ADVANTAGE_SCALING_PRE_READY_NEGATIVE_GAMMA="${ADVANTAGE_SCALING_PRE_READY_NEGATIVE_GAMMA:-1.0}"
ADVANTAGE_SCALING_APPLY_TO_ALL_WAIT="${ADVANTAGE_SCALING_APPLY_TO_ALL_WAIT:-True}"

REFLECTION_STOP_AFTER_READY="${REFLECTION_STOP_AFTER_READY:-True}"
# 0 means no cap: use all extracted Wait-start spans until
# REFLECTION_STOP_AFTER_READY stops further answer-forcing probes.
REFLECTION_MAX_SPANS_PER_SAMPLE="${REFLECTION_MAX_SPANS_PER_SAMPLE:-0}"
REFLECTION_SPAN_SELECTION="${REFLECTION_SPAN_SELECTION:-first}"
REFLECTION_MAX_ANSWER_TOKENS="${REFLECTION_MAX_ANSWER_TOKENS:-32}"
REFLECTION_LABEL_FILTER="${REFLECTION_LABEL_FILTER:-signed}"
REFLECTION_LABEL_THRESHOLD="${REFLECTION_LABEL_THRESHOLD:-0.0}"
REFLECTION_LENGTH_AWARE_GAMMA="${REFLECTION_LENGTH_AWARE_GAMMA:-0.0}"
REFLECTION_LENGTH_AWARE_TARGET="${REFLECTION_LENGTH_AWARE_TARGET:-128}"
REFLECTION_LENGTH_AWARE_SCOPE="${REFLECTION_LENGTH_AWARE_SCOPE:-response}"
REFLECTION_POST_ANSWER_MODE="${REFLECTION_POST_ANSWER_MODE:-off}"
REFLECTION_POST_ANSWER_PENALTY="${REFLECTION_POST_ANSWER_PENALTY:-0.0}"
REFLECTION_POST_ANSWER_THRESHOLD="${REFLECTION_POST_ANSWER_THRESHOLD:-1.0}"
REFLECTION_CLASS_THRESHOLD="${REFLECTION_CLASS_THRESHOLD:-0.75}"
PROBE_PROMPT_MARGIN="${PROBE_PROMPT_MARGIN:-16}"
MAX_PROBE_PROMPT_LENGTH="${MAX_PROBE_PROMPT_LENGTH:-$((MAX_MODEL_LEN - REFLECTION_MAX_ANSWER_TOKENS - PROBE_PROMPT_MARGIN))}"

REWARD_GT_COEF="${REWARD_GT_COEF:-1}"
ACTOR_USE_KL_LOSS="${ACTOR_USE_KL_LOSS:-False}"

COT_DUMP_ENABLE="${COT_DUMP_ENABLE:-True}"
COT_DUMP_EVERY="${COT_DUMP_EVERY:-1}"
COT_DUMP_MAX_TRAIN_SAMPLES="${COT_DUMP_MAX_TRAIN_SAMPLES:-0}"
COT_DUMP_MAX_VAL_SAMPLES="${COT_DUMP_MAX_VAL_SAMPLES:-0}"
COT_DUMP_DIR="${COT_DUMP_DIR:-$RUAR_ROOT/cot_dumps/$RUN_TAG/$DATASET_TAG}"

CKPT_DIR="${CKPT_DIR:-$RUAR_ROOT/checkpoints/ruar/$DATASET_TAG/$RUN_TAG/$RUN_CONDITION}"
mkdir -p "$RUAR_ROOT/logs" "$COT_DUMP_DIR" "$CKPT_DIR"

echo "[RUAR] dataset=$DATASET_TAG condition=$RUN_CONDITION steps=$TOTAL_STEPS model=$MODEL_PATH"
echo "[RUAR] train_file=$TRAIN_FILE val_file=$VAL_FILE"
echo "[RUAR] prompt=$MAX_PROMPT_LENGTH response=$MAX_RESPONSE_LENGTH model_len=$MAX_MODEL_LEN rollout_n=$ROLLOUT_N"
echo "[RUAR] length_penalty=$LENGTH_PENALTY_ENABLE coef=$LENGTH_PENALTY_COEF target=$LENGTH_PENALTY_TARGET mode=$LENGTH_PENALTY_MODE cap=$LENGTH_PENALTY_CAP std_floor=$LENGTH_PENALTY_STD_FLOOR only_correct=$LENGTH_PENALTY_ONLY_CORRECT"
echo "[RUAR] advantage_scaling=$ADVANTAGE_SCALING_ENABLE ready_threshold=$ADVANTAGE_SCALING_READY_THRESHOLD consecutive=$ADVANTAGE_SCALING_CONSECUTIVE_REQUIRED scope=$ADVANTAGE_SCALING_SCOPE"
echo "[RUAR] stop_after_ready=$REFLECTION_STOP_AFTER_READY max_probe_prompt=$MAX_PROBE_PROMPT_LENGTH"

python - <<'PY'
from importlib.metadata import version
import torch

for package in ["torch", "transformers", "tokenizers", "vllm"]:
    print(package, version(package))
print("cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY

python -m ruar_training.main \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
    actor_rollout_ref.model.use_remove_padding="$USE_REMOVE_PADDING" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.actor.use_dynamic_bsz="$USE_DYNAMIC_BSZ" \
    actor_rollout_ref.actor.use_kl_loss="$ACTOR_USE_KL_LOSS" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$TOKEN_BUDGET" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="$ULYSSES_SEQUENCE_PARALLEL_SIZE" \
    actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_OPTIMIZER_OFFLOAD" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$TOKEN_BUDGET" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$N_GPUS" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_sleep_mode="$RUAR_VLLM_ENABLE_SLEEP_MODE" \
    actor_rollout_ref.rollout.initial_sleep="$RUAR_VLLM_INITIAL_SLEEP" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
    actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$TOKEN_BUDGET" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="$ULYSSES_SEQUENCE_PARALLEL_SIZE" \
    actor_rollout_ref.ref.fsdp_config.param_offload="$REF_PARAM_OFFLOAD" \
    algorithm.adv_estimator=rloo \
    algorithm.reward_gt_coef="$REWARD_GT_COEF" \
    algorithm.length_penalty.enable="$LENGTH_PENALTY_ENABLE" \
    algorithm.length_penalty.coef="$LENGTH_PENALTY_COEF" \
    algorithm.length_penalty.target="$LENGTH_PENALTY_TARGET" \
    algorithm.length_penalty.mode="$LENGTH_PENALTY_MODE" \
    algorithm.length_penalty.cap="$LENGTH_PENALTY_CAP" \
    algorithm.length_penalty.std_floor="$LENGTH_PENALTY_STD_FLOOR" \
    algorithm.length_penalty.only_correct="$LENGTH_PENALTY_ONLY_CORRECT" \
    algorithm.advantage_scaling.enable="$ADVANTAGE_SCALING_ENABLE" \
    algorithm.advantage_scaling.only_correct="$ADVANTAGE_SCALING_ONLY_CORRECT" \
    algorithm.advantage_scaling.ready_threshold="$ADVANTAGE_SCALING_READY_THRESHOLD" \
    algorithm.advantage_scaling.consecutive_required="$ADVANTAGE_SCALING_CONSECUTIVE_REQUIRED" \
    algorithm.advantage_scaling.scope="$ADVANTAGE_SCALING_SCOPE" \
    algorithm.advantage_scaling.exclude_final_answer="$ADVANTAGE_SCALING_EXCLUDE_FINAL_ANSWER" \
    algorithm.advantage_scaling.post_ready_default_gamma_pos="$ADVANTAGE_SCALING_POST_READY_DEFAULT_GAMMA_POS" \
    algorithm.advantage_scaling.post_ready_default_gamma_neg="$ADVANTAGE_SCALING_POST_READY_DEFAULT_GAMMA_NEG" \
    algorithm.advantage_scaling.post_ready_positive_gamma="$ADVANTAGE_SCALING_POST_READY_POSITIVE_GAMMA" \
    algorithm.advantage_scaling.post_ready_negative_gamma="$ADVANTAGE_SCALING_POST_READY_NEGATIVE_GAMMA" \
    algorithm.advantage_scaling.pre_ready_negative_gamma="$ADVANTAGE_SCALING_PRE_READY_NEGATIVE_GAMMA" \
    algorithm.advantage_scaling.apply_to_all_wait="$ADVANTAGE_SCALING_APPLY_TO_ALL_WAIT" \
    reflection.enable=True \
    reflection.cue_types="['wait']" \
    reflection.answer_rollouts=4 \
    reflection.max_spans_per_sample="$REFLECTION_MAX_SPANS_PER_SAMPLE" \
    reflection.span_selection="$REFLECTION_SPAN_SELECTION" \
    reflection.max_answer_tokens="$REFLECTION_MAX_ANSWER_TOKENS" \
    reflection.max_probe_prompt_length="$MAX_PROBE_PROMPT_LENGTH" \
    reflection.label_filter="$REFLECTION_LABEL_FILTER" \
    reflection.label_threshold="$REFLECTION_LABEL_THRESHOLD" \
    reflection.length_aware_gamma="$REFLECTION_LENGTH_AWARE_GAMMA" \
    reflection.length_aware_target="$REFLECTION_LENGTH_AWARE_TARGET" \
    reflection.length_aware_scope="$REFLECTION_LENGTH_AWARE_SCOPE" \
    reflection.post_answer_mode="$REFLECTION_POST_ANSWER_MODE" \
    reflection.post_answer_penalty="$REFLECTION_POST_ANSWER_PENALTY" \
    reflection.post_answer_threshold="$REFLECTION_POST_ANSWER_THRESHOLD" \
    reflection.class_threshold="$REFLECTION_CLASS_THRESHOLD" \
    reflection.stop_after_ready="$REFLECTION_STOP_AFTER_READY" \
    reflection.verify_num_processes=8 \
    trainer.val_before_train=False \
    trainer.run_final_validation="$RUN_FINAL_VALIDATION" \
    trainer.logger=['console'] \
    trainer.project_name=ruar \
    trainer.experiment_name="$RUN_TAG" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.resume_mode="${RESUME_MODE:-disable}" \
    trainer.resume_from_path="${RESUME_FROM_PATH:-False}" \
    trainer.cot_dump.enable="$COT_DUMP_ENABLE" \
    trainer.cot_dump.path="$COT_DUMP_DIR/${RUN_CONDITION}.jsonl" \
    trainer.cot_dump.condition="$RUN_CONDITION" \
    trainer.cot_dump.every_n_steps="$COT_DUMP_EVERY" \
    trainer.cot_dump.max_samples_per_step="$COT_DUMP_MAX_TRAIN_SAMPLES" \
    trainer.cot_dump.max_val_samples_per_step="$COT_DUMP_MAX_VAL_SAMPLES" \
    trainer.cot_dump.include_prompt=True \
    trainer.cot_dump.overwrite=True \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_training_steps="$TOTAL_STEPS"

echo "[RUAR] Done condition=$RUN_CONDITION"
