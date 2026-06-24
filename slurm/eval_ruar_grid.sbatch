#!/bin/bash
#SBATCH --job-name=eval_ruar_grid
#SBATCH --output=logs/eval_ruar_grid_%j.out
#SBATCH --error=logs/eval_ruar_grid_%j.err
#SBATCH --time=99:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

set -euo pipefail

resolve_ruar_root() {
    if [[ -n "${RUAR_ROOT:-}" ]]; then
        cd "$RUAR_ROOT" && pwd
        return
    fi
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/slurm" && -d "$SLURM_SUBMIT_DIR/ruar_training" ]]; then
        cd "$SLURM_SUBMIT_DIR" && pwd
        return
    fi

    local script_root
    script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    if [[ -d "$script_root/slurm" && -d "$script_root/ruar_training" ]]; then
        echo "$script_root"
        return
    fi

    echo "Could not locate RUAR_ROOT. Submit from the RUAR directory or set RUAR_ROOT=/path/to/RUAR." >&2
    return 2
}

RUAR_ROOT="$(resolve_ruar_root)" || exit 2
export RUAR_ROOT
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

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN=python
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
    else
        echo "Missing Python executable: set PYTHON_BIN or activate RUAR_VENV." >&2
        exit 2
    fi
fi

RUN_TAG="${RUN_TAG:-ruar_qwen3_8b_numina_3k_50step}"
CKPT_COND="${CKPT_COND:-ruar_wait_rloo}"
CKPT_ROOT="${CKPT_ROOT:-$RUAR_ROOT/checkpoints/ruar/numina_3k/$RUN_TAG/$CKPT_COND}"
RUAR_STEP="${RUAR_STEP:-50}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
BASE_MODEL_LABEL="${BASE_MODEL_LABEL:-base}"
RUAR_MODEL_LABEL="${RUAR_MODEL_LABEL:-ruar}"
INCLUDE_BASE_MODEL="${INCLUDE_BASE_MODEL:-1}"
INCLUDE_RUAR_MODEL="${INCLUDE_RUAR_MODEL:-1}"
EXPORT_RUAR_CHECKPOINT="${EXPORT_RUAR_CHECKPOINT:-1}"

EVAL_COND_PREFIX="${EVAL_COND_PREFIX:-ruar_qwen3_8b}"
EVAL_PROJECT_NAME="${EVAL_PROJECT_NAME:-ruar_eval}"
EVAL_ROOT="${EVAL_ROOT:-$RUAR_ROOT/cot_dumps/eval_math_aime}"
DATA_ROOT="${DATA_ROOT:-$RUAR_ROOT/data/paper_eval}"
BENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR:-$RUAR_ROOT/data/benchmarks}"
CKPT_EVAL_ROOT="${CKPT_EVAL_ROOT:-$RUAR_ROOT/checkpoints/eval/math_aime}"
RESULT_CSV="${RESULT_CSV:-$RUAR_ROOT/reports/ruar_math_aime_results.csv}"
REPORT_MD="${REPORT_MD:-$RUAR_ROOT/reports/ruar_math_aime_report.md}"
REPORT_TITLE="${REPORT_TITLE:-RUAR MATH500/AIME2024 Evaluation}"
BASE_MODEL_NOTE="${BASE_MODEL_NOTE:-RUAR is evaluated against the base model using greedy decoding.}"

ROLLOUT_N="${ROLLOUT_N:-1}"
VAL_SIZE="${VAL_SIZE:--1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-32768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-34816}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-34816}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
TOKEN_BUDGET="${TOKEN_BUDGET:-65536}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-True}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-False}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"
VALIDATE_SAMPLE="${VALIDATE_SAMPLE:-False}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
WRITE_PROGRESS_REPORT="${WRITE_PROGRESS_REPORT:-1}"
MODEL_FILTER="${MODEL_FILTER:-}"
DATASET_FILTER="${DATASET_FILTER:-math500,aime2024}"
DATASET_SPECS="${DATASET_SPECS:-math500|math500|boxed_math/math500,aime2024|aime2024|boxed_math/aime2024}"
export REPORT_TITLE BASE_MODEL_NOTE DATASET_FILTER

mkdir -p "$RUAR_ROOT/logs" "$EVAL_ROOT" "$DATA_ROOT" "$CKPT_EVAL_ROOT" "$RUAR_ROOT/reports"

csv_filter_matches() {
    local value="$1"
    local filter_csv="$2"
    if [[ -z "$filter_csv" ]]; then
        return 0
    fi

    local item
    IFS=',' read -r -a items <<< "$filter_csv"
    for item in "${items[@]}"; do
        if [[ "$item" == "$value" ]]; then
            return 0
        fi
    done
    return 1
}

has_hf_weights() {
    local hf_dir="$1"
    compgen -G "$hf_dir/model*.safetensors" >/dev/null || [[ -f "$hf_dir/pytorch_model.bin" ]]
}

resolve_ruar_model() {
    if [[ -n "${RUAR_MODEL_PATH:-}" ]]; then
        echo "$RUAR_MODEL_PATH"
        return 0
    fi

    local actor_dir="$CKPT_ROOT/global_step_${RUAR_STEP}/actor"
    local hf_dir="$actor_dir/huggingface"
    if [[ ! -d "$actor_dir" ]]; then
        echo "Missing RUAR actor checkpoint: $actor_dir" >&2
        echo "Set RUAR_MODEL_PATH to a merged Hugging Face checkpoint, or set CKPT_ROOT/RUAR_STEP." >&2
        exit 2
    fi

    if has_hf_weights "$hf_dir"; then
        echo "$hf_dir"
        return 0
    fi

    if [[ "$EXPORT_RUAR_CHECKPOINT" == "1" ]]; then
        "$PYTHON_BIN" "$RUAR_ROOT/scripts/export_fsdp_checkpoint.py" --local-dir "$actor_dir" >&2
        if has_hf_weights "$hf_dir"; then
            echo "$hf_dir"
            return 0
        fi
    fi

    echo "Missing merged Hugging Face weights under $hf_dir" >&2
    echo "Run: python scripts/export_fsdp_checkpoint.py --local-dir $actor_dir" >&2
    exit 2
}

prepare_dataset() {
    local dataset_key="$1"
    local data_source="$2"
    local data_dir="$DATA_ROOT/$dataset_key"
    if [[ -f "$data_dir/test.parquet" ]]; then
        echo "[data] Using prepared benchmark: $data_dir"
        return 0
    fi

    mkdir -p "$data_dir"
    "$PYTHON_BIN" "$RUAR_ROOT/scripts/prepare_eval_benchmark.py" \
        --dataset "$dataset_key" \
        --data-source "$data_source" \
        --data-dir "$BENCHMARK_DATA_DIR" \
        --train-size 1 \
        --val-size "$VAL_SIZE" \
        --out-dir "$data_dir"
}

append_result() {
    local model_label="$1"
    local dataset_label="$2"
    local cot_path="$3"
    "$PYTHON_BIN" - "$RESULT_CSV" "$model_label" "$dataset_label" "$cot_path" <<'PY'
import csv
import json
import os
import sys

csv_path, model_label, dataset_label, cot_path = sys.argv[1:5]
scores = []
with open(cot_path, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        record = json.loads(line)
        score = record.get("score_raw")
        if score is None:
            score = record.get("acc")
        if score is None:
            score = record.get("score")
        if score is not None:
            scores.append(float(score))
if not scores:
    raise RuntimeError(f"No scores found in {cot_path}")

rows = []
if os.path.exists(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
rows = [r for r in rows if not (r["model"] == model_label and r["dataset"] == dataset_label)]
rows.append({
    "model": model_label,
    "dataset": dataset_label,
    "score": f"{sum(scores) / len(scores):.10f}",
    "n": str(len(scores)),
    "cot_dump": cot_path,
})
fieldnames = ["model", "dataset", "score", "n", "cot_dump"]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)
print(f"[summary] {model_label} {dataset_label}: score={sum(scores) / len(scores):.6f} n={len(scores)}")
PY
}

write_report() {
    if [[ "$WRITE_PROGRESS_REPORT" != "1" ]]; then
        return 0
    fi
    "$PYTHON_BIN" - "$RESULT_CSV" "$REPORT_MD" "$RUN_TAG" "$ROLLOUT_N" "$VAL_SIZE" "$MAX_PROMPT_LENGTH" "$MAX_RESPONSE_LENGTH" "$MAX_MODEL_LEN" "$MAX_NUM_BATCHED_TOKENS" "$MAX_NUM_SEQS" "$TOKEN_BUDGET" "$ROLLOUT_TEMPERATURE" "$ROLLOUT_TOP_P" "$VALIDATE_SAMPLE" "${SLURM_JOB_ID:-unknown}" <<'PY'
import csv
import datetime as dt
import os
import sys

(
    csv_path,
    md_path,
    run_tag,
    rollout_n,
    val_size,
    max_prompt_length,
    max_response_length,
    max_model_len,
    max_num_batched_tokens,
    max_num_seqs,
    token_budget,
    rollout_temperature,
    rollout_top_p,
    validate_sample,
    job_id,
) = sys.argv[1:16]

rows = []
if os.path.exists(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
order_model = {"base": 0, "ruar": 1}
order_dataset = {
    "gsm8k": 0,
    "math500": 1,
    "aime2024": 2,
    "hmmt25": 3,
    "gpqa_diamond": 4,
    "arc_challenge": 5,
    "commonsenseqa": 6,
}
rows.sort(key=lambda r: (order_model.get(r["model"], 99), order_dataset.get(r["dataset"], 99)))

with open(md_path, "w", encoding="utf-8") as f:
    f.write(f"# {os.environ.get('REPORT_TITLE', 'RUAR Evaluation')}\n\n")
    f.write(f"- Updated: {dt.datetime.now().isoformat(timespec='seconds')}\n")
    f.write(f"- Run tag: `{run_tag}`\n")
    f.write(f"- Eval mode: `ROLLOUT_N={rollout_n}`, `VAL_SIZE={val_size}`\n")
    f.write("- Decoding: greedy single response per problem, `temperature=0.0`, `top_p=1.0`, `validate_sample=False`\n")
    f.write("- Scoring: boxed final-answer extraction for math; loose letter accuracy for held-out MCQ.\n")
    f.write(f"- Max prompt length: `{max_prompt_length}`\n")
    f.write(f"- Max response length: `{max_response_length}`\n")
    f.write(f"- vLLM max seq len: `{max_model_len}`, max batched tokens: `{max_num_batched_tokens}`\n")
    f.write(f"- Fixed wrapper batching: `max_num_seqs={max_num_seqs}`, `token_budget={token_budget}`\n")
    dataset_filter = os.environ.get("DATASET_FILTER")
    if dataset_filter:
        f.write(f"- Dataset filter: `{dataset_filter}`\n")
    f.write(f"- Slurm job: `{job_id}`\n\n")
    f.write("| model | dataset | score | n | cot dump |\n")
    f.write("|---|---:|---:|---:|---|\n")
    for row in rows:
        f.write(
            f"| {row['model']} | {row['dataset']} | {float(row['score']):.6f} | "
            f"{row['n']} | `{row['cot_dump']}` |\n"
        )
    if not rows:
        f.write("| pending | pending | pending | pending | pending |\n")
    f.write("\n## Notes\n\n")
    f.write("- `score` is the mean correctness over the COT dump.\n")
    f.write(f"- {os.environ.get('BASE_MODEL_NOTE', 'The base model is evaluated with the same decoding configuration.')}\n")
    f.write("- Do not mix rows generated with different max-response or decoding settings in one report.\n")
PY
}

run_eval() {
    local model_label="$1"
    local model_path="$2"
    local dataset_key="$3"
    local dataset_label="$4"
    local data_source="$5"

    prepare_dataset "$dataset_key" "$data_source"
    local data_dir="$DATA_ROOT/$dataset_key"
    local cond="${EVAL_COND_PREFIX}_${model_label}_${dataset_label}_n${ROLLOUT_N}"
    local cot_dir="$EVAL_ROOT/$dataset_label"
    local cot_path="$cot_dir/${cond}.jsonl"
    local eval_ckpt_dir="$CKPT_EVAL_ROOT/$cond"
    mkdir -p "$cot_dir" "$eval_ckpt_dir"

    if [[ "$SKIP_EXISTING" == "1" && -s "$cot_path" ]]; then
        echo "[eval] SKIP existing model=$model_label dataset=$dataset_label cot_dump=$cot_path"
        append_result "$model_label" "$dataset_label" "$cot_path"
        write_report
        return 0
    fi

    echo "[eval] START model=$model_label dataset=$dataset_label model_path=$model_path"
    "$PYTHON_BIN" -m ruar_training.main \
        data.train_files="$data_dir/test.parquet" \
        data.val_files="$data_dir/test.parquet" \
        data.train_batch_size=1 \
        data.max_prompt_length="$MAX_PROMPT_LENGTH" \
        data.max_response_length="$MAX_RESPONSE_LENGTH" \
        data.shuffle=False \
        data.truncation=right \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path="$model_path" \
        actor_rollout_ref.model.use_remove_padding="$USE_REMOVE_PADDING" \
        actor_rollout_ref.model.enable_gradient_checkpointing=False \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=1 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$TOKEN_BUDGET" \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size="$ULYSSES_SEQUENCE_PARALLEL_SIZE" \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n="$ROLLOUT_N" \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
        actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS" \
        actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
        actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
        actor_rollout_ref.rollout.enable_chunked_prefill="$ENABLE_CHUNKED_PREFILL" \
        actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
        actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P" \
        actor_rollout_ref.rollout.top_k="$ROLLOUT_TOP_K" \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$TOKEN_BUDGET" \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size="$ULYSSES_SEQUENCE_PARALLEL_SIZE" \
        algorithm.adv_estimator=rloo \
        trainer.val_before_train=True \
        trainer.val_only=True \
        trainer.validate_sample="$VALIDATE_SAMPLE" \
        trainer.logger=['console'] \
        trainer.project_name="$EVAL_PROJECT_NAME" \
        trainer.experiment_name="$cond" \
        trainer.n_gpus_per_node=1 \
        trainer.nnodes=1 \
        trainer.default_local_dir="$eval_ckpt_dir" \
        trainer.resume_mode=disable \
        trainer.run_final_validation=False \
        trainer.cot_dump.enable=True \
        trainer.cot_dump.path="$cot_path" \
        trainer.cot_dump.condition="$cond" \
        trainer.cot_dump.every_n_steps=1 \
        trainer.cot_dump.max_samples_per_step=0 \
        trainer.cot_dump.max_val_samples_per_step=0 \
        trainer.cot_dump.include_prompt=True \
        trainer.cot_dump.overwrite=True \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_training_steps=1

    append_result "$model_label" "$dataset_label" "$cot_path"
    write_report
    echo "[eval] DONE model=$model_label dataset=$dataset_label"
}

write_report

declare -a MODELS=()
if [[ "$INCLUDE_BASE_MODEL" == "1" ]]; then
    MODELS+=("$BASE_MODEL_LABEL|$BASE_MODEL")
fi
if [[ "$INCLUDE_RUAR_MODEL" == "1" ]]; then
    RUAR_MODEL_RESOLVED="$(resolve_ruar_model)"
    MODELS+=("$RUAR_MODEL_LABEL|$RUAR_MODEL_RESOLVED")
fi

IFS=',' read -r -a DATASETS <<< "$DATASET_SPECS"

for model_spec in "${MODELS[@]}"; do
    IFS='|' read -r model_label model_path <<< "$model_spec"
    if ! csv_filter_matches "$model_label" "$MODEL_FILTER"; then
        continue
    fi
    for dataset_spec in "${DATASETS[@]}"; do
        IFS='|' read -r dataset_key dataset_label data_source <<< "$dataset_spec"
        if ! csv_filter_matches "$dataset_label" "$DATASET_FILTER"; then
            continue
        fi
        run_eval "$model_label" "$model_path" "$dataset_key" "$dataset_label" "$data_source"
    done
done

echo "[eval] Grid complete. Report: $REPORT_MD"
