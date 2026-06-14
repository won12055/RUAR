# Installation

RUAR has two installation levels:

- Method utilities: enough to import `ruar`, run unit tests, inspect extracted
  reflective steps, compute answer-forcing utilities with injected callables,
  and test advantage rescaling.
- Full training: adds the distributed RL, vLLM, and Ray packages needed for
  Qwen3-8B 50-step reproduction.

## Method Utilities

```bash
cd RUAR
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install -r requirements.txt
```

Run the core checks:

```bash
python -m pytest tests
```

If `pytest` is not installed:

```bash
python -m pip install pytest
python -m pytest tests
```

## Full Qwen3-8B Training Environment

Use a Linux machine with CUDA GPUs. The paper-scale run uses two GPUs for the
Qwen3-8B configuration in `configs/ruar_qwen3_8b.env`. Slurm is assumed by the
provided shell launchers; adapt the launch command if your cluster uses another
scheduler.

Create an environment:

```bash
cd RUAR
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
```

Install a CUDA-compatible PyTorch build first. Choose the index URL that matches
your CUDA runtime; for example:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

Then install RUAR and the training dependencies:

```bash
python -m pip install -e .
python -m pip install -r requirements-train.txt
```

The paper-default launcher uses `USE_REMOVE_PADDING=True`, which requires a
`flash-attn` build compatible with your Python, PyTorch, and CUDA stack.
`flash-attn` is not pinned in `requirements-train.txt` because its binary
extension is environment-specific. Install a matching wheel/source build after
PyTorch, then verify the CUDA extension import:

```bash
# Example only; choose the flash-attn build that matches your environment.
python -m pip install flash-attn --no-build-isolation

python - <<'PY'
import flash_attn
import flash_attn_2_cuda
print("flash-attn import ok:", flash_attn.__file__)
PY
```

For a lightweight smoke run you may set `USE_REMOVE_PADDING=False` to use the
non-FlashAttention path. Paper-default Qwen3-8B reproduction should use
`USE_REMOVE_PADDING=True` with a working `flash-attn` installation.

Verify key packages:

```bash
python - <<'PY'
import importlib.metadata as m
for name in ["torch", "vllm", "transformers", "tokenizers", "ray", "hydra-core"]:
    print(name, m.version(name))
PY
```

The validated environment used:

```text
python 3.10
torch 2.10.0
vllm 0.17.1
transformers 4.57.6
tokenizers 0.22.2
ray 2.55.1
hydra-core 1.3.2
omegaconf 2.3.0
flash-attn 2.8.3
```

## Required Inputs for Training

Before launching Qwen3-8B training, prepare:

- `MODEL_PATH`: local path to the Qwen3-8B Hugging Face checkpoint.
- Training and validation parquet files for the math training split used by the
  launcher.
- Rows containing `prompt`, `data_source`, and `reward_info.ground_truth`.
  The included rule-based verifier handles the supported math data sources.
- Writable output directories for checkpoints, COT dumps, probe dumps, and logs.

The repository includes `scripts/prepare_numina3k.py` for preparing the
Numina-3K split from `AI-MO/NuminaMath-CoT` or from a local JSONL export. The
launcher can run it before submission when `PREPARE_TRAIN_DATA=1`.

Generated artifacts are intentionally ignored by `.gitignore`:

```text
checkpoints/
cot_dumps/
data/
logs/
outputs/
wandb/
```

## Required Inputs for Evaluation

The evaluation launcher included in this repository uses the same Python
environment as full training.

Prepare:

- `BASE_MODEL`: local path or model id for the base Qwen3-8B checkpoint.
- `CKPT_ROOT`: RUAR checkpoint root containing `global_step_50/actor`, or
  `RUAR_MODEL_PATH`: a merged Hugging Face checkpoint to evaluate directly.
- Benchmark JSONL files under `data/benchmarks/<dataset>/test.jsonl`, where the
  public MATH/AIME reproduction expects `math500` and `aime2024`.

The benchmark JSONL rows should contain a question field (`problem`,
`question`, `Question`, or `input`) and an answer field (`answer` or
`final_answer`). `scripts/prepare_eval_benchmark.py` converts these rows into
parquet files with `prompt`, `data_source`, and `reward_info.ground_truth`.

The default evaluation uses greedy single-response decoding and
`MAX_RESPONSE_LENGTH=32768`, matching the paper evaluation setting.

## Launch Defaults

`scripts/train_ruar_qwen3_8b.sh` loads `configs/ruar_qwen3_8b.env`
automatically unless `CONFIG_FILE` is set to another env file.

Launch the included Slurm training job:

```bash
export MODEL_PATH=/path/to/Qwen3-8B
# Optional: prepare data/numina_3k/train.parquet and val.parquet if missing.
export PREPARE_TRAIN_DATA=1
# Optional if the files are not under data/numina_3k/.
export TRAIN_FILE=/path/to/train.parquet
export VAL_FILE=/path/to/val.parquet
bash scripts/train_ruar_qwen3_8b.sh
```

Cluster-specific Slurm options can be supplied without editing the script:

```bash
SBATCH_PARTITION=gpu SBATCH_NODELIST=node01 bash scripts/train_ruar_qwen3_8b.sh
```

Use the same environment variables for evaluation, for example:

```bash
SBATCH_NODELIST=ubuntu bash scripts/eval_ruar_math_aime.sh
```

The Qwen3-8B defaults use:

- `TOTAL_STEPS=50`
- `ROLLOUT_N=16`
- `TRAIN_BATCH_SIZE=8`
- `MAX_RESPONSE_LENGTH=16384`
- `USE_REMOVE_PADDING=True`
- `REFLECTION_STOP_AFTER_READY=True`
- `ADVANTAGE_SCALING_READY_THRESHOLD=0.75`
- `ADVANTAGE_SCALING_CONSECUTIVE_REQUIRED=3`
- `ADVANTAGE_SCALING_POST_READY_DEFAULT_GAMMA_POS=0.25`
- `ADVANTAGE_SCALING_POST_READY_DEFAULT_GAMMA_NEG=1.25`
