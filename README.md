# RUAR: Utility-Guided Advantage Rescaling

This folder contains the reproduction-facing code for **WhenReflectionHelps:
Utility-Guided Advantage Rescaling for Efficient Reasoning**.

It keeps the RUAR-specific method code, the Qwen3-8B training entrypoint, and
paper-facing utilities in one package. Large artifacts such as checkpoints,
logs, and rollout dumps are intentionally kept outside the repository. The
small Numina-3K split and paper-aligned evaluation parquet files are included
under `data/`.

## Contents

- `ruar/reflective_step_extraction.py`: extracts Wait-start reflective reasoning
  steps. In the main setting, `Wait` starts target steps, while `Wait` and
  `Alternatively` determine endpoints.
- `ruar/answer_forcing_utility_estimation.py`: estimates before/after
  correct-answer probabilities and utilities, `u_j = p_j^+ - p_j^-`.
- `ruar/advantage_rescaling.py`: standalone implementation of utility-guided
  and post-ready advantage multipliers.
- `ruar_eval/metrics.py`: evaluation-only AES computation following the
  O1-Pruner reporting metric, relative to the base model.
- `ruar_training/`: the distributed training entrypoint used by the Qwen3-8B
  reproduction launcher.
- `verl/`: vendored training backend used by `ruar_training`.
- `scripts/train_ruar_qwen3_8b.sh`: submits the Qwen3-8B RUAR training run.
- `slurm/train_ruar_qwen3_8b.sbatch`: Slurm job body for Qwen3-8B training.
- `data/numina_3k/`: Numina-3K train/validation parquet files used by the
  Qwen3-8B launcher.
- `data/paper_eval/`: paper-aligned `test.parquet` files for the seven
  benchmark suite.
- `scripts/prepare_numina3k.py`: optional utility to rebuild the Numina-3K
  training parquet split.
- `scripts/eval_ruar_math_aime.sh`: evaluates base and RUAR checkpoints on
  MATH500 and AIME2024.
- `scripts/eval_ruar_7bench.sh`: evaluates the paper-style seven benchmark
  suite, including held-out MCQ prompts.
- `scripts/prepare_eval_benchmark.py`: converts local benchmark JSONL files to
  the RUAR parquet schema for evaluation.
- `scripts/export_fsdp_checkpoint.py`: exports a saved RUAR FSDP actor
  checkpoint to Hugging Face weights for evaluation.
- `scripts/summarize_ruar_results.py`: computes accuracy, token count, token
  reduction, and AES from eval CSV/COT dumps.
- `requirements-train.txt`: Python package set used for the full Qwen3-8B
  training environment.
- `docs/installation.md`: environment setup instructions for lightweight
  method utilities and full training reproduction.
- `paper/`: the submitted PDF and the copied RTF/LaTeX source text.

## Install

For method utilities:

```bash
cd RUAR
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install -r requirements.txt
```

For full Qwen3-8B training, install a CUDA-compatible PyTorch build first,
then:

```bash
python -m pip install -e .
python -m pip install -r requirements-train.txt
```

The paper default uses `USE_REMOVE_PADDING=True`, so install a `flash-attn`
build matching your Python/PyTorch/CUDA stack and verify
`import flash_attn, flash_attn_2_cuda`. For a smoke run only, set
`USE_REMOVE_PADDING=False`. See `docs/installation.md` for CUDA, vLLM, Ray, and
launch details.

## Main Hyperparameters

The launch scripts set the paper-aligned RUAR defaults:

- rollout count `N=16`, train batch size `8`, learning rate `2e-6`
- training max response length `16384`
- answer-forcing samples `K=4` through the training launcher
- answer-ready threshold `tau=0.75`, streak length `c=3`
- post-ready multipliers `gamma_post^+=0.25`, `gamma_post^-=1.25`
- length penalty coefficient `lambda_len=0.2`
- `REFLECTION_STOP_AFTER_READY=True`, matching the paper's answer-ready probing rule

## Training Environment

The Qwen3-8B launcher runs the training code included in this repository:
`scripts/train_ruar_qwen3_8b.sh` submits `slurm/train_ruar_qwen3_8b.sbatch`,
which starts `python -m ruar_training.main`.

Set the local model path before launching:

```bash
export MODEL_PATH=/path/to/Qwen3-8B
```

If `TRAIN_FILE` and `VAL_FILE` are not set, the launcher expects
`data/numina_3k/train.parquet` and `data/numina_3k/val.parquet` under this
repository. Training parquet rows should provide `prompt`, `data_source`, and
`reward_info.ground_truth`.

To rebuild that split from NuminaMath-CoT through the included script:

```bash
python scripts/prepare_numina3k.py --out-dir data/numina_3k
```

For offline use, pass a local JSONL export:

```bash
python scripts/prepare_numina3k.py \
  --input-jsonl /path/to/numina_math_cot_train.jsonl \
  --out-dir data/numina_3k
```

## Train

```bash
cd RUAR
export MODEL_PATH=/path/to/Qwen3-8B
bash scripts/train_ruar_qwen3_8b.sh
```

For dependency scheduling:

```bash
SBATCH_DEPENDENCY=afterok:<job_id> bash scripts/train_ruar_qwen3_8b.sh
```

## Evaluate

The paper-aligned evaluation `test.parquet` files are included under
`data/paper_eval/`. Then submit:

```bash
cd RUAR
export BASE_MODEL=/path/to/Qwen3-8B
export CKPT_ROOT=/path/to/ruar/checkpoint/root
bash scripts/eval_ruar_math_aime.sh
```

The evaluation launcher is self-contained within this repository. It reads
evaluation parquet files with `reward_info.ground_truth`, exports
`global_step_50/actor` to `actor/huggingface` when needed, and runs greedy
single-sample decoding with `MAX_RESPONSE_LENGTH=32768`.

```bash
bash scripts/eval_ruar_7bench.sh
```

After the eval job writes `reports/ruar_math_aime_results.csv`:

```bash
python scripts/summarize_ruar_results.py \
  --results-csv reports/ruar_math_aime_results.csv \
  --format md
```

## Paper Alignment

The public API uses the paper terms directly:

- `ReflectiveStep` for extracted reflective reasoning steps.
- `estimate_answer_forcing_utilities` for before/after answer-forcing probes.
- `compute_ruar_gammas` and `rescale_advantages` for sign-dependent utility
  and post-ready advantage rescaling.
- `find_answer_ready_step` for the earliest stable answer-ready boundary.
