# RUAR: Utility-Guided Advantage Rescaling

This folder contains the reproduction-facing code for **When Reflection Helps:
Utility-Guided Advantage Rescaling for Efficient Reasoning**.

It keeps the RUAR-specific method code, the DS-7B default training entrypoint,
and paper-facing utilities in one package. Large artifacts such as checkpoints,
logs, and rollout dumps are intentionally kept outside the repository. The
small Numina-3K split and paper-aligned evaluation parquet files are included
under `data/`.

## Contents

- `ruar/reflective_step_extraction.py`: extracts reflective reasoning steps
  from configurable start and end delimiter sets. The default DS-7B setting
  uses `Wait` for both sets.
- `ruar/answer_forcing_utility_estimation.py`: estimates before/after
  correct-answer probabilities and utilities, `u_j = p_j^+ - p_j^-`.
- `ruar/advantage_rescaling.py`: standalone implementation of utility-guided
  and post-ready advantage multipliers.
- `ruar_training/`: the distributed training entrypoint used by the
  reproduction launcher.
- `verl/`: vendored training backend used by `ruar_training`.
- `scripts/train_ruar.sh`: submits the default DS-7B RUAR training run.
- `configs/ruar_ds7b.env`: paper-aligned DS-7B defaults.
- `configs/ruar_qwen3_8b.env`: optional Qwen3-8B configuration.
- `slurm/train_ruar.sbatch`: shared Slurm training job body.
- `data/numina_3k/`: Numina-3K train/validation parquet files used by the
  training launcher.
- `data/paper_eval/`: paper-aligned `test.parquet` files for the seven
  benchmark suite.
- `scripts/prepare_numina3k.py`: optional utility to rebuild the Numina-3K
  training parquet split.
- `scripts/eval_ruar_7bench.sh`: evaluates the paper-style seven benchmark
  suite, including held-out MCQ prompts.
- `scripts/prepare_eval_benchmark.py`: converts local benchmark JSONL files to
  the RUAR parquet schema for evaluation.
- `scripts/export_fsdp_checkpoint.py`: exports a saved RUAR FSDP actor
  checkpoint to Hugging Face weights for evaluation.
- `scripts/summarize_ruar_results.py`: summarizes accuracy and response length
  from evaluation CSV/COT dumps and computes example-count-weighted averages.
- `requirements-train.txt`: Python package set used for the full training
  environment.
- `docs/installation.md`: environment setup instructions for lightweight
  method utilities and full training reproduction.

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

For full training, install a CUDA-compatible PyTorch build first,
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
- 30 policy updates, producing the selected `global_step_30` checkpoint
- training max response length `16384`
- answer-forcing samples `K=4` with at most `16` generated tokens per sample
- answer-ready threshold `tau=0.75`, streak length `c=3`
- post-ready multipliers `gamma_post^+=0.25`, `gamma_post^-=1.25`
- length penalty coefficient `lambda_len=0.2`
- default DS-7B delimiters: `Wait` starts and ends reflective steps
- `REFLECTION_STOP_AFTER_READY=True`, matching the paper's answer-ready probing rule

The paper uses the following model-specific selected budgets and delimiter
sets. The public launcher defaults to DS-7B.

| Backbone | Updates | Start delimiters | End delimiters |
|---|---:|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | 110 | `Wait` | `Wait` |
| DeepSeek-R1-Distill-Qwen-7B | 30 | `Wait` | `Wait` |
| Qwen3-1.7B | 15 | `Wait`, `Alternatively` | `Wait`, `Alternatively` |
| Qwen3-8B | 30 | `Wait`, `Alternatively` | `Wait`, `Alternatively` |

The launcher keeps one vLLM rollout session active for the complete probe
phase of an update and reuses one CPU verifier worker pool across probe rounds.
This avoids repeated actor-weight synchronization, engine wake/sleep cycles,
and process-pool startup without changing the extracted boundaries, probe
sample count, scoring rule, or answer-ready stopping criterion.

## Training Environment

The default launcher runs the DS-7B configuration included in this repository:
`scripts/train_ruar.sh` loads `configs/ruar_ds7b.env` and submits the shared
Slurm job body, which starts `python -m ruar_training.main`.

Set the local model path before launching:

```bash
export MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-7B
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
export MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-7B
bash scripts/train_ruar.sh
```

For dependency scheduling:

```bash
SBATCH_DEPENDENCY=afterok:<job_id> bash scripts/train_ruar.sh
```

## Evaluate

The paper-aligned evaluation `test.parquet` files are included under
`data/paper_eval/`. Then submit:

```bash
cd RUAR
export BASE_MODEL=/path/to/DeepSeek-R1-Distill-Qwen-7B
export CKPT_ROOT=/path/to/ruar/checkpoint/root
bash scripts/eval_ruar_7bench.sh
```

The evaluation launcher is self-contained within this repository. It reads
evaluation parquet files with `reward_info.ground_truth`, exports
`global_step_30/actor` to `actor/huggingface` when needed, and runs greedy
single-sample decoding with `MAX_RESPONSE_LENGTH=16384`. The held-out MCQ
benchmarks use the `mcq/*` data sources and boxed option-letter prompts used by
the reproduction runs.

After the eval job writes `reports/ruar_7bench_results.csv`:

```bash
python scripts/summarize_ruar_results.py \
  --results-csv reports/ruar_7bench_results.csv \
  --format md
```

## Paper Alignment

The public API uses the paper terms directly:

- `ReflectiveStep` for extracted reflective reasoning steps.
- `estimate_answer_forcing_utilities` for before/after answer-forcing probes.
- `compute_ruar_gammas` and `rescale_advantages` for sign-dependent utility
  and post-ready advantage rescaling.
- `find_answer_ready_step` for the earliest stable answer-ready boundary.
