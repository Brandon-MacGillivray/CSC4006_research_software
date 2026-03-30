# Replication Guide

## Purpose

This guide explains how to reproduce the main workflows used by this repository:

- original ablation study on RHD
- transfer-learning study between COCO-hand and RHD
- evaluation outputs used for analysis tables
- Jetson latency benchmarking outputs

The repository does not bundle datasets or trained checkpoints, so full reproduction
requires access to the external datasets and suitable hardware.

## What Is Reproduced

The codebase supports reproduction of the following artefacts:

- training checkpoints for ablation and transfer runs
- evaluation JSON files for fusion, heatmap-only, and coord-only prediction modes
- benchmark summary CSVs and per-image prediction payloads
- aggregated CSV tables under `docs/aggregated_results/`

The paper-facing tables in `docs/hand_pose_article.tex` are derived from these aggregated
outputs.

## Prerequisites

Before reproducing results, ensure that:

- the repository is installed as described in `INSTALL.md`
- the required datasets are present in the expected layout or passed via `--root`
- a CUDA-capable GPU is available for practical training
- a Jetson Orin Nano is available if you want to match the reported Jetson latency figures

## Important Reproducibility Rules

- use stage-2 `best.pt` checkpoints for evaluation and benchmarking
- do not mix `21`-keypoint and `10`-keypoint checkpoints
- keep track of the seed used for each run
- use the same prediction mode when comparing like-for-like outputs
- Jetson latency values are device-specific and should be benchmarked on the target hardware

## A. Original RHD Ablation Study

### 1. Generate the Experiment Matrix

The matrix is already checked in under `experiment_plan/`, but you can regenerate it:

```bash
python scripts/generate_experiment_matrix.py
```

This writes:

- `experiment_plan/experiment_matrix.json`
- `experiment_plan/experiment_matrix.csv`
- `experiment_plan/train_commands.txt`
- `experiment_plan/eval_commands.txt`
- `experiment_plan/benchmark_commands.txt`

### 2. Run Training

Option 1: run commands manually from `experiment_plan/train_commands.txt`

Option 2: use the Slurm array wrapper:

```bash
sbatch --array=1-50 slurm/train_matrix.sbatch
```

Training outputs are written under `training_results/<job-id>/`.

### 3. Run Evaluation

Run the evaluation commands from `experiment_plan/eval_commands.txt`, or dispatch them
through Slurm:

```bash
sbatch --array=1-250 slurm/eval_matrix.sbatch
```

These commands produce:

- native evaluation JSON files
- shared-10 fusion evaluation JSON files
- fusion diagnostic JSON files

### 4. Run Benchmarking

Run the commands from `experiment_plan/benchmark_commands.txt` on the target device.
For the paper latency numbers, this should be the Jetson Orin Nano environment.

Each benchmark run writes:

- per-image JSON outputs under the selected benchmark output root
- one summary row to the configured benchmark summary CSV
- `resolved_images.txt` for traceability

### 5. Aggregate Results

Aggregate evaluation and benchmark outputs into report-friendly CSV files:

```bash
python scripts/aggregate_results.py \
  --eval-json-dir eval_results \
  --benchmark-csv benchmark_results/orin_maxn_summary.csv \
  --out-dir docs/aggregated_results
```

Key output files:

- `docs/aggregated_results/aggregated_accuracy.csv`
- `docs/aggregated_results/aggregated_branch_ablation.csv`
- `docs/aggregated_results/aggregated_latency.csv`
- `docs/aggregated_results/aggregated_latency_breakdown.csv`
- `docs/aggregated_results/aggregated_fusion_diagnostics.csv`

## B. Transfer-Learning Study

### 1. Generate the Transfer Matrix

The transfer matrix is already checked in under `experiment_plan_transfer/`, but you can
regenerate it:

```bash
python scripts/generate_transfer_experiment_matrix.py
```

This writes train, eval, and benchmark command files for:

- `R_ONLY`
- `C_ONLY`
- `C_TO_R`
- `R_TO_C`

### 2. Run Training

Run the commands from `experiment_plan_transfer/train_commands.txt`, or use Slurm:

```bash
sbatch --array=1-20 slurm/train_transfer_matrix.sbatch
```

Important:

- `C_TO_R` depends on the corresponding `C_ONLY` checkpoint
- `R_TO_C` depends on the corresponding `R_ONLY` checkpoint

In practice, run the scratch source-domain checkpoints first or split the array submission
into phases.

### 3. Run Cross-Dataset Evaluation

Run commands from `experiment_plan_transfer/eval_commands.txt`, or use Slurm:

```bash
sbatch --array=1-40 slurm/eval_transfer_matrix.sbatch
```

These produce evaluation JSON files for both evaluation datasets defined by the generator.

### 4. Run Transfer Benchmarking

Run the generated transfer benchmark commands on the desired benchmark hardware. These
commands write summary rows and per-image outputs in the same pattern as the original
ablation study.

## Mapping Outputs to Paper Tables

The LaTeX article in `docs/hand_pose_article.tex` draws on aggregated results.

Typical mapping:

- main accuracy comparisons: `aggregated_accuracy.csv`
- branch ablation table: `aggregated_branch_ablation.csv`
- latency summary table: `aggregated_latency.csv`
- latency breakdown table: `aggregated_latency_breakdown.csv`
- fusion diagnostics table: `aggregated_fusion_diagnostics.csv`

Hyperparameter tables are obtained by selecting the relevant experiment rows such as
`B0`, `L1`, `L2`, `S1`, `S2`, `W1`, `W2`, `E1`, and `E2`.

## Notes on Exact Match Expectations

- Metric values depend on dataset contents, preprocessing assumptions, random seed, and
  checkpoint lineage
- Latency values depend strongly on hardware, system load, and device configuration
- Desktop benchmark numbers should not be expected to match Jetson figures

## Recommended Archive for Submission

For an assessment submission or repository snapshot, preserve:

- source code
- root artefact files (`README`, `REQUIREMENTS`, `INSTALL`, `REPLICATION GUIDE`, `LICENSE`)
- experiment matrices and generated command files
- aggregated result CSVs used to populate the report
- any report-ready figures or tables derived from those outputs
