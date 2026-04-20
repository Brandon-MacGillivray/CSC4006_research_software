# Replication Guide

## What This Document Is For

This repository can be used in three different ways:

1. inspect the submitted paper outputs that are already bundled in the repo
2. run evaluation again using the preserved final checkpoints
3. retrain the models and recreate the checkpoints from scratch

This guide assumes that the external `RHD` and `HK26K / coco_hand` datasets
already exist outside the repository and can be passed in as local dataset
roots.

## What Is Already In The Repository

You do not need to recreate everything from zero to understand the project.
The repository already includes:

- the paper source and PDF under `docs/paper/`
- the final preserved checkpoints under `docs/artefacts/checkpoints/`
- the bundled `RH8` evaluation dataset under `docs/artefacts/datasets/rh8/`
- the final result tables under `docs/artefacts/results/paper_tables/`
- aggregated CSV summaries under `docs/artefacts/results/aggregated/`
- RH8 evaluation JSONs under `docs/artefacts/results/rh8/`
- MediaPipe baseline outputs under `docs/artefacts/results/mediapipe_rhd/`

So the repo already contains both:

- the final outputs
- the code needed to regenerate them

## Before You Start

Prepare the Conda environment described in `INSTALL.md`, then confirm the repo
loads correctly:

```bash
python -m pytest tests -q
python scripts/train.py --help
python scripts/eval_metrics.py --help
python scripts/predict_image.py --help
```

If those commands work, the repo is ready for replication work.

## Choose What You Want To Do

### A. Just Check The Submitted Results

If the goal is simply to inspect what was submitted, start here:

- `docs/artefacts/results/paper_tables/`
- `docs/artefacts/results/aggregated/`
- `docs/artefacts/results/rh8/`
- `docs/artefacts/results/mediapipe_rhd/`
- `docs/artefacts/results/benchmarks/`

This path requires no retraining and no checkpoint regeneration.

### B. Run Evaluation On The Bundled RH8 Dataset

This is the easiest real rerun because both the dataset and the final
checkpoints are already in the repo.

Example: evaluate the baseline `B0` checkpoint on `RH8`.

```bash
python scripts/eval_metrics.py --dataset coco_hand --ckpt docs/artefacts/checkpoints/drh-b0-k21-right-s101.best.pt --root docs/artefacts/datasets/rh8 --split val --prediction-mode fusion --shared-10-eval --out-json out/rh8/drh-b0-k21-right-s101.fusion.shared10.rh8.json
```

Example: evaluate the reduced `B1` checkpoint on `RH8`.

```bash
python scripts/eval_metrics.py --dataset coco_hand --ckpt docs/artefacts/checkpoints/drh-b1-k10-right-s101.best.pt --root docs/artefacts/datasets/rh8 --split val --prediction-mode fusion --out-json out/rh8/drh-b1-k10-right-s101.fusion.native10.rh8.json
```

Example: run the MediaPipe baseline on `RH8`.

```bash
python scripts/eval_mediapipe_coco_hand.py --model-asset-path src/handpose/models/hand_landmarker.task --root docs/artefacts/datasets/rh8 --split val --dataset-name rh8 --hand auto --shared-10-eval --out-json out/rh8/mediapipe.rh8.shared10.json
```

Compare your outputs against:

- `docs/artefacts/results/rh8/`

The full RH8 command set is also preserved in:

- `docs/artefacts/commands/qual_eval_rh8_commands.sh`

### C. Re-run Evaluation On The External Datasets

If you already have the external datasets, you can rerun the main evaluations.

#### RHD Ablation Evaluation

Example: rerun one ablation checkpoint on `RHD`.

```bash
python scripts/eval_metrics.py --ckpt docs/artefacts/checkpoints/drh-b0-k21-right-s101.best.pt --root <RHD_ROOT> --split evaluation --hand right --prediction-mode fusion --out-json out/eval/drh-b0-k21-right-s101.fusion.native.json
```

Other ablation command templates are stored under:

- `experiments/ablation/eval_commands.txt`
- `experiments/ablation/shared10_diag_eval_commands.txt`

#### Transfer Evaluation

Example: evaluate a transfer checkpoint on `HK26K / coco_hand`.

```bash
python scripts/eval_metrics.py --ckpt docs/artefacts/checkpoints/trf-r-to-c-s505.best.pt --dataset coco_hand --root <HK26K_ROOT> --split evaluation --hand right --prediction-mode fusion --out-json out/transfer/trf-r-to-c-s505.on-coco_hand.fusion.native.json
```

Example: evaluate the same checkpoint on `RHD`.

```bash
python scripts/eval_metrics.py --ckpt docs/artefacts/checkpoints/trf-r-to-c-s505.best.pt --dataset rhd --root <RHD_ROOT> --split evaluation --hand right --prediction-mode fusion --out-json out/transfer/trf-r-to-c-s505.on-rhd.fusion.native.json
```

Other transfer command templates are stored under:

- `experiments/transfer/eval_commands.txt`

#### MediaPipe Baseline On RHD

```bash
python scripts/eval_mediapipe_rhd.py --model-asset-path src/handpose/models/hand_landmarker.task --root <RHD_ROOT> --split evaluation --hand auto --shared-10-eval --out-json out/mediapipe_rhd/mediapipe_rhd_shared10.json
```

Compare regenerated outputs against:

- `docs/artefacts/results/mediapipe_rhd/`
- `docs/artefacts/results/paper_tables/`

## Retraining The Models

If your goal is to remake the models, use `scripts/train.py`.

The trainer writes outputs to:

- `<checkpoint_root>/<job_id>/losses.csv`
- `<checkpoint_root>/<job_id>/checkpoints/stage1_best.pt`
- `<checkpoint_root>/<job_id>/checkpoints/best.pt`

The final stage-2 model is always:

- `<checkpoint_root>/<job_id>/checkpoints/best.pt`

### Rebuild The Ablation Models

The ablation experiment family is defined in:

- `experiments/ablation/experiment_matrix.csv`
- `experiments/ablation/train_commands.txt`

Example: train one baseline `B0` model from scratch on `RHD`.

```bash
python scripts/train.py --root <RHD_ROOT> --checkpoint-root out/training_results --job-id drh-b0-k21-right-s101 --seed 101 --input-size 256 --batch-size-stage1 64 --batch-size-stage2 64 --accum-steps-stage1 1 --accum-steps-stage2 4 --stage1-epochs 100 --stage2-epochs 100 --lr-stage1 1e-3 --lr-stage2 1e-4 --hand right --lambda-hm 1.0 --lambda-coord 1.0 --heatmap-sigma 2.0 --wing-w 10.0 --wing-epsilon 2.0
```

Example: train one reduced `B1` model on the shared-10 layout.

```bash
python scripts/train.py --root <RHD_ROOT> --checkpoint-root out/training_results --job-id drh-b1-k10-right-s101 --seed 101 --input-size 256 --batch-size-stage1 64 --batch-size-stage2 64 --accum-steps-stage1 1 --accum-steps-stage2 4 --stage1-epochs 100 --stage2-epochs 100 --lr-stage1 1e-3 --lr-stage2 1e-4 --hand right --lambda-hm 1.0 --lambda-coord 1.0 --heatmap-sigma 2.0 --wing-w 10.0 --wing-epsilon 2.0 --tips-bases-only
```

### Rebuild The Transfer Models

The transfer experiment family is defined in:

- `experiments/transfer/experiment_matrix.csv`
- `experiments/transfer/train_commands.txt`

There are two cases:

- direct training on a single dataset, such as `R_ONLY` or `C_ONLY`
- fine-tuning from a parent checkpoint, such as `C_TO_R` or `R_TO_C`

Example: train one `R_ONLY` model from scratch.

```bash
python scripts/train.py --dataset rhd --root <RHD_ROOT> --checkpoint-root out/transfer_training_results --job-id trf-r-only-s101 --experiment-id R_ONLY --experiment-family transfer --training-sequence rhd --seed 101 --input-size 256 --batch-size-stage1 64 --batch-size-stage2 64 --accum-steps-stage1 1 --accum-steps-stage2 4 --stage1-epochs 100 --stage2-epochs 100 --lr-stage1 0.001 --lr-stage2 0.0001 --hand right
```

Example: fine-tune one `C_TO_R` model from an existing `C_ONLY` checkpoint.

```bash
python scripts/train.py --dataset rhd --root <RHD_ROOT> --checkpoint-root out/transfer_training_results --job-id trf-c-to-r-s101 --experiment-id C_TO_R --experiment-family transfer --training-sequence coco_hand->rhd --init-ckpt out/transfer_training_results/trf-c-only-s101/checkpoints/best.pt --skip-stage1 --seed 101 --input-size 256 --batch-size-stage1 64 --batch-size-stage2 64 --accum-steps-stage1 1 --accum-steps-stage2 4 --stage1-epochs 100 --stage2-epochs 100 --lr-stage1 0.001 --lr-stage2 0.0001 --hand right
```

After training, compare the recreated `best.pt` files against the preserved
archive under:

- `docs/artefacts/checkpoints/`

## Aggregating Regenerated Outputs

If you rerun evaluation or benchmarking, aggregate the regenerated files with:

```bash
python scripts/aggregate_results.py --eval-json-dir out/eval --benchmark-csv out/benchmarks/jetson_orin_nano_maxn_summary.csv --out-dir out/aggregated
```

This recreates repository-style summary CSVs such as:

- `aggregated_accuracy.csv`
- `aggregated_branch_ablation.csv`
- `aggregated_fusion_diagnostics.csv`
- `aggregated_latency.csv`
- `aggregated_latency_breakdown.csv`

Compare them against:

- `docs/artefacts/results/aggregated/`

## Optional Benchmark Reproduction

Benchmarking is the most hardware-dependent part of the project.

Example:

```bash
python scripts/benchmark_pipeline.py --ckpt docs/artefacts/checkpoints/drh-b0-k21-right-s101.best.pt --root <RHD_ROOT> --summary-csv out/benchmarks/jetson_orin_nano_maxn_summary.csv --output-root out/benchmarks --device cuda --prediction-mode fusion --run-label drh-b0-k21-right-s101-fusion
```

More benchmark command templates are stored under:

- `experiments/ablation/benchmark_commands.txt`
- `experiments/transfer/benchmark_commands.txt`

## Where The Full Command Sets Live

If you want the exact matrix-style surfaces used in the project, use these
files:

- `experiments/ablation/train_commands.txt`
- `experiments/ablation/eval_commands.txt`
- `experiments/ablation/shared10_diag_eval_commands.txt`
- `experiments/ablation/benchmark_commands.txt`
- `experiments/transfer/train_commands.txt`
- `experiments/transfer/eval_commands.txt`
- `experiments/transfer/benchmark_commands.txt`

These files still contain historical absolute machine paths, so replace those
paths before running them on a new system.

## In One Sentence

If you want the simplest real replication path, run evaluation on the bundled
`RH8` dataset using the preserved checkpoints. If you want to remake the
models, use `scripts/train.py` together with the experiment matrices under
`experiments/`.
