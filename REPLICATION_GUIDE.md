# Replication Guide

## Purpose

This document is written for an assessor who wants to work through the
repository in a practical order and verify that the submitted results are
traceable and reproducible.

The intended progression is:

1. confirm that the repository loads correctly
2. inspect the submitted artefacts already bundled in the repo
3. run the cheapest real rerun using bundled data and preserved checkpoints
4. rerun selected external-dataset evaluations
5. retrain models if deeper reproduction is required
6. aggregate regenerated outputs and compare them against the submitted CSVs

This guide assumes that the external `RHD` and `HK26K / coco_hand` datasets
already exist outside the repository and can be supplied as local dataset roots.

## Step 1: Confirm That The Repository Loads

First prepare the Conda environment described in `INSTALL.md`, then run:

```bash
python -m pytest tests -q
python scripts/train.py --help
python scripts/eval_metrics.py --help
python scripts/predict_image.py --help
```

Success means:

- the lightweight test suite completes without failures
- the main command-line entry points print their usage information and exit cleanly

If these commands fail, stop here and resolve the environment issue before
attempting replication.

## Step 2: Inspect The Submitted Artefacts

Before rerunning anything expensive, inspect the artefacts that are already
bundled in the repository:

- `docs/artefacts/checkpoints/`
- `docs/artefacts/results/rh8/`
- `docs/artefacts/results/mediapipe_rhd/`
- `docs/artefacts/results/aggregated/`
- `docs/artefacts/results/paper_tables/`
- `docs/artefacts/results/benchmarks/`

These folders provide the submitted reference outputs that regenerated files
should be compared against.

## Step 3: Run The Cheapest Real Rerun

The easiest meaningful rerun uses the bundled `RH8` dataset together with the
preserved checkpoints in `docs/artefacts/checkpoints/`.

### RH8 Baseline Check

Purpose:
verify that a preserved learned checkpoint can still be evaluated against the
bundled real-image artefact and reproduce a submitted JSON result.

Inputs:

- dataset root: `docs/artefacts/datasets/rh8`
- checkpoint: `docs/artefacts/checkpoints/drh-b0-k21-right-s101.best.pt`

Command:

```bash
python scripts/eval_metrics.py --dataset coco_hand --ckpt docs/artefacts/checkpoints/drh-b0-k21-right-s101.best.pt --root docs/artefacts/datasets/rh8 --split val --prediction-mode fusion --shared-10-eval --out-json out/rh8/drh-b0-k21-right-s101.fusion.shared10.rh8.json
```

Writes:

- `out/rh8/drh-b0-k21-right-s101.fusion.shared10.rh8.json`

Compare against:

- `docs/artefacts/results/rh8/drh-b0-k21-right-s101.fusion.shared10.rh8.json`
- `docs/artefacts/results/rh8/rh8_family_summary.csv`

### RH8 Reduced Shared-10 Check

Purpose:
verify the reduced `B1` configuration on the bundled real-image artefact.

Inputs:

- dataset root: `docs/artefacts/datasets/rh8`
- checkpoint: `docs/artefacts/checkpoints/drh-b1-k10-right-s101.best.pt`

Command:

```bash
python scripts/eval_metrics.py --dataset coco_hand --ckpt docs/artefacts/checkpoints/drh-b1-k10-right-s101.best.pt --root docs/artefacts/datasets/rh8 --split val --prediction-mode fusion --out-json out/rh8/drh-b1-k10-right-s101.fusion.native10.rh8.json
```

Writes:

- `out/rh8/drh-b1-k10-right-s101.fusion.native10.rh8.json`

Compare against:

- `docs/artefacts/results/rh8/drh-b1-k10-right-s101.fusion.native10.rh8.json`

### RH8 MediaPipe Check

Purpose:
verify the MediaPipe baseline on the same bundled real-image artefact.

Inputs:

- dataset root: `docs/artefacts/datasets/rh8`
- model asset: `src/handpose/models/hand_landmarker.task`

Command:

```bash
python scripts/eval_mediapipe_coco_hand.py --model-asset-path src/handpose/models/hand_landmarker.task --root docs/artefacts/datasets/rh8 --split val --dataset-name rh8 --hand auto --shared-10-eval --out-json out/rh8/mediapipe.rh8.shared10.json
```

Writes:

- `out/rh8/mediapipe.rh8.shared10.json`

Compare against:

- `docs/artefacts/results/rh8/mediapipe.rh8.shared10.json`

The full checked-in RH8 command set is also available in:

- `docs/artefacts/commands/qual_eval_rh8_commands.sh`

## Step 4: Re-run Selected External-Dataset Evaluations

Once the bundled RH8 pass works, the next level is to rerun selected evaluations
that depend on the external datasets.

### RHD Ablation Evaluation

Purpose:
rerun a representative learned model on `RHD` and compare the output against the
submitted RHD result families.

Inputs:

- dataset root: `<RHD_ROOT>`
- checkpoint: `docs/artefacts/checkpoints/drh-b0-k21-right-s101.best.pt`

Command:

```bash
python scripts/eval_metrics.py --ckpt docs/artefacts/checkpoints/drh-b0-k21-right-s101.best.pt --root <RHD_ROOT> --split evaluation --hand right --prediction-mode fusion --out-json out/eval/drh-b0-k21-right-s101.fusion.native.json
```

Writes:

- `out/eval/drh-b0-k21-right-s101.fusion.native.json`

Compare against:

- `docs/artefacts/results/paper_tables/table_01_core_accuracy_rhd.csv`
- `docs/artefacts/results/paper_tables/table_02_branch_ablation_rhd.csv`
- `docs/artefacts/results/paper_tables/table_06_fusion_diagnostics.csv`
- `docs/artefacts/results/paper_tables/table_07_hyperparameter_sensitivity.csv`
- `docs/artefacts/results/aggregated/aggregated_accuracy.csv`
- `docs/artefacts/results/aggregated/aggregated_branch_ablation.csv`
- `docs/artefacts/results/aggregated/aggregated_fusion_diagnostics.csv`

Further ablation command templates:

- `experiments/ablation/eval_commands.txt`
- `experiments/ablation/shared10_diag_eval_commands.txt`

### Transfer Evaluation

Purpose:
rerun a representative transfer-learning checkpoint on both external datasets.

Inputs:

- dataset roots: `<HK26K_ROOT>` and `<RHD_ROOT>`
- checkpoint: `docs/artefacts/checkpoints/trf-r-to-c-s505.best.pt`

Commands:

```bash
python scripts/eval_metrics.py --ckpt docs/artefacts/checkpoints/trf-r-to-c-s505.best.pt --dataset coco_hand --root <HK26K_ROOT> --split evaluation --hand right --prediction-mode fusion --out-json out/transfer/trf-r-to-c-s505.on-coco_hand.fusion.native.json
python scripts/eval_metrics.py --ckpt docs/artefacts/checkpoints/trf-r-to-c-s505.best.pt --dataset rhd --root <RHD_ROOT> --split evaluation --hand right --prediction-mode fusion --out-json out/transfer/trf-r-to-c-s505.on-rhd.fusion.native.json
```

Writes:

- `out/transfer/trf-r-to-c-s505.on-coco_hand.fusion.native.json`
- `out/transfer/trf-r-to-c-s505.on-rhd.fusion.native.json`

Compare against:

- `docs/artefacts/results/paper_tables/table_03_transfer_learning_matrix.csv`
- `docs/artefacts/results/rh8/rh8_family_summary.csv`

Further transfer command templates:

- `experiments/transfer/eval_commands.txt`

### MediaPipe Baseline On RHD

Purpose:
rerun the external-dataset MediaPipe baseline that feeds the submitted baseline
comparison.

Inputs:

- dataset root: `<RHD_ROOT>`
- model asset: `src/handpose/models/hand_landmarker.task`

Command:

```bash
python scripts/eval_mediapipe_rhd.py --model-asset-path src/handpose/models/hand_landmarker.task --root <RHD_ROOT> --split evaluation --hand auto --shared-10-eval --out-json out/mediapipe_rhd/mediapipe_rhd_shared10.json
```

Writes:

- `out/mediapipe_rhd/mediapipe_rhd_shared10.json`

Compare against:

- `docs/artefacts/results/mediapipe_rhd/mediapipe_rhd_shared10.json`
- `docs/artefacts/results/mediapipe_rhd/mediapipe_summary.csv`
- `docs/artefacts/results/paper_tables/table_04_mediapipe_baseline_comparison.csv`

## Step 5: Retrain The Models

If you need to remake the learned models rather than just reevaluate preserved
checkpoints, use `scripts/train.py`.

For each run, the trainer writes:

- `<checkpoint_root>/<job_id>/losses.csv`
- `<checkpoint_root>/<job_id>/checkpoints/stage1_best.pt`
- `<checkpoint_root>/<job_id>/checkpoints/best.pt`

The final stage-2 model for comparison is always:

- `<checkpoint_root>/<job_id>/checkpoints/best.pt`

### Rebuild One Ablation Model

Purpose:
recreate a representative model from the ablation family.

Inputs:

- dataset root: `<RHD_ROOT>`
- checkpoint root: `out/training_results`

Command:

```bash
python scripts/train.py --root <RHD_ROOT> --checkpoint-root out/training_results --job-id drh-b0-k21-right-s101 --seed 101 --input-size 256 --batch-size-stage1 64 --batch-size-stage2 64 --accum-steps-stage1 1 --accum-steps-stage2 4 --stage1-epochs 100 --stage2-epochs 100 --lr-stage1 1e-3 --lr-stage2 1e-4 --hand right --lambda-hm 1.0 --lambda-coord 1.0 --heatmap-sigma 2.0 --wing-w 10.0 --wing-epsilon 2.0
```

Compare against:

- `docs/artefacts/checkpoints/drh-b0-k21-right-s101.best.pt`

Full ablation training surfaces:

- `experiments/ablation/experiment_matrix.csv`
- `experiments/ablation/train_commands.txt`

### Rebuild One Transfer Model

Purpose:
recreate a representative transfer-learning model from scratch or by fine-tuning.

Inputs:

- dataset root: `<RHD_ROOT>`
- checkpoint root: `out/transfer_training_results`

Direct-training example:

```bash
python scripts/train.py --dataset rhd --root <RHD_ROOT> --checkpoint-root out/transfer_training_results --job-id trf-r-only-s101 --experiment-id R_ONLY --experiment-family transfer --training-sequence rhd --seed 101 --input-size 256 --batch-size-stage1 64 --batch-size-stage2 64 --accum-steps-stage1 1 --accum-steps-stage2 4 --stage1-epochs 100 --stage2-epochs 100 --lr-stage1 0.001 --lr-stage2 0.0001 --hand right
```

Fine-tuning example:

```bash
python scripts/train.py --dataset rhd --root <RHD_ROOT> --checkpoint-root out/transfer_training_results --job-id trf-c-to-r-s101 --experiment-id C_TO_R --experiment-family transfer --training-sequence coco_hand->rhd --init-ckpt out/transfer_training_results/trf-c-only-s101/checkpoints/best.pt --skip-stage1 --seed 101 --input-size 256 --batch-size-stage1 64 --batch-size-stage2 64 --accum-steps-stage1 1 --accum-steps-stage2 4 --stage1-epochs 100 --stage2-epochs 100 --lr-stage1 0.001 --lr-stage2 0.0001 --hand right
```

Compare against:

- `docs/artefacts/checkpoints/trf-r-only-s101.best.pt`
- `docs/artefacts/checkpoints/trf-c-to-r-s101.best.pt`

Full transfer training surfaces:

- `experiments/transfer/experiment_matrix.csv`
- `experiments/transfer/train_commands.txt`

## Step 6: Aggregate Regenerated Outputs

If you rerun evaluation or benchmarking, aggregate the regenerated files to
recreate repository-style summary CSVs.

Command:

```bash
python scripts/aggregate_results.py --eval-json-dir out/eval --benchmark-csv out/benchmarks/jetson_orin_nano_maxn_summary.csv --out-dir out/aggregated
```

Writes:

- `out/aggregated/aggregated_accuracy.csv`
- `out/aggregated/aggregated_branch_ablation.csv`
- `out/aggregated/aggregated_fusion_diagnostics.csv`
- `out/aggregated/aggregated_latency.csv`
- `out/aggregated/aggregated_latency_breakdown.csv`

Compare against:

- `docs/artefacts/results/aggregated/aggregated_accuracy.csv`
- `docs/artefacts/results/aggregated/aggregated_branch_ablation.csv`
- `docs/artefacts/results/aggregated/aggregated_fusion_diagnostics.csv`
- `docs/artefacts/results/aggregated/aggregated_latency.csv`
- `docs/artefacts/results/aggregated/aggregated_latency_breakdown.csv`

## Step 7: Benchmark If Required

Benchmarking is the most hardware-dependent part of the project. It should be
checked after the evaluation path is already working.

Purpose:
rerun one representative edge-runtime measurement.

Inputs:

- dataset root: `<RHD_ROOT>`
- checkpoint: `docs/artefacts/checkpoints/drh-b0-k21-right-s101.best.pt`

Command:

```bash
python scripts/benchmark_pipeline.py --ckpt docs/artefacts/checkpoints/drh-b0-k21-right-s101.best.pt --root <RHD_ROOT> --summary-csv out/benchmarks/jetson_orin_nano_maxn_summary.csv --output-root out/benchmarks --device cuda --prediction-mode fusion --run-label drh-b0-k21-right-s101-fusion
```

Writes:

- `out/benchmarks/jetson_orin_nano_maxn_summary.csv`

Compare against:

- `docs/artefacts/results/paper_tables/table_05_edge_runtime_summary.csv`
- `docs/artefacts/results/paper_tables/table_08_latency_breakdown.csv`
- `docs/artefacts/results/benchmarks/`

Further benchmark templates:

- `experiments/ablation/benchmark_commands.txt`
- `experiments/transfer/benchmark_commands.txt`

## Full Command Templates

If you want the matrix-style command surfaces used throughout the project, use:

- `experiments/ablation/train_commands.txt`
- `experiments/ablation/eval_commands.txt`
- `experiments/ablation/shared10_diag_eval_commands.txt`
- `experiments/ablation/benchmark_commands.txt`
- `experiments/transfer/train_commands.txt`
- `experiments/transfer/eval_commands.txt`
- `experiments/transfer/benchmark_commands.txt`

These files are checked in as portable templates. Replace placeholders such as
`<RHD_ROOT>`, `<HK26K_ROOT>`, `<CHECKPOINT_ROOT>`, and
`<TRANSFER_CHECKPOINT_ROOT>` before running them on a new system. A short
placeholder reference is provided in `experiments/README.md`.

## Shortest Useful Path

If you only want to check one live result, run the bundled `RH8` baseline check
from Step 3 and compare the generated JSON against the preserved reference
under `docs/artefacts/results/rh8/`.
