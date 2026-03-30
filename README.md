# DRHand Replica (RHD / COCO-Hand 2D Hand Pose Training)

PyTorch training code for a dual-branch hand pose model (heatmaps + coordinates), inspired by the DRHand paper (`Dual Regression for Efficient Hand Pose Estimation`).

The repo now supports:

- `rhd` training/evaluation on the RHD dataset
- `coco_hand` training/evaluation on a COCO-style 21-keypoint hand dataset
- `21` keypoints (default) or `10` keypoints (`--tips-bases-only`)
- scratch training and transfer fine-tuning (`C->R`, `R->C`)

## Artefact Files

The repository includes the following root-level documents to support installation,
reproducibility, and assessment as a research artefact:

- `REQUIREMENTS.md`: hardware, software, dataset, and storage expectations
- `environment.yml`: Conda environment definition for the base workflow
- `requirements.txt`: base Python package dependencies for the core workflow
- `INSTALL.md`: setup and smoke-test instructions
- `REPLICATION_GUIDE.md`: step-by-step workflow for reproducing the reported runs
- `LICENSE`: distribution terms for the code in this repository

## Structure

- `src/handpose/data/`: shared dataset factory + transforms + dataset-specific loaders
- `src/handpose/data/rhd/`: RHD parsing, paths, and dataset loader
- `src/handpose/data/coco_hand/`: COCO-hand parsing, paths, and dataset loader
- `src/handpose/models/`: network blocks, architecture, model wrapper, losses
- `src/handpose/training/`: train loops, optimizer policies, runtime helpers
- `src/handpose/checkpoints.py`: shared checkpoint schema, save/load helpers
- `src/handpose/evaluation/`: eval pipeline, metrics core, result payload helpers
- `src/handpose/inference/`: image preprocess + inference + overlay utilities
- `scripts/`: runnable CLIs (`train.py`, `eval_metrics.py`, `benchmark_pipeline.py`, matrix generators)
- `slurm/`: Slurm array wrappers
- `docs/`: paper notes / scratch command history

## Requirements

- Python 3.10+ (tested with 3.12 syntax)
- PyTorch
- `numpy`
- `Pillow`
- `matplotlib` (for `plot_losses.py`)
- `mediapipe` (optional, only for the MediaPipe baseline script)

Example install:

```bash
conda env create -f environment.yml
conda activate drhand
```

Optional baseline dependency:

```bash
pip install mediapipe
```

Optional webcam demo dependency:

```bash
pip install opencv-python
```

## Expected Dataset Layouts

### RHD

Default root: `data/RHD_published_v2`

```text
data/RHD_published_v2/
  training/
    color/
      00000.png
      ...
    anno_training.pickle
  evaluation/
    color/
      00000.png
      ...
    anno_evaluation.pickle
```

### COCO-Hand

Default root: `data/hand_keypoint_dataset`

```text
data/hand_keypoint_dataset/
  images/
    train/
    val/
  coco_annotation/
    train/
      _annotations.coco.json
    val/
      _annotations.coco.json
```

The repo assumes the COCO-hand dataset uses the same 21-keypoint order as MediaPipe-style hand landmarks.

## Training

`scripts/train.py` always uses the same two-stage workflow:

1. Stage 1 trains backbone + heatmap head
2. Stage 2 trains the coordinate branch and optionally fine-tunes the rest

Outputs are written to:

- `training_results/<job-id>/losses.csv`
- `training_results/<job-id>/checkpoints/`

If `--job-id` is omitted, it uses `local`.

### Scratch Training

Train on RHD:

```bash
python scripts/train.py --dataset rhd --root data/RHD_published_v2 --job-id exp_rhd_k21
```

Train on COCO-hand:

```bash
python scripts/train.py --dataset coco_hand --root data/hand_keypoint_dataset --job-id exp_coco_k21
```

Train 10-keypoint mode:

```bash
python scripts/train.py --dataset rhd --root data/RHD_published_v2 --job-id exp_rhd_k10 --tips-bases-only
```

### Transfer Fine-Tuning

Transfer runs load an existing `best.pt` and skip stage 1:

```bash
python scripts/train.py \
  --dataset rhd \
  --root data/RHD_published_v2 \
  --job-id trf-c-to-r-s101 \
  --experiment-id C_TO_R \
  --experiment-family transfer \
  --training-sequence coco_hand->rhd \
  --init-ckpt transfer_training_results/trf-c-only-s101/checkpoints/best.pt \
  --skip-stage1
```

Important:

- transfer runs require the same keypoint layout as the parent checkpoint
- `--skip-stage1` requires `--init-ckpt`
- stage-2 optimizer state is rebuilt; transfer runs do not reuse the parent optimizer state

### Hand Selection

`scripts/train.py` supports:

- `--hand right` (default)
- `--hand left`
- `--hand auto`

For non-RHD datasets, the stored hand metadata is normalized to `single`.

## Useful CLI Options

### Training (`scripts/train.py`)

- `--dataset rhd|coco_hand`: choose the dataset loader
- `--seed`: seed Python / NumPy / PyTorch for repeatable runs
- `--accum-steps-stage1`, `--accum-steps-stage2`: gradient accumulation
- `--lambda-hm`, `--lambda-coord`: stage-2 loss weights
- `--heatmap-sigma`: Gaussian target sigma
- `--wing-w`, `--wing-epsilon`: Wing loss shape parameters
- `--freeze-backbone-stage2`, `--freeze-heatmap-stage2`: freeze parts of the model during stage 2
- `--train-dataset-length N`: train on the first `N` training samples
- `--experiment-id`, `--experiment-family`, `--training-sequence`: metadata for aggregation/reporting
- `--init-ckpt`, `--skip-stage1`: transfer fine-tuning controls

### Evaluation (`scripts/eval_metrics.py`)

- `--dataset auto|rhd|coco_hand`: choose dataset explicitly or resolve from checkpoint metadata
- `--device auto|cpu|cuda`: force evaluation device
- `--prediction-mode fusion|heatmap|coord`: branch selection for evaluation
- `--batch-size`, `--num-workers`: evaluation loader settings
- `--debug-coords`: print first-batch diagnostics to `stderr`
- `--with-fusion-diagnostics`: export fusion-selection statistics in the result JSON

### MediaPipe Baseline (`scripts/eval_mediapipe_rhd.py`)

- `--model-asset-path`: path to the MediaPipe Hand Landmarker `.task` model
- `--hand left|right|auto`: choose the RHD target hand
- `--ignore-handedness`: ignore MediaPipe handedness labels and always score the highest-confidence detection
- `--shared-10-eval`: evaluate only the shared 10 keypoints
- `--out-json`: write a result JSON compatible with the repo's aggregation flow

## Plot Training Curves

```bash
python scripts/plot_losses.py --job-id exp_rhd_k21
```

This reads `training_results/<job-id>/losses.csv` and writes `loss_plot.png` in the same run directory.

## Testing

Basic repository smoke test:

```bash
python scripts/smoke_test.py
```

Synthetic unit tests:

```bash
pytest -q
```

The smoke test checks that the main non-optional CLIs start correctly. The `pytest`
suite covers checkpoint validation, fusion helpers, evaluation helper logic, and model
output shapes without requiring external datasets.

## Evaluate Trained Models

Use `scripts/eval_metrics.py` on a stage-2 checkpoint (`best.pt`) to report:

- normalized `SSE` per sample (`metrics.sse_norm`)
- root-relative normalized `EPE` (`metrics.epe_norm`)
- `PCK@sigma` (`metrics.pck`) with `--pck-threshold` (default `0.2`)
- sample/keypoint counts
- timing for the selected `--prediction-mode`

Evaluate on the dataset stored in the checkpoint:

```bash
python scripts/eval_metrics.py \
  --dataset auto \
  --ckpt training_results/exp_rhd_k21/checkpoints/best.pt \
  --root data/RHD_published_v2 \
  --split evaluation \
  --out-json eval_results/exp_rhd_k21.json
```

Evaluate a checkpoint cross-dataset:

```bash
python scripts/eval_metrics.py \
  --dataset coco_hand \
  --ckpt transfer_training_results/trf-c-to-r-s101/checkpoints/best.pt \
  --root data/hand_keypoint_dataset \
  --split val \
  --out-json transfer_eval_results/trf-c-to-r-s101.on-coco_hand.json
```

Shared-10 evaluation is still available for fair 21-vs-10 comparisons:

```bash
python scripts/eval_metrics.py \
  --dataset rhd \
  --ckpt training_results/exp_rhd_k21/checkpoints/best.pt \
  --root data/RHD_published_v2 \
  --split evaluation \
  --shared-10-eval \
  --out-json eval_results/exp_rhd_k21_shared10.json
```

### Evaluate The MediaPipe Baseline On RHD

```bash
python scripts/eval_mediapipe_rhd.py \
  --model-asset-path path/to/hand_landmarker.task \
  --root data/RHD_published_v2 \
  --split evaluation \
  --hand right \
  --out-json eval_results/mediapipe_rhd.json
```

This writes a JSON payload aligned with the normal eval output format so `scripts/aggregate_results.py` can consume it.

## Single Image Inference

Use `scripts/predict_image.py` to run one image through a trained checkpoint.

```bash
python scripts/predict_image.py --ckpt training_results/exp_rhd_k21/checkpoints/best.pt --image path/to/image.png
```

Heatmap-only:

```bash
python scripts/predict_image.py --ckpt training_results/exp_rhd_k21/checkpoints/best.pt --image path/to/image.png --prediction-mode heatmap
```

Overlay output:

```bash
python scripts/predict_image.py --ckpt training_results/exp_rhd_k21/checkpoints/best.pt --image path/to/image.png --overlay --overlay-out infer_results/image_overlay.png
```

## Benchmarking

Use `scripts/benchmark_pipeline.py` for end-to-end single-image latency benchmarking.

Image discovery is dataset-aware:

- RHD: `<root>/evaluation/color`
- COCO-hand: `<root>/images/val`

Example:

```bash
python scripts/benchmark_pipeline.py \
  --dataset auto \
  --ckpt training_results/exp_rhd_k21/checkpoints/best.pt \
  --root data/RHD_published_v2 \
  --summary-csv benchmark_results/summary.csv \
  --output-root benchmark_results \
  --device cuda \
  --prediction-mode fusion \
  --run-label orin_nano
```

Outputs:

- per-image prediction JSON files under `benchmark_results/<run-label>/...`
- aggregate CSV metrics in `--summary-csv`
- `resolved_images.txt` showing exactly which files were benchmarked

If a benchmark image fails, the script now prints the first failing traceback and raises a final error that includes `first_failure=...`.

## Experiment Helpers

### Original Ablation Matrix

Use `scripts/generate_experiment_matrix.py` to emit the original DRHand ablation study into `experiment_plan/`.

### Transfer Matrix

Use `scripts/generate_transfer_experiment_matrix.py` to emit the transfer-learning study into `experiment_plan_transfer/`.

By default it generates:

- `R_ONLY`
- `C_ONLY`
- `C_TO_R`
- `R_TO_C`

across seeds `101 202 303 404 505`, plus train/eval/benchmark command files.

### Aggregate Results

Use `scripts/aggregate_results.py` to aggregate eval JSONs and benchmark summary CSVs into mean/std tables.

Aggregation now separates:

- `experiment_family`
- `training_sequence`
- `train_dataset_name`
- `eval_dataset_name`
- `benchmark_dataset_name`

so transfer runs do not get mixed with the original ablation study.

## Slurm

The repo includes generic array wrappers plus transfer-specific wrappers.

### Original Matrix Wrappers

- `slurm/train_matrix.sbatch`
- `slurm/eval_matrix.sbatch`

These default to `experiment_plan/...`.

### Transfer Matrix Wrappers

- `slurm/train_transfer_matrix.sbatch`
- `slurm/eval_transfer_matrix.sbatch`

These default to `experiment_plan_transfer/...`.

Example submits:

```bash
sbatch --array=1-20 slurm/train_transfer_matrix.sbatch
sbatch --array=1-40 slurm/eval_transfer_matrix.sbatch
```

The transfer train array contains dependencies by design:

- `C_TO_R` needs the matching `C_ONLY` checkpoint for the same seed
- `R_TO_C` needs the matching `R_ONLY` checkpoint for the same seed

So in practice, run the scratch transfer rows first or split submission into phases.

## Important Notes

- Use stage-2 `best.pt` for fused prediction metrics
- `21`-keypoint and `10`-keypoint checkpoints are not shape-compatible
- transfer fine-tuning requires the parent checkpoint to match the requested keypoint layout
- checkpoints follow the strict schema in `src/handpose/checkpoints.py`
- do not reuse the same `--job-id` unless you want files overwritten/appended
- `docs/commands.txt` contains older notes; prefer this README and the current scripts
