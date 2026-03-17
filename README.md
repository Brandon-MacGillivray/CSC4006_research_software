# DRHand Replica (RHD 2D Hand Pose Training)

PyTorch training code for a dual-branch hand pose model (heatmaps + coordinates), inspired by the DRHand paper (`Dual Regression for Efficient Hand Pose Estimation`).

This repo trains on the RHD dataset and supports two keypoint modes:

- `21` keypoints (default)
- `10` keypoints (finger tips + bases) via `--tips-bases-only`

## Structure

- `src/handpose/data/`: dataset loader, hand/keypoint selection, keypoint constants
- `src/handpose/models/`: network blocks, architecture, model wrapper, losses
- `src/handpose/training/`: train loops, optimizer policies, runtime helpers
- `src/handpose/checkpoints.py`: shared checkpoint schema, save/load helpers
- `src/handpose/evaluation/`: eval pipeline, metrics core, result payload helpers
- `src/handpose/inference/`: image preprocess + inference + overlay utilities
- `scripts/`: runnable CLI entrypoints (`train.py`, `eval_metrics.py`, `predict_image.py`, `plot_losses.py`)
- `slurm/`: Slurm job scripts
- `docs/`: paper notes / scratch command history

## Requirements

- Python 3.10+ (tested with 3.12 syntax)
- PyTorch
- `numpy`
- `Pillow`
- `matplotlib` (for `plot_losses.py`)

Example install (CPU or CUDA variant as appropriate for your machine):

```bash
pip install torch torchvision numpy pillow matplotlib
```

## Expected Dataset Layout (RHD)

`src/handpose/data/dataset.py` expects this structure (default root is `data/RHD_published_v2`):

```text
data/RHD_published_v2/
  training/
    color/
      00000.png
      00001.png
      ...
    anno_training.pickle
  evaluation/
    color/
      00000.png
      00001.png
      ...
    anno_evaluation.pickle
```

## Training

The script uses two stages:

1. Stage 1 trains backbone + heatmap head
2. Stage 2 trains coordinate head (and optionally fine-tunes other parts)

Outputs are written to:

- `training_results/<job-id>/losses.csv`
- `training_results/<job-id>/checkpoints/`

If `--job-id` is omitted, it uses `local`.

### Train (21 keypoints, default)

```bash
python scripts/train.py --root data/RHD_published_v2 --job-id exp_k21
```

### Train (10 keypoints: finger tips + bases)

```bash
python scripts/train.py --root data/RHD_published_v2 --job-id exp_k10 --tips-bases-only
```

### Hand Selection

`scripts/train.py` supports:

- `--hand right` (default)
- `--hand left`
- `--hand auto`

`src/handpose/data/dataset.py` first selects the hand from the 42 RHD landmarks, then applies the active keypoint mode.

## Slurm (Optional)

Use `slurm/train_1.sbatch` and edit these variables in the script first:

- `RHD_ROOT`
- `CHECKPOINT_ROOT`
- `CONDA_SOURCE`
- `CONDA_ENV_NAME`
- `USE_TIPS_BASES_ONLY=0` or `1`

Submit with:

```bash
sbatch slurm/train_1.sbatch
```

## Useful CLI Options

### Training (`scripts/train.py`)

- `--seed`: seed Python / NumPy / PyTorch for repeatable experiment runs
- `--accum-steps-stage1`, `--accum-steps-stage2`: gradient accumulation steps per stage
- `--lambda-hm`, `--lambda-coord`: stage-2 loss weights (`total = lambda_hm * heatmap + lambda_coord * coord`)
- `--heatmap-sigma`: Gaussian target sigma used by the heatmap loss
- `--wing-w`, `--wing-epsilon`: Wing loss shape parameters for the coordinate branch
- `--freeze-backbone-stage2`, `--freeze-heatmap-stage2`: freeze parts of the model during stage 2
- `--train-dataset-length N`: train on first `N` training samples (debug/ablation convenience)

### Evaluation (`scripts/eval_metrics.py`)

- `--device auto|cpu|cuda`: force evaluation device (default `auto`)
- `--prediction-mode fusion|heatmap|coord`: evaluate fused predictions or one branch alone
- `--batch-size`, `--num-workers`: evaluation loader settings
- `--debug-coords`: print first-batch coordinate diagnostics to `stderr`
- `--with-fusion-diagnostics`: export fusion-selection statistics in the result JSON

## Plot Training Curves

```bash
python scripts/plot_losses.py --job-id exp_k21
python scripts/plot_losses.py --job-id exp_k10
```

This reads `training_results/<job-id>/losses.csv` and writes `loss_plot.png` in the same run directory.

## Evaluate Trained Models (Metrics)

Use `scripts/eval_metrics.py` on the stage-2 checkpoint (`best.pt`) to report:

- visibility-masked normalized `SSE` per sample (`metrics.sse_norm`)
- visibility-masked normalized root-relative `EPE` (`metrics.epe_norm`)
- visibility-masked `PCK@sigma` (`metrics.pck`) with `--pck-threshold` (default `0.2`)
- sample/keypoint counts (`num_samples`, `num_points`, `num_visible_points`, `num_eval_keypoints`)
- prediction timing for the selected `--prediction-mode` (`ms_per_image`, `images_per_second`)

Note: `epe_norm` uses root keypoint `0` by default. For tip/base 10-joint checkpoints, it uses root keypoint `1`. In shared-10 eval of a 21-joint checkpoint, `epe_norm` is reported as `null`.

### Example: Evaluate a 21-keypoint model on all 21 keypoints

```bash
python scripts/eval_metrics.py --ckpt training_results/exp_k21/checkpoints/best.pt --root data/RHD_published_v2 --split evaluation --hand right --out-json eval_results/exp_k21_full.json
```

### Fair 21 vs 10 Comparison (recommended)

Evaluate both models on the same shared 10 landmarks:

```bash
python scripts/eval_metrics.py --ckpt training_results/exp_k21/checkpoints/best.pt --root data/RHD_published_v2 --split evaluation --hand right --shared-10-eval --out-json eval_results/exp_k21_shared10.json
```

```bash
python scripts/eval_metrics.py --ckpt training_results/exp_k10/checkpoints/best.pt --root data/RHD_published_v2 --split evaluation --hand right --shared-10-eval --out-json eval_results/exp_k10_shared10.json
```

Predictions in evaluation always use the fusion rule (`d_i < alpha`) from the shared inference pipeline.
Keep all settings the same across experiments (`--hand`, input size, dataset split, etc.).

## Single Image Inference

Use `scripts/predict_image.py` to run one image through a trained checkpoint and return predicted coordinates for the selected `--prediction-mode`.

### Predict coordinates (printed to stdout as JSON)

```bash
python scripts/predict_image.py --ckpt training_results/exp_k21/checkpoints/best.pt --image path/to/image.png
```

### Predict heatmap-only coordinates

```bash
python scripts/predict_image.py --ckpt training_results/exp_k21/checkpoints/best.pt --image path/to/image.png --prediction-mode heatmap
```

### Predict coordinates and save overlay image

```bash
python scripts/predict_image.py --ckpt training_results/exp_k21/checkpoints/best.pt --image path/to/image.png --overlay --overlay-out infer_results/image_overlay.png
```

## Jetson End-to-End Benchmark

Use `scripts/benchmark_pipeline.py` to benchmark end-to-end single-image latency on the full evaluation dataset.
The script:

- discovers images automatically from `<root>/evaluation/color/`
- sorts them by filename
- benchmarks the full discovered set
- writes one summary CSV row for the checkpoint
- saves the resolved image list next to the outputs for reproducibility

Example:

```bash
python scripts/benchmark_pipeline.py --ckpt training_results/exp_k21/checkpoints/best.pt --root data/RHD_published_v2 --summary-csv benchmark_results/summary.csv --output-root benchmark_results --device cuda --prediction-mode fusion --run-label orin_nano
```

Outputs:

- per-image prediction JSON files under `benchmark_results/<run-label>/...`
- aggregate CSV metrics in `--summary-csv`
- `resolved_images.txt` showing exactly which dataset files were benchmarked

The minimal benchmark measures:

- `image_read_ms`
- `preprocess_ms`
- `host_to_device_ms`
- `forward_predict_ms`
- `write_json_ms`
- `total_e2e_ms`

One-time model setup is reported separately as `session_setup_ms` in the summary CSV.

## Experiment Helpers

Use `scripts/generate_experiment_matrix.py` to emit a JSON/CSV experiment manifest plus train/eval/benchmark command files for the default research matrix.

Use `scripts/aggregate_results.py` to aggregate evaluation JSON files and benchmark summary CSVs into mean/std tables across seeds.

## Important Notes

- Use stage-2 `best.pt` for fused prediction metrics. Stage-1 checkpoints mainly train the heatmap branch.
- `21`-keypoint and `10`-keypoint checkpoints are not shape-compatible.
- Checkpoints must follow the current strict schema (`checkpoint_version`, `model_state`, `num_keypoints`, `keypoint_indices`, `stage`, `epoch`).
- Do not reuse the same `--job-id` for multiple experiments unless you want files appended/overwritten.
- `docs/commands.txt` contains old notes and experimental commands; prefer this README and the scripts themselves.
