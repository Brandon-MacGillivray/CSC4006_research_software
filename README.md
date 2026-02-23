# DRHand Replica (RHD 2D Hand Pose Training)

PyTorch training code for a dual-branch hand pose model (heatmaps + coordinates), inspired by the DRHand paper (`Dual Regression for Efficient Hand Pose Estimation`).

This repo trains on the RHD dataset and supports two keypoint modes:

- `21` keypoints (default)
- `10` keypoints (finger tips + bases) via `--tips-bases-only`

## Files

- `train_1.py`: main two-stage training script
- `train_1.sbatch`: Slurm job script
- `dataset.py`: RHD dataset loader (`left` / `right` / `auto` hand selection)
- `architecture.py`: backbone + heatmap head + coordinate head
- `losses.py`: heatmap MSE and Wing loss
- `plot_losses.py`: plot train/val losses from `losses.csv`
- `eval_metrics.py`: checkpoint evaluation (`EPE`, `SSE`, `PCK`, timing)
- `commands.txt`: scratch notes / old commands (not source of truth)

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

`dataset.py` expects this structure (default root is `data/RHD_published_v2`):

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
python train_1.py --root data/RHD_published_v2 --job-id exp_k21
```

### Train (10 keypoints: finger tips + bases)

```bash
python train_1.py --root data/RHD_published_v2 --job-id exp_k10 --tips-bases-only
```

### Hand Selection

`train_1.py` supports:

- `--hand right` (default)
- `--hand left`
- `--hand auto`

`dataset.py` first selects the hand from the 42 RHD landmarks, then applies the active keypoint mode.

## Slurm (Optional)

Use `train_1.sbatch` and set these variables first:

- `RHD_ROOT`
- `CHECKPOINT_ROOT`
- `CONDA_SOURCE`
- `CONDA_ENV_NAME`
- `USE_TIPS_BASES_ONLY=0` or `1`

Submit with:

```bash
sbatch train_1.sbatch
```

Note: `train_1.sbatch` currently runs `python handpose/train_1.py`. If this repo is your working directory root, change it to `python train_1.py`.

## Plot Training Curves

```bash
python plot_losses.py --job-id exp_k21
python plot_losses.py --job-id exp_k10
```

This reads `training_results/<job-id>/losses.csv` and writes `loss_plot.png` in the same run directory.

## Evaluate Trained Models (Metrics)

Use `eval_metrics.py` on the stage-2 checkpoint (`best.pt`) to compute:

- `EPE` (normalized and pixels)
- `SSE` (normalized)
- `PCK` at configurable thresholds
- forward-only timing / FPS

### Example: Evaluate a 21-keypoint model on all 21 keypoints

```bash
python eval_metrics.py --ckpt training_results/exp_k21/checkpoints/best.pt --root data/RHD_published_v2 --split evaluation --hand right --out-json eval_results/exp_k21_full.json
```

### Fair 21 vs 10 Comparison (recommended)

Evaluate both models on the same shared 10 landmarks:

```bash
python eval_metrics.py --ckpt training_results/exp_k21/checkpoints/best.pt --root data/RHD_published_v2 --split evaluation --hand right --shared-10-eval --out-json eval_results/exp_k21_shared10.json
```

```bash
python eval_metrics.py --ckpt training_results/exp_k10/checkpoints/best.pt --root data/RHD_published_v2 --split evaluation --hand right --shared-10-eval --out-json eval_results/exp_k10_shared10.json
```

Keep all settings the same across experiments (`--hand`, input size, dataset split, etc.).

## Important Notes

- Use stage-2 `best.pt` for coordinate metrics. Stage-1 checkpoints mainly train the heatmap branch.
- `21`-keypoint and `10`-keypoint checkpoints are not shape-compatible.
- Do not reuse the same `--job-id` for multiple experiments unless you want files appended/overwritten.
- `commands.txt` contains old notes and experimental commands; prefer this README and the scripts themselves.
