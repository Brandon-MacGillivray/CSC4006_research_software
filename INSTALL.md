# Installation

## 1. Clone the Repository

```bash
git clone <repo-url>
cd CSC4006_DRHand_replica
```

## 2. Create a Python Environment

Create the Conda environment from the repository definition:

```bash
conda env create -f environment.yml
```

Activate the environment:

Linux/macOS:

```bash
conda activate drhand
```

Windows PowerShell:

```powershell
conda activate drhand
```

## 3. Install Base Dependencies

The base Conda environment installs the dependencies declared in `environment.yml`.
If you need to refresh the pip-managed packages inside that environment, run:

```bash
pip install -r requirements.txt
```

Optional dependencies:

```bash
pip install mediapipe
pip install opencv-python
```

Install `mediapipe` only if you need the MediaPipe baseline scripts.
Install `opencv-python` only if you need `webcam_demo.py`.

## 4. Verify the Installation

The quickest smoke test is to run the command-line help for the main entry points.
These commands exercise imports and argument parsing without requiring datasets.

```bash
python scripts/train.py --help
python scripts/eval_metrics.py --help
python scripts/benchmark_pipeline.py --help
python scripts/predict_image.py --help
```

Expected result:

- each command prints a usage message
- the command exits without a Python traceback

Repository smoke test:

```bash
python scripts/smoke_test.py
```

Expected result:

- the script runs the main non-optional CLIs with `--help`
- the script exits with `Smoke test passed.`

Minimal automated tests:

```bash
pytest -q
```

Expected result:

- the synthetic unit tests under `tests/` pass
- no external dataset is required for these tests

Optional feature checks:

```bash
python scripts/eval_mediapipe_rhd.py --help
python webcam_demo.py --help
```

These require the optional dependencies listed above.

## 5. Set Up Datasets

This repository does not include datasets. Arrange them to match the layouts described in
`README.md`.

Typical roots:

- RHD: `data/RHD_published_v2`
- COCO-hand: `data/hand_keypoint_dataset`

You may place datasets elsewhere and override the path with `--root`.

## 6. Minimal End-to-End Smoke Test

If the RHD dataset is available, a very small training run can be used as a stronger
installation check:

```bash
python scripts/train.py \
  --dataset rhd \
  --root data/RHD_published_v2 \
  --job-id smoke_rhd \
  --train-dataset-length 16 \
  --batch-size-stage1 8 \
  --batch-size-stage2 8 \
  --stage1-epochs 1 \
  --stage2-epochs 1 \
  --num-workers 0
```

Expected result:

- a run directory is created under `training_results/smoke_rhd/`
- `losses.csv` is written
- checkpoints are written under `training_results/smoke_rhd/checkpoints/`

## 7. Optional Single-Image Inference

If you already have a trained checkpoint and an input image:

```bash
python scripts/predict_image.py \
  --ckpt training_results/<job-id>/checkpoints/best.pt \
  --image path/to/image.png \
  --overlay
```

Expected result:

- JSON prediction output printed to the terminal
- an overlay image written beside the input image unless `--overlay-out` is provided

## 8. Notes

- `scripts/train.py` automatically uses CUDA if available; otherwise it falls back to CPU
- the repository uses Conda for environment creation; `requirements.txt` is retained for
  pip-based refreshes and CI
- Jetson benchmarking requires the target Jetson device and appropriate system tools
- Slurm wrappers under `slurm/` are intended for cluster execution and are not required
  for local use
