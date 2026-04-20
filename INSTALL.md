# Installation

## Purpose

This document explains how to prepare a Python environment for the submitted
software artefact, install the required dependencies, and perform an initial
validation of the repository.

## Repository Preparation

Clone or unpack the repository and change into the repository root before
running any commands.

Example:

```bash
git clone https://github.com/Brandon-MacGillivray/CSC4006_DRHand_replica.git
cd CSC4006_DRHand_replica
conda create -n torch python=3.12
conda activate torch
```

## Python Environment Setup

This repository was developed and run primarily using a Miniconda-managed Conda
environment. That is the recommended setup path for this artefact.

### Miniconda / Conda

Create and activate a dedicated Conda environment manually:

```bash
conda create -n torch python=3.12
conda activate torch
```

If Miniconda or Conda is not available, another isolated Python environment can
be used, but the documented and preferred path for this artefact is Conda.

## Install Core Dependencies

Install a PyTorch build appropriate to the target machine first, then install
the remaining core packages.

PyTorch:

- this project used the official PyTorch local-install selector:
  `https://pytorch.org/get-started/locally/`
- install a build compatible with the selected Python version, operating system,
  and CUDA/runtime configuration
- for GPU training or evaluation, use a CUDA-enabled build that matches the
  target machine or cluster
- for lightweight validation only, a CPU-only build is acceptable

In practice, the PyTorch installation command should be taken directly from the
official selector for the target platform rather than hard-coded here, because
the correct command depends on the chosen Python, operating system, package
manager, and CUDA configuration.

Core packages:

```bash
python -m pip install numpy pillow pytest mediapipe
```

Recommended all-at-once non-PyTorch package installation:

```bash
python -m pip install numpy pillow pytest matplotlib mediapipe psutil
```

## Install Additional Dependencies

Install the following packages if the corresponding supporting workflows are
needed:

```bash
python -m pip install matplotlib psutil
```

These support:

- `matplotlib`
  plotting and qualitative rendering
- `psutil`
  additional process-level benchmarking metrics

## PyTorch Verification

After installing PyTorch from the official selector, verify that the package can
be imported and that the selected build matches the available hardware:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected result:

- PyTorch imports without error
- a version string is printed
- `torch.cuda.is_available()` reports `True` on a correctly configured GPU
  system, or `False` on a CPU-only setup

## First Validation

After the environment is prepared, run the lightweight test suite:

```bash
python -m pytest tests -q
```

Then confirm that the main entry points are visible and import correctly:

```bash
python scripts/train.py --help
python scripts/eval_metrics.py --help
python scripts/predict_image.py --help
```

Expected result:

- the test suite completes without failures
- each command prints usage information and exits successfully

## Example Inference With Bundled Artefacts

One concrete repository-level example can be run using a bundled RH8 image and a
preserved final checkpoint:

```bash
python scripts/predict_image.py ^
  --ckpt docs/artefacts/checkpoints/drh-b0-k21-right-s101.best.pt ^
  --image docs/artefacts/datasets/rh8/images/val/WIN_20260405_12_56_28_Pro.jpg ^
  --prediction-mode fusion ^
  --overlay
```

Expected result:

- a JSON prediction payload is printed to the terminal
- an overlay image is written next to the source image unless `--overlay-out` is
  supplied explicitly

## Bundled Artefacts Available Immediately

The following artefacts are already included in the repository and do not need
to be downloaded separately:

- the RH8 evaluation artefact under `docs/artefacts/datasets/rh8/`
- the preserved final checkpoint archive under `docs/artefacts/checkpoints/`
- the MediaPipe model asset at `src/handpose/models/hand_landmarker.task`

This means the repository can be inspected and partially validated without first
recreating the full training environment on external datasets.

## External Dataset Setup

The full experimental workflows require external datasets that are not bundled
in the repository.

### RHD

- required for synthetic-domain training and evaluation
- supply the dataset root expected by the RHD training and evaluation scripts

### HK26K / `coco_hand`

- required for pseudo-labeled real-image transfer-learning workflows
- supply the dataset root in the COCO-style layout expected by the `coco_hand`
  training and evaluation scripts

## Using the Preserved Checkpoints

The repository includes the final checkpoints used for the reported experiments.
These are stored under:

- `docs/artefacts/checkpoints/`

The archive includes:

- `README.md`
- `checkpoint_manifest.csv`

These checkpoints can be used directly for evaluation, qualitative rendering,
and benchmarking workflows without retraining the models from scratch.

## Optional Workflow Notes

### MediaPipe Baseline

Keep the bundled `hand_landmarker.task` asset in its current repository path.
The `mediapipe` package should already be present as a required dependency.

### Benchmarking

The benchmark pipeline can run locally, but full embedded benchmarking depends
on hardware-specific tools and environments not bundled in the repository.

### Slurm Execution

The scripts under `slurm/` are optional wrappers intended for cluster-based
matrix execution. They are not required for local use of the repository.

The Slurm wrappers default to:

- `CONDA_ENV_NAME=torch`

If a different Conda environment name is used on the target system, override it
at submission time.

## Known Installation Limitations

- If `torch` is not installed, the test suite will fail during import.
- GPU support depends on the selected PyTorch build rather than the repository
  itself.
- Full reproduction of all experiments still depends on the external `RHD` and
  `HK26K / coco_hand` datasets.

## Troubleshooting

- `PyTorch import fails`
  ensure that the Conda environment is activated and that PyTorch was installed
  using a command from the official selector that matches the current platform
- `torch.cuda.is_available()` returns `False`
  verify that the installed PyTorch build, NVIDIA driver, and target CUDA setup
  are compatible
- `MediaPipe import fails`
  reinstall `mediapipe` into the active environment and confirm that the active
  Conda environment is the one intended for this repository
- `Tests fail during collection`
  confirm that the intended Conda environment is active and includes `torch` and
  `pytest`
- `Dataset-dependent scripts fail`
  verify that the external dataset root passed to the command matches the
  expected layout for `RHD` or `coco_hand`
