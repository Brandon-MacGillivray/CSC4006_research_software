# Requirements

## Overview

This repository is a research software artefact for training, evaluating, benchmarking,
and demonstrating a DRHand-inspired 2D hand-pose model. Source code is included in
the repository, but datasets, trained checkpoints, and benchmark hardware are external.

## Hardware Requirements

### Minimum

- 64-bit machine capable of running Python 3.10+
- At least 8 GB RAM for light inspection, script execution, and small-scale tests
- Sufficient storage for datasets and generated outputs

### Recommended for Training

- NVIDIA GPU with CUDA support
- At least 8 GB GPU memory for practical training runs
- 16 GB or more system RAM

### Required to Reproduce Reported Latency Numbers

- NVIDIA Jetson Orin Nano for Jetson-specific benchmarking
- JetPack/CUDA environment matching the target Jetson device

### Optional Hardware

- Webcam for `webcam_demo.py`
- External peripherals such as the ZED stereo camera are not required for the current
  RGB-only code path in this repository

## Software Requirements

### Operating System

- Linux is recommended for training, Slurm use, and Jetson benchmarking
- Windows can be used for local development and most CLI workflows
- macOS may work for CPU-only workflows, but has not been the target environment

### Python

- Python 3.10 or newer
- The current README notes that the repository has been tested with Python 3.12 syntax
- Conda is the recommended environment manager for local installation

### Python Packages

The repository provides:

- `environment.yml` for Conda environment creation
- `requirements.txt` for pip-managed dependencies and CI use

Base dependencies include:

- `torch`
- `torchvision`
- `numpy`
- `Pillow`
- `matplotlib`
- `psutil`

Optional dependencies:

- `mediapipe` for the MediaPipe baseline evaluation script
- `opencv-python` for `webcam_demo.py`

## External Data and Model Assets

The following artefacts are not bundled with the repository:

- RHD dataset
- COCO-style hand-keypoint dataset
- MediaPipe hand landmarker `.task` model for the baseline script

The expected on-disk layouts for RHD and COCO-hand are described in `README.md`.

## Storage Requirements

Storage usage depends on the size of external datasets and the number of experiments run.
Plan for storage in the following categories:

- raw datasets
- checkpoints under `training_results/` or `transfer_training_results/`
- evaluation JSON outputs
- benchmark JSON outputs and summary CSVs
- aggregated CSV outputs used for the report

Repeated matrix runs can generate a large number of checkpoints and output files.

## Environment and Path Assumptions

Some helper scripts assume a layout in which:

- datasets live under a sibling `data/` directory
- generated experiment outputs live under a sibling `sharedscratch/` directory

These defaults can be overridden with command-line arguments, so the repository can still
be used in other layouts.

## Notes on Reproducibility

- Use stage-2 `best.pt` checkpoints for fused evaluation and benchmarking
- `21`-keypoint and `10`-keypoint checkpoints are not shape-compatible
- Jetson latency figures are hardware-specific and should not be expected to match on a
  desktop CPU or GPU
