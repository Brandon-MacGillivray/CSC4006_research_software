# Requirements

## Purpose

This document defines the software, hardware, dataset, and external artefact
requirements for the submitted CSC4006 software artefact. It describes what is
needed to run the repository correctly, which dependencies are mandatory for the
core workflows, and which requirements apply only to optional or environment-
specific tasks.

## Operating Assumptions

- A recent CPython 3 environment is required together with a PyTorch build that
  supports the selected Python version and platform.
- The repository was developed and run primarily from a Miniconda-managed Conda
  environment.
- CPU-only execution is possible for lightweight validation, script inspection,
  and some inference tasks.
- GPU acceleration is strongly recommended for training and for larger-scale
  evaluation workflows.
- The repository is structured for command-line execution from the repository
  root.

## Core Software Dependencies

The following dependencies are required for the main training, evaluation,
checkpoint, inference, and testing paths:

- `Python`
- `PyTorch`
- `NumPy`
- `Pillow`
- `mediapipe`
- `pytest`

These are directly required by the core package and command-line scripts under
`src/`, `scripts/`, and `tests/`.

No single environment file is supplied for this repository. Environments were
created manually for the required pipelines, because CUDA support, cluster
configuration, and optional dependencies can vary across target systems.

The PyTorch package itself was installed using commands selected from the
official PyTorch local-install page rather than from a fixed repository-managed
environment file.

## Additional Software Dependencies

The following dependencies are used for specific supporting workflows:

- `matplotlib`
  required for plotting and qualitative rendering scripts
- `psutil`
  optional enhancement for benchmark resource sampling in
  `scripts/benchmark_pipeline.py`

The repository does not currently require OpenCV for its active workflow set.

## Hardware Requirements

- `Minimum`
  a standard CPU-based development machine for document review, basic command
  inspection, and lightweight validation
- `Recommended`
  a CUDA-capable GPU system for model training and larger evaluation runs
- `Benchmark-specific`
  Jetson-class embedded hardware is required only if the embedded benchmarking
  story is being reproduced directly
- `Cluster-specific`
  a Slurm-managed environment is required only for the wrapper scripts under
  `slurm/`

## Dataset Requirements

### RHD

- Status: external dataset
- Role: primary synthetic training and evaluation dataset
- Requirement: must be supplied separately in the layout expected by the RHD
  loaders and evaluation scripts

### HK26K / `coco_hand`

- Status: external dataset
- Role: pseudo-labeled real-image dataset used for transfer-learning and
  cross-domain experiments
- Requirement: must be supplied separately in the COCO-style layout expected by
  the `coco_hand` dataset path in the training and evaluation scripts

### RH8

- Status: bundled artefact
- Role: small manually corrected real-image evaluation dataset used for
  supplementary benchmark and qualitative analysis
- Location: `docs/artefacts/datasets/rh8/`

## Model and Artefact Requirements

### Preserved Final Checkpoints

- Status: bundled artefact
- Role: archived final checkpoints used for the reported experiments
- Location: `docs/artefacts/checkpoints/`
- Notes: the archive includes a folder-level `README.md` and
  `checkpoint_manifest.csv`

### MediaPipe Model Asset

- Status: bundled artefact
- Role: model asset for MediaPipe baseline workflows
- Location: `src/handpose/models/hand_landmarker.task`

## Workflow-Specific Requirements

### Training

Required:

- Python
- PyTorch
- NumPy
- target dataset (`RHD` or `coco_hand`)

Recommended:

- CUDA-capable GPU

### Evaluation

Required:

- Python
- PyTorch
- NumPy
- `mediapipe`
- checkpoint file
- relevant dataset root

### Benchmarking

Required:

- Python
- PyTorch
- NumPy
- Pillow
- checkpoint file
- relevant dataset root

Optional but useful:

- `psutil`

Environment-specific:

- Jetson-specific tooling and hardware if reproducing embedded runtime results

### Qualitative Rendering

Required:

- Python
- PyTorch
- NumPy
- Pillow
- `matplotlib`
- `mediapipe`
- checkpoint file
- relevant dataset root

### Automated Tests

Required:

- Python
- PyTorch
- `pytest`

The current lightweight test suite validates checkpoint schema handling, fusion
logic, and model output-shape contracts without requiring external datasets.

## Storage Requirements

- The preserved final checkpoint archive occupies approximately `2 GB`.
- Additional storage is required for external datasets, generated evaluation
  outputs, benchmark logs, and any regenerated qualitative figures.

## Limitations and External Infrastructure

- Full reproduction of all reported experiments depends on access to the
  external `RHD` and `HK26K / coco_hand` datasets.
- Full embedded benchmarking depends on hardware and tools not bundled in this
  repository.
- Slurm execution depends on external scheduler infrastructure and is not
  required for local command-line use.
