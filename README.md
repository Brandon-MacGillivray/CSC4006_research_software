# Lightweight Hand Pose Estimation for Edge HCI

## Summary

This repository is submitted as the software artefact for a CSC4006 Research and
Development Project on lightweight hand pose estimation for edge HCI. It implements
a DRHand-inspired 2D hand pose estimation pipeline with dual-branch heatmap and
coordinate prediction, together with fused inference for final evaluation.

The artefact supports training, evaluation, benchmarking, qualitative rendering,
single-image inference, experiment-matrix generation, and result aggregation.
Experiments are supported on the Rendered Handpose Dataset (RHD) and on RH8, a
small manually corrected real-image hand dataset represented in COCO-style format.

The repository includes source code, runnable scripts, experiment definitions,
paper assets, curated non-code artefacts, and an archived set of final checkpoints
used for the reported experiments. External source datasets and hardware-specific
benchmark environments are provided separately where required.

## Repository Access

The development repository for this artefact is hosted at:

- `https://github.com/Brandon-MacGillivray/CSC4006_DRHand_replica.git`

This submitted repository snapshot was prepared from that development repository
as the assessment artefact for CSC4006. The repository contains the cleaned
software, supporting documentation, curated artefacts, and preserved final
checkpoints required for assessment and reproducibility.

## Included and External Artefacts

Included in this repository:

- source code under `src/`
- runnable scripts under `scripts/`
- experiment definitions under `experiments/`
- paper assets under `docs/paper/`
- curated figures, results, commands, and bundled artefacts under `docs/`
- the preserved archive of final checkpoints used for the reported experiments

Provided separately where required:

- the full external RHD source dataset
- hardware-specific benchmark environments, including Jetson-based runtime setups
- any local cluster or scheduler infrastructure needed for Slurm-based execution

## Quick Start

After creating a Miniconda-managed Conda environment with the required
dependencies, the quickest way to verify the repository is to run the
lightweight automated tests:

```bash
conda create -n torch python=3.12
conda activate torch
```

Then run:

```bash
python -m pytest tests -q
```

The main command-line entry points can then be checked directly:

```bash
python scripts/train.py --help
python scripts/eval_metrics.py --help
python scripts/predict_image.py --help
```

Expected result:

- the test suite completes without failures
- each command prints its usage information and exits successfully

This quick-start path validates the core Python environment, the local import
layout, and the main runnable scripts before attempting dataset-dependent
training or evaluation workflows.

PyTorch itself should be installed using the official selector at
`https://pytorch.org/get-started/locally/` so that the chosen build matches the
target platform and CUDA configuration.

## Validation

Validation of the software artefact is intended to happen at three levels.

1. Environment validation:
   the lightweight `pytest` suite checks checkpoint schema handling, fusion
   logic, and core model output contracts without requiring external datasets.

2. Script-entry validation:
   the main runnable scripts should execute successfully with `--help`, showing
   that the repository import layout and command-line interfaces are intact.

3. Workflow validation:
   dataset-dependent training, evaluation, benchmarking, and qualitative
   rendering workflows should produce outputs consistent with the curated
   artefacts preserved under `docs/artefacts/results/` and
   `docs/figures/qualitative/`.

For full experimental validation, the preserved checkpoints in
`docs/artefacts/checkpoints/` can be used together with the relevant datasets
to rerun evaluation and comparison workflows.

## Repository Structure

- `src/`
  reusable Python package implementing data handling, model definition,
  inference, evaluation, checkpoint handling, and training utilities
- `scripts/`
  command-line entry points for training, evaluation, benchmarking, prediction,
  qualitative rendering, aggregation, and experiment-matrix generation
- `experiments/`
  ablation and transfer experiment definitions together with generated command files
- `docs/paper/`
  the submitted research article source, compiled PDF, and paper figures
- `docs/figures/`
  curated qualitative figures used for assessment and reporting
- `docs/artefacts/`
  preserved checkpoints, bundled RH8 dataset artefacts, curated results, and
  supporting command files
- `slurm/`
  cluster-oriented wrapper scripts for matrix-based execution

## Main Workflows

- `Training`
  train DRHand-style models on supported datasets using `scripts/train.py`
- `Evaluation`
  evaluate learned checkpoints with `scripts/eval_metrics.py`
- `Benchmarking`
  run latency and runtime measurements with `scripts/benchmark_pipeline.py`
- `Single-image inference`
  generate predictions for individual images with `scripts/predict_image.py`
- `Qualitative rendering`
  render comparison figures for RH8 and RHD examples with
  `scripts/render_qualitative_coco_hand.py` and `scripts/render_qualitative_rhd.py`
- `Result aggregation`
  aggregate curated result artefacts with `scripts/aggregate_results.py`
- `Experiment generation`
  regenerate ablation and transfer command matrices with
  `scripts/generate_experiment_matrix.py` and
  `scripts/generate_transfer_experiment_matrix.py`

## Datasets and Assets

- `RHD`
  the full Rendered Handpose Dataset is treated as an external source dataset
  and must be supplied separately in the expected layout for dataset-dependent
  training and evaluation workflows
- `HK26K / coco_hand`
  the pseudo-labeled real-image hand dataset used for transfer-learning and
  cross-domain experiments is treated as an external source dataset and is
  referred to in the code as `coco_hand`
- `RH8`
  the bundled real-image RH8 artefact is provided under
  `docs/artefacts/datasets/rh8/` with `images/` and COCO-style annotation files
- `Preserved checkpoints`
  the final checkpoints used for the reported experiments are archived under
  `docs/artefacts/checkpoints/` together with a manifest and archive notes
- `MediaPipe asset`
  the MediaPipe hand landmarker model asset used by the baseline scripts is
  included at `src/handpose/models/hand_landmarker.task`

## Supporting Documents

The current repository snapshot includes the following supporting documents and
reference artefacts:

- `REQUIREMENTS.md`
  software, hardware, dataset, and artefact requirements for the repository
- `INSTALL.md`
  environment setup, dependency installation, and first-run validation steps
- `REPLICATION_GUIDE.md`
  replication pathways for bundled artefacts and external-dataset reruns
- `LICENSE`
  MIT licence for the submitted repository contents unless otherwise noted
- `docs/README.md`
  overview of the non-code documentation and artefact layout under `docs/`
- `docs/paper/hand_pose_article.tex`
  submitted research article source
- `docs/paper/hand_pose_article.pdf`
  compiled research article PDF
- `docs/artefacts/checkpoints/README.md`
  description of the preserved checkpoint archive
- `docs/artefacts/checkpoints/checkpoint_manifest.csv`
  manifest of the archived final checkpoints used for the reported experiments

## License

This repository is released under the MIT License. See `LICENSE` for the full
licence text.
