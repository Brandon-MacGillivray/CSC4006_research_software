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
