# Checkpoint Archive

This folder contains the preserved final model checkpoints used to support the
reported results in this repository.

## Scope

- The archive contains `70` final checkpoint files copied from the Kelvin2 path
  `/users/40321491/sharedscratch/training_results/`.
- Each archived file is the final stage-2 checkpoint formerly stored as
  `training_results/<job_id>/checkpoints/best.pt`.
- The files were renamed to stable archive names of the form
  `<job_id>.best.pt` so they can be stored together without recreating the full
  training-output directory structure.

## Purpose

These checkpoints are preserved as submission artefacts for reproducibility and
assessment. They are the final checkpoints required to support:

- core accuracy comparisons
- branch ablation
- transfer-learning results
- MediaPipe comparison
- runtime benchmarking
- fusion diagnostics
- qualitative comparisons
- hyperparameter sensitivity analysis

This folder is an archive of final paper-relevant checkpoints only. It does not
attempt to preserve all intermediate epochs, logs, or full training-output
directories from Kelvin2.

## Contents

- `checkpoint_manifest.csv`: manifest listing each archived checkpoint, its
  experiment family, seed, keypoint layout, file size, SHA256 hash, and report
  usage
- `*.best.pt`: preserved final checkpoints

## Usage

The archived files can be used directly with repository scripts by passing the
full file path, for example:

```bash
python scripts/eval_metrics.py --ckpt docs/artefacts/checkpoints/drh-b0-k21-right-s101.best.pt ...
```

```bash
python scripts/predict_image.py --ckpt docs/artefacts/checkpoints/trf-r-to-c-s505.best.pt ...
```

## Notes

- The archive is intentionally separate from `training_results/` because these
  files are preserved artefacts rather than live run outputs.
- SHA256 hashes are included in the manifest so the archive can be verified
  after transfer or submission packaging.
