# Experiment Command Templates

This folder contains the checked-in experiment matrices and command templates
for the ablation and transfer experiment families.

The command files are intentionally portable templates rather than machine-
specific exports. Replace the placeholder roots with paths on the target
system before running them.

Common placeholders:

- `<RHD_ROOT>`: root of the external Rendered Handpose Dataset
- `<HK26K_ROOT>`: root of the external HK26K / `coco_hand` dataset
- `<CHECKPOINT_ROOT>`: output root for ablation training checkpoints
- `<EVAL_ROOT>`: output root for ablation evaluation JSON files
- `<BENCHMARK_ROOT>`: output root for ablation benchmark results
- `<TRANSFER_CHECKPOINT_ROOT>`: output root for transfer-training checkpoints
- `<TRANSFER_EVAL_ROOT>`: output root for transfer evaluation JSON files
- `<TRANSFER_BENCHMARK_ROOT>`: output root for transfer benchmark results

The generator scripts that write these files are:

- `scripts/generate_experiment_matrix.py`
- `scripts/generate_transfer_experiment_matrix.py`
