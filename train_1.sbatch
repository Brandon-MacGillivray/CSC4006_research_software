#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=k2-gpu-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR}"
RHD_ROOT="/users/40321491/data/RHD_published_v2"
CHECKPOINT_ROOT="/users/40321491/sharedscratch/training_results"
CONDA_SOURCE="/users/40321491/miniconda/etc/profile.d/conda.sh"
CONDA_ENV_NAME="torch"

INPUT_SIZE=256
HAND="right"
NUM_WORKERS=4
TRAIN_DATASET_LENGTH=0

BATCH_SIZE_STAGE1=64
BATCH_SIZE_STAGE2=64
ACCUM_STEPS_STAGE1=1
ACCUM_STEPS_STAGE2=4
STAGE1_EPOCHS=50
STAGE2_EPOCHS=50
LR_STAGE1=1e-3
LR_STAGE2=1e-4
STAGE1_PATIENCE=5
STAGE1_MIN_DELTA=0.0
LAMBDA_HM=1.0
LAMBDA_COORD=1.0

USE_TIPS_BASES_ONLY=0
FREEZE_BACKBONE_STAGE2=0
FREEZE_HEATMAP_STAGE2=0

cd "${REPO_DIR}"

source "${CONDA_SOURCE}"
conda activate "${CONDA_ENV_NAME}"

echo "===== SLURM JOB INFO ====="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "Start Time:   $(date --iso-8601=seconds)"
echo "Repo Dir:     ${REPO_DIR}"
echo "Dataset Root: ${RHD_ROOT}"
echo "Checkpoint:   ${CHECKPOINT_ROOT}"
echo "Conda Env:    ${CONDA_ENV_NAME}"
echo "Python:       $(python -V 2>&1)"
nvidia-smi -L || true
echo "=========================="

CMD=(
  python handpose/train_1.py
  --root "${RHD_ROOT}"
  --checkpoint-root "${CHECKPOINT_ROOT}"
  --job-id "${SLURM_JOB_ID}"
  --input-size "${INPUT_SIZE}"
  --batch-size-stage1 "${BATCH_SIZE_STAGE1}"
  --batch-size-stage2 "${BATCH_SIZE_STAGE2}"
  --accum-steps-stage1 "${ACCUM_STEPS_STAGE1}"
  --accum-steps-stage2 "${ACCUM_STEPS_STAGE2}"
  --stage1-epochs "${STAGE1_EPOCHS}"
  --stage2-epochs "${STAGE2_EPOCHS}"
  --lr-stage1 "${LR_STAGE1}"
  --lr-stage2 "${LR_STAGE2}"
  --stage1-patience "${STAGE1_PATIENCE}"
  --stage1-min-delta "${STAGE1_MIN_DELTA}"
  --hand "${HAND}"
  --num-workers "${NUM_WORKERS}"
  --train-dataset-length "${TRAIN_DATASET_LENGTH}"
  --lambda-hm "${LAMBDA_HM}"
  --lambda-coord "${LAMBDA_COORD}"
)

if [[ "${USE_TIPS_BASES_ONLY}" == "1" ]]; then
  CMD+=(--tips-bases-only)
fi

if [[ "${FREEZE_BACKBONE_STAGE2}" == "1" ]]; then
  CMD+=(--freeze-backbone-stage2)
fi

if [[ "${FREEZE_HEATMAP_STAGE2}" == "1" ]]; then
  CMD+=(--freeze-heatmap-stage2)
fi

set -x
srun "${CMD[@]}"
set +x

echo "Finished at $(date --iso-8601=seconds)"