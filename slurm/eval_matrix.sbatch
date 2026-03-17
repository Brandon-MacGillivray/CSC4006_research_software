#!/bin/bash
#SBATCH --job-name=drh-eval-matrix
#SBATCH --partition=k2-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bmacgillivray01@qub.ac.uk
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# Edit the mail address above, or override it at submit time with:
#   sbatch --mail-user=you@example.com --mail-type=END,FAIL handpose/slurm/eval_matrix.sbatch
#
# Run one generated evaluation command per Slurm array task.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
COMMAND_FILE="${COMMAND_FILE:-${REPO_DIR}/experiment_plan/eval_commands.txt}"
CONDA_SOURCE="${CONDA_SOURCE:-/users/40321491/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-torch}"

# Enter the repo. Slurm stdout/stderr paths remain relative to the submit directory.
cd "${REPO_DIR}"

# Load the requested Conda environment before dispatching the generated command.
source "${CONDA_SOURCE}"
conda activate "${CONDA_ENV_NAME}"

# The matrix must already have been generated into a plain text command file.
if [[ ! -f "${COMMAND_FILE}" ]]; then
  echo "Missing command file: ${COMMAND_FILE}" >&2
  exit 1
fi

# Ignore blank/comment lines so the array size maps to real commands only.
mapfile -t COMMANDS < <(awk 'NF && $0 !~ /^[[:space:]]*#/' "${COMMAND_FILE}")
NUM_COMMANDS="${#COMMANDS[@]}"

# This wrapper is intended for `sbatch --array=...`; fail early otherwise.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script must be submitted as a Slurm array job." >&2
  exit 1
fi

# Convert Slurm's 1-based task ID to a 0-based Bash array index.
INDEX=$((SLURM_ARRAY_TASK_ID - 1))
if (( INDEX < 0 || INDEX >= NUM_COMMANDS )); then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is out of range for ${NUM_COMMANDS} commands." >&2
  exit 1
fi

# Select exactly one evaluation command for this task.
CMD="${COMMANDS[$INDEX]}"

echo "===== SLURM JOB INFO ====="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Array Task:   ${SLURM_ARRAY_TASK_ID}/${NUM_COMMANDS}"
echo "Node:         $(hostname)"
echo "Start Time:   $(date --iso-8601=seconds)"
echo "Repo Dir:     ${REPO_DIR}"
echo "Command File: ${COMMAND_FILE}"
echo "Conda Env:    ${CONDA_ENV_NAME}"
echo "Python:       $(python -V 2>&1)"
echo "Command:      ${CMD}"
nvidia-smi -L || true
echo "=========================="

# Run the generated command in a login-capable shell so conda/python resolution matches interactive use.
set -x
srun bash -lc "${CMD}"
set +x

echo "Finished at $(date --iso-8601=seconds)"
