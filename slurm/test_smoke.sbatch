#!/bin/bash
#SBATCH --job-name=drh-smoke
#SBATCH --partition=k2-hipri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Minimal Slurm smoke test. This does not use the repo code or a GPU.
# Submit from home with:
#   sbatch handpose/slurm/test_smoke.sbatch
#
# To test email notifications as well:
#   sbatch --mail-user=bmacgillivray01@qub.ac.uk --mail-type=END,FAIL handpose/slurm/test_smoke.sbatch

echo "===== SLURM SMOKE TEST ====="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Job Name:    ${SLURM_JOB_NAME}"
echo "Node:        $(hostname)"
echo "Start Time:  $(date --iso-8601=seconds)"
echo "Submit Dir:  ${SLURM_SUBMIT_DIR}"
echo "Working Dir: $(pwd)"
echo "============================"

# Sleep briefly so the job stays visible in the queue and emits a clean log.
sleep 10

echo "Finished at $(date --iso-8601=seconds)"
