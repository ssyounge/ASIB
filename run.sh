#!/bin/bash
# run.sh
set -euo pipefail
#SBATCH --job-name=run_asmb_experiment
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/asmb_%j/run.log
#SBATCH --error=outputs/asmb_%j/run.log

# Ensure script runs from repository root
# Use SLURM_SUBMIT_DIR when running under sbatch. This ensures paths
# resolve relative to the directory the job was submitted from.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

source ~/.bashrc
conda activate tlqkf

# Ensure log directory exists
: "${SLURM_JOB_ID:=manual}"
mkdir -p "outputs/asmb_${SLURM_JOB_ID}"

python main.py --config-path configs/experiment --config-name res152_effi_b2
