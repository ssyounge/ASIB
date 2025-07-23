#!/bin/bash
# run.sh
#SBATCH --job-name=run_asmb_experiment
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/asmb_%j/run.log
#SBATCH --error=outputs/asmb_%j/run.log

source ~/.bashrc
conda activate tlqkf

# Ensure log directory exists
: "${SLURM_JOB_ID:=manual}"
mkdir -p "outputs/asmb_${SLURM_JOB_ID}"

python main.py --config-path configs/experiment --config-name res152_effi_b2
