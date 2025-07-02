#!/bin/bash
# run.sh
#SBATCH --job-name=run_ibkd_experiment
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=outputs/ibkd_%j/run.log
#SBATCH --error=outputs/ibkd_%j/run.log

# Create a unique output directory per SLURM job
JOB_ID=${SLURM_JOB_ID:-manual}
OUTPUT_DIR="outputs/ibkd_${JOB_ID}"
mkdir -p "$OUTPUT_DIR"

source ~/.bashrc
conda activate facil_env

# Launch IBKD experiments
bash scripts/run_ibkd.sh --mode loop --output_dir "$OUTPUT_DIR"
