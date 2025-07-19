#!/bin/bash
# run.sh
#SBATCH --job-name=run_asmb_experiment
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=outputs/asmb_%j/run.log
#SBATCH --error=outputs/asmb_%j/run.log

# tqdm 완전 OFF
export PROGRESS=0

# Create a unique output directory per SLURM job
JOB_ID=${SLURM_JOB_ID:-manual}
OUTPUT_DIR="outputs/asmb_${JOB_ID}"
mkdir -p "$OUTPUT_DIR"

source ~/.bashrc
conda activate tlqkf

# Launch experiments using the unified script
bash scripts/run_experiments.sh --mode loop --output_dir "$OUTPUT_DIR"
