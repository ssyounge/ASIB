#!/bin/bash
# run.sh
#SBATCH --job-name=run_asmb_experiment
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=outputs/asmb_%j/run.log
#SBATCH --error=outputs/asmb_%j/run.log

# Create a unique output directory per SLURM job
JOB_ID=${SLURM_JOB_ID:-manual}
OUTPUT_DIR="outputs/asmb_${JOB_ID}"
mkdir -p "$OUTPUT_DIR"

source ~/.bashrc
conda activate facil_env

# Launch experiments defined in configs/run_setup.yaml
python scripts/launch_experiments.py --setup configs/run_setup.yaml
