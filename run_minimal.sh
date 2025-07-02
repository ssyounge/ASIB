#!/bin/bash
# run_minimal.sh
#SBATCH --job-name=minimal_kd
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=outputs/minimal_%j/run.log
#SBATCH --error=outputs/minimal_%j/run.log

# Paths to teacher checkpoints
TEACHER1_CKPT="/path/to/teacher1.pth"
TEACHER2_CKPT="/path/to/teacher2.pth"

# Create unique output directory per job
JOB_ID=${SLURM_JOB_ID:-manual}
OUTPUT_DIR="outputs/minimal_${JOB_ID}"
mkdir -p "$OUTPUT_DIR"

source ~/.bashrc
conda activate facil_env

# Copy checkpoints to expected locations
mkdir -p checkpoints
cp "$TEACHER1_CKPT" checkpoints/resnet152_ft.pth
cp "$TEACHER2_CKPT" checkpoints/efficientnet_b2_ft.pth

# Run the minimal script
python main.py --cfg configs/minimal.yaml
