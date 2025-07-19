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
conda activate tlqkf

# Launch experiments using the unified script

bash scripts/run_experiments.sh --mode loop --output_dir "$OUTPUT_DIR"

# 로그 파일 tail 을 바로 확인하려면 (옵션)
# tail -f "$OUTPUT_DIR/train.log"
