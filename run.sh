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

# Create a unique output directory per SLURM job
JOB_ID=${SLURM_JOB_ID:-manual}
OUTPUT_DIR="outputs/asmb_${JOB_ID}"
mkdir -p "$OUTPUT_DIR"

python main.py --config-name base \
  +teacher1_type=resnet152 \
  +teacher2_type=efficientnet_b2 \
  +student_type=resnet152_adapter \
  +student_pretrained=true \
  +student_freeze_level=1 \
  num_stages=3 \
  batch_size=128 \
  +results_dir=outputs/debug_run \
  +exp_id=debug_run
