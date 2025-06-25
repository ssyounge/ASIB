#!/bin/bash
# run.sh
#SBATCH --job-name=run_asmb_experiment
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/asmb_%j.log
#SBATCH --error=logs/asmb_%j.err

JOB_ID=${SLURM_JOB_ID:-manual}
mkdir -p logs
cp configs/hparams.yaml "logs/asmb_${JOB_ID}_hparams.yaml"
BASE_CFG_PATH=${BASE_CONFIG:-configs/default.yaml}
cp "$BASE_CFG_PATH" "logs/asmb_${JOB_ID}_base.yaml"
# Save a fully merged YAML with all hyperparameters
python scripts/generate_config.py \
  --base "$BASE_CFG_PATH" \
  --hparams configs/hparams.yaml \
  --out "logs/asmb_${JOB_ID}_full.yaml"

source ~/.bashrc
conda activate facil_env

bash scripts/run_experiments.sh --mode loop
