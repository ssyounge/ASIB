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
# Merge hparams and partial_freeze configs into a single YAML log
python scripts/generate_config.py \
  --base configs/partial_freeze.yaml \
  --hparams configs/hparams.yaml \
  --out "logs/asmb_${JOB_ID}_hparams.yaml"

source ~/.bashrc
conda activate facil_env

bash scripts/run_experiments.sh --mode loop
