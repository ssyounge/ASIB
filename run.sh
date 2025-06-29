#!/bin/bash
# run.sh
#SBATCH --job-name=run_asmb_experiment
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=outputs/asmb_%j/run.log
#SBATCH --error=outputs/asmb_%j/run.log

# Unique output directory per SLURM job
JOB_ID=${SLURM_JOB_ID:-manual}
OUTPUT_DIR="outputs/asmb_${JOB_ID}"
mkdir -p "$OUTPUT_DIR"
cp configs/hparams.yaml "${OUTPUT_DIR}/hparams.yaml"
BASE_CFG_PATH=${BASE_CONFIG:-configs/default.yaml}
cp "$BASE_CFG_PATH" "${OUTPUT_DIR}/base.yaml"
# Save a fully merged YAML with all hyperparameters
python scripts/generate_config.py \
  --base "$BASE_CFG_PATH" \
  --hparams configs/hparams.yaml \
  --out "${OUTPUT_DIR}/full.yaml"

source ~/.bashrc
conda activate facil_env

bash scripts/run_experiments.sh --mode loop --output_dir "$OUTPUT_DIR"
