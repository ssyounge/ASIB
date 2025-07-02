# scripts/run_ibkd.sh

#!/usr/bin/env bash
#SBATCH --job-name=ibkd_cifar
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=outputs/ibkd_%j.log

set -e

# Conda environment is optional when running locally
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi
if [ -z "$CONDA_DEFAULT_ENV" ] && command -v conda >/dev/null; then
    conda activate facil_env
fi

# Teacher checkpoints can be overridden via environment variables
T1_CKPT="${TEACHER1_CKPT:-ckpts/resnet152_ft.pth}"
T2_CKPT="${TEACHER2_CKPT:-ckpts/efficientnet_b2_ft.pth}"

# Create a per-job output directory
JOB_ID="${SLURM_JOB_ID:-local}"
mkdir -p "outputs/ibkd_${JOB_ID}"

python main.py --cfg configs/minimal.yaml "$@"

