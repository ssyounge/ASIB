#!/bin/bash
#SBATCH --job-name=asmb_exp_clean
#SBATCH --partition=dell_rtx3090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/asmb_%j/run.log
#SBATCH --error=outputs/asmb_%j/run.log

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source ~/.bashrc
conda activate tlqkf
export PYTHONPATH="$(pwd):$PYTHONPATH"

: "${SLURM_JOB_ID:=manual}"
mkdir -p "outputs/asmb_${SLURM_JOB_ID}"

python main.py --config-name experiment/res152_effi_l2
