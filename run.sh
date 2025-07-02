#!/bin/bash
# run.sh
#SBATCH --job-name=run_ibkd_experiment
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH -o outputs/ibkd_%j.log

set -e

source ~/.bashrc
conda activate facil_env

# Launch IBKD experiment
bash scripts/run_ibkd.sh
