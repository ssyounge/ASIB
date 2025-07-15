#!/bin/bash -l
# scripts/run_vib_overlap.sh
# Submit with: sbatch --array=0-5 scripts/run_vib_overlap.sh
#SBATCH --job-name=vib_overlap
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-5
#SBATCH --chdir=${SLURM_SUBMIT_DIR:-$PWD}

set -euo pipefail

# Determine repository root
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    ROOT_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd -P)"
else
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
fi
cd "$ROOT_DIR" || { echo "[ERROR] cd $ROOT_DIR failed"; exit 1; }

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tlqkf
export ASMB_KD_ROOT=/home/suyoung425/ASMB_KD

# Output and checkpoint directories
CKPT_DIR="$ASMB_KD_ROOT/checkpoints"
OUT_ROOT="$ASMB_KD_ROOT/outputs"
JOB_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}
OUT_DIR="$OUT_ROOT/vib_overlap_${JOB_ID}"
mkdir -p "$CKPT_DIR" "$OUT_DIR" "$OUT_ROOT/slurm"

# Rho values for the job array
RHO_VALUES=(0.0 0.2 0.4 0.6 0.8 1.0)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
rho=${RHO_VALUES[$TASK_ID]:-0.0}

T1_CKPT="$CKPT_DIR/T1_rho${rho}.pth"
T2_CKPT="$CKPT_DIR/T2_rho${rho}.pth"

srun --chdir="$ROOT_DIR" "$CONDA_PREFIX/bin/python" "$ROOT_DIR/main.py" \
    --cfg configs/experiments/exp_adaptgate.yaml \
    --teacher1_ckpt "$T1_CKPT" \
    --teacher2_ckpt "$T2_CKPT" \
    --results_dir "$OUT_DIR/rho${rho}"

trap 'pkill -P $$ || true' EXIT
