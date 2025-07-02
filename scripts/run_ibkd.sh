#!/usr/bin/env bash
#SBATCH --job-name=ibkd_cifar
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/suyoung425/ASMB_KD        # repo root
#SBATCH --output=outputs/ibkd_%j.log            # 절대경로나 chdir 둘 중 하나만 택
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail

# 0) 디버그용 정보
echo "[DEBUG] PWD=$(pwd)"
echo "[DEBUG] HOST=$(hostname)"

# 1) Conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate facil_env
python - <<'PY'
import torch, sys; print("[DEBUG] torch", torch.__version__, "| CUDA =", torch.cuda.is_available())
PY

# 2) 출력 폴더
JOB_ID=${SLURM_JOB_ID:-local}
OUT_DIR="$HOME/exp_outputs/ibkd_${JOB_ID}"
mkdir -p "${OUT_DIR}"

# 3) 실행
python main.py \
  --cfg configs/minimal.yaml \
  --results_dir "${OUT_DIR}" \
  --teacher1_ckpt "${TEACHER1_CKPT:-ckpts/resnet152_ft.pth}" \
  --teacher2_ckpt "${TEACHER2_CKPT:-ckpts/efficientnet_b2_ft.pth}" \
  "$@"
