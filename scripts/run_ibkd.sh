#!/usr/bin/env bash
# scripts/run_ibkd.sh  ── Information‑Bottleneck KD 실험 런처
#SBATCH --job-name=ibkd_cifar
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x_%j.log        # ibkd_cifar_<jobid>.log

set -euo pipefail

######################### 0) 프로젝트 루트 이동 #########################
cd "$(dirname "$0")/.."           # repo root

######################### 1) Conda 환경 활성화 #########################
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
else
    eval "$(conda shell.bash hook)"
fi
conda activate facil_env

######################### 2) 디버그: 파이썬/토치 버전 ###################
echo "[DEBUG] python: $(which python)"
python - <<'PY'
import sys, torch, os
print("[DEBUG] sys.executable =", sys.executable)
print("[DEBUG] torch =", torch.__version__, "| CUDA =", torch.cuda.is_available())
PY

######################### 3) 체크포인트·출력 경로 #######################
T1_CKPT="${TEACHER1_CKPT:-ckpts/resnet152_ft.pth}"
T2_CKPT="${TEACHER2_CKPT:-ckpts/efficientnet_b2_ft.pth}"

OUT_ROOT="${OUT_ROOT:-${SLURM_TMPDIR:-${HOME}/ibkd_runs}}"
JOB_ID="${SLURM_JOB_ID:-local}"
OUT_DIR="${OUT_ROOT}/ibkd_${JOB_ID}"
mkdir -p "$OUT_DIR"

######################### 4) 실험 실행 #################################
python main.py \
  --cfg configs/minimal.yaml \
  --teacher1_ckpt "$T1_CKPT" \
  --teacher2_ckpt "$T2_CKPT" \
  --results_dir  "$OUT_DIR" \
  "$@"
