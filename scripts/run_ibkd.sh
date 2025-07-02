#!/usr/bin/env bash
# scripts/run_ibkd.sh

#SBATCH --job-name=ibkd_cifar
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=outputs/ibkd_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail

############################ 0) 프로젝트 루트 이동 ############################
# SLURM 작업 위치가 불분명할 때를 대비
cd "$(dirname "$0")/.."          # script 경로의 한 디렉터리 위 (= repo root)

############################ 1) Conda 환경 활성화 ############################
# Conda 초기화 스크립트를 명시적으로 호출해야 non‑interactive shell에서 인식
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "[ERROR] conda not found"; exit 1
fi
conda activate facil_env

# 디버그: 실제 Python / torch 확인
echo "[DEBUG] python: $(which python)"
python - <<'PY'
import sys, importlib.util, os
print("[DEBUG] sys.executable =", sys.executable)
try:
    import torch; print("[DEBUG] torch =", torch.__version__)
except ModuleNotFoundError as e:
    print("[ERROR] torch import failed:", e); sys.exit(1)
PY

############################ 2) 체크포인트 / 출력 디렉터리 ############################
T1_CKPT="${TEACHER1_CKPT:-ckpts/resnet152_ft.pth}"
T2_CKPT="${TEACHER2_CKPT:-ckpts/efficientnet_b2_ft.pth}"

JOB_ID="${SLURM_JOB_ID:-local}"
OUT_DIR="outputs/ibkd_${JOB_ID}"
mkdir -p "$OUT_DIR"

############################ 3) 실험 실행 ############################
python main.py \
  --cfg configs/minimal.yaml \
  --teacher1_ckpt "$T1_CKPT" \
  --teacher2_ckpt "$T2_CKPT" \
  --results_dir "$OUT_DIR" \
  "$@"
