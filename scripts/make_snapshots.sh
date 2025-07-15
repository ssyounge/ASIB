#!/bin/bash -l
# scripts/make_snapshots.sh
#SBATCH --job-name=snap_res152
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=${SLURM_SUBMIT_DIR:-$PWD}

set -euo pipefail

# (1) 프로젝트 루트
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    ROOT_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd -P)"
else
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
fi
cd "$ROOT_DIR" || { echo "[ERROR] cd $ROOT_DIR failed"; exit 1; }

# ───────── Conda 환경 ─────────
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tlqkf              # ← 당신이 만든 env 이름

# ───────── 실험 기본 경로 ─────────
#  ※ 파이썬 코드의 to_writable() 가 ${ASMB_KD_ROOT} 를 expand 합니다.
export ASMB_KD_ROOT=/home/suyoung425/ASMB_KD

# ───────── 스냅샷 파라미터 ─────────
CKPT_DIR="${ASMB_KD_ROOT}/checkpoints/snapshots"
EPOCHS=60                               # 총 epoch
INTERVAL=20                             # 몇 epoch마다 저장?
MODEL=resnet152                         # 교사 backbone

# ───────── 교사 스냅샷 트레이닝 ─────────
mkdir -p "$CKPT_DIR"
"$CONDA_PREFIX/bin/python" "$ROOT_DIR/scripts/train_teacher.py" \
    --teacher "$MODEL" \
    --epochs "$EPOCHS" \
    --lr 1e-3 \
    --snapshot_interval "$INTERVAL" \
    --ckpt "$CKPT_DIR/${MODEL}_ft.pth"
