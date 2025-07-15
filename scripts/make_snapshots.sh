#!/bin/bash -l
# scripts/make_snapshots.sh
#SBATCH --job-name=snap_res152
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=${SLURM_SUBMIT_DIR:-$PWD}

set -euo pipefail

# (1) 프로젝트 루트: 스크립트 위치 기준으로 자동 결정
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# ───────── Conda 환경 ─────────
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tlqkf              # ← 당신이 만든 env 이름

# ───────── 스냅샷 파라미터 ─────────
CKPT_DIR="${ASMB_KD_ROOT:-$HOME/.asmb_kd}/checkpoints/snapshots"          # 저장 폴더
EPOCHS=60                               # 총 epoch
INTERVAL=20                             # 몇 epoch마다 저장?
MODEL=resnet152                         # 교사 backbone

# Conda env의 python 바이너리를 쓰고, 새 파일명 fine_tuning.py 사용
mkdir -p "$CKPT_DIR"
"$CONDA_PREFIX/bin/python"  scripts/fine_tuning.py \
       --teacher_type "$MODEL" \
       --epochs "$EPOCHS" \
       --snapshot_interval "$INTERVAL" \
       --ckpt_dir "$CKPT_DIR" \
       --finetune_lr 1e-3            # 필요 시 수정
