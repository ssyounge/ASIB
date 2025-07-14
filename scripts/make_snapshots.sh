#!/bin/bash
#SBATCH --job-name=snap_res152
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=${SLURM_SUBMIT_DIR:-$PWD}

set -euo pipefail

# ───────── Conda 환경 ─────────
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tlqkf              # ← 당신이 만든 env 이름

# ───────── 스냅샷 파라미터 ─────────
CKPT_DIR=checkpoints/snapshots          # 저장 폴더
EPOCHS=60                               # 총 epoch
INTERVAL=20                             # 몇 epoch마다 저장?
MODEL=resnet152                         # 교사 backbone

mkdir -p "$CKPT_DIR"

python scripts/train_teacher.py \
       --teacher_type "$MODEL" \
       --epochs "$EPOCHS" \
       --snapshot_interval "$INTERVAL" \
       --ckpt_dir "$CKPT_DIR" \
       --finetune_lr 1e-3            # 필요 시 수정
