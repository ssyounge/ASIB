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

# 1) 경로 설정
T1_CKPT="checkpoints/resnet152_ft.pth"
T2_CKPT="checkpoints/efficientnet_b2_ft.pth"
JOB_ID=${SLURM_JOB_ID:-local}
OUT_DIR="$HOME/exp_outputs/ibkd_${JOB_ID}"
mkdir -p "${OUT_DIR}"

# 2) ──────────────────────────────────────────────
#    Teacher fine-tune (있으면 skip)
ft_teacher () {
  local MODEL=$1
  local CKPT=$2
  if [ -f "$CKPT" ]; then
      echo "[INFO] $MODEL ckpt exists → skip fine-tune"
  else
      echo "[INFO] fine-tuning $MODEL → $CKPT"
      python scripts/fine_tuning.py \
          --config configs/minimal.yaml \
          --teacher_type "$MODEL" \
          --finetune_ckpt_path "$CKPT" \
          --finetune_epochs "${FINETUNE_EPOCHS:-3}" \
          --finetune_lr     "${FINETUNE_LR:-1e-4}" \
          --device cuda
  fi
}
ft_teacher resnet152       "$T1_CKPT"
ft_teacher efficientnet_b2 "$T2_CKPT"

# 3) IB-KD 학습
python main.py \
  --cfg configs/minimal.yaml \
  --results_dir "${OUT_DIR}" \
  --teacher1_ckpt "$T1_CKPT" \
  --teacher2_ckpt "$T2_CKPT"
