#!/bin/bash
# scripts/run_ibkd.sh
#SBATCH --job-name=ibkd_cifar
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=outputs/ibkd_%j.log            # 절대경로나 chdir 둘 중 하나만 택
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail

# 사용 예)  PROJECT_ROOT=/data/ASMB_KD  ./run_ibkd.sh

# ① 현재 위치를 기본값으로, 필요하면 환경변수로 덮어쓰기
ROOT_DIR="${PROJECT_ROOT:-$(pwd)}"

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
            --config configs/base.yaml \
            --teacher_type "$MODEL" \
            --finetune_ckpt_path "$CKPT" \
            ${FINETUNE_EPOCHS:+--finetune_epochs "$FINETUNE_EPOCHS"} \
            ${FINETUNE_LR:+--finetune_lr "$FINETUNE_LR"} \
            --device cuda
  fi
}
ft_teacher resnet152       "$T1_CKPT"
ft_teacher efficientnet_b2 "$T2_CKPT"

# 3) IB-KD 학습
shift 0   # 인수 필요 없음; 있으면 그대로 Python 쪽으로

# ➜ base.yaml + control.yaml 만 넘기면
#    main.py 가 method / train_mode 값을 읽어
#    자동으로 configs/method/***.yaml, configs/scenario/***.yaml 을 merge 합니다.

srun --chdir="$ROOT_DIR" \
     python main.py \
     --cfg "configs/base.yaml,configs/control.yaml" \
     "$@"

# ➜ 주의: 다른 인자(실험 id, 추가 override 등)는
#     ./run_ibkd.sh --batch_size 256 처럼 이어서 넘기면 됩니다.
