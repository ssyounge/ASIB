#!/bin/bash
# scripts/run_launcher.sh
#SBATCH --job-name=launcher_job
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x_%j.out        # SLURM 로그
#SBATCH --chdir=/home/suyoung425/ASMB_KD        # ★ 프로젝트 루트 고정
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#-------------------------------------------------------------------
# 기본 설정
#-------------------------------------------------------------------
set -euo pipefail

# (1) 프로젝트 루트:  --chdir 덕분에 이미 루트
ROOT_DIR=$(pwd)

# 안전: 그래도 환경변수로 덮어쓰기 허용
if [ -n "${PROJECT_ROOT:-}" ]; then
    ROOT_DIR="${PROJECT_ROOT}"
    cd "$ROOT_DIR"
fi

# (2) 출력 디렉터리
JOB_ID=${SLURM_JOB_ID:-local}
OUT_ROOT="$ROOT_DIR/outputs"
OUT_DIR="$OUT_ROOT/results/ibkd_${JOB_ID}"
mkdir -p "$OUT_DIR" "$OUT_ROOT/slurm"

# 0) 디버그용 정보
echo "[DEBUG] PWD=$(pwd)"
echo "[DEBUG] HOST=$(hostname)"

# 1) Conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate facil_env
python - <<'PY'
import torch, sys; print("[DEBUG] torch", torch.__version__, "| CUDA =", torch.cuda.is_available())
PY

# 1) 체크포인트 경로 (루트 기준 절대경로로)
CKPT_DIR="$ROOT_DIR/checkpoints"
T1_CKPT="$CKPT_DIR/resnet152_ft.pth"
T2_CKPT="$CKPT_DIR/efficientnet_b2_ft.pth"

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

srun --chdir="$ROOT_DIR" python scripts/launcher.py "$@"

# ➜ 주의: 다른 인자(실험 id, 추가 override 등)는
#     ./run_ibkd.sh --batch_size 256 처럼 이어서 넘기면 됩니다.
trap 'pkill -P $$ || true' EXIT   # 자식 프로세스(예: TensorBoard) 강제 종료
