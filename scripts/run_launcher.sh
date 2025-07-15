#!/bin/bash -l
# scripts/run_launcher.sh
#SBATCH --job-name=launcher_job
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x_%j.out        # SLURM 로그
# 실행 디렉터리는 제출 위치 기준으로 자동 결정되도록 변경
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#-------------------------------------------------------------------
# 기본 설정
#-------------------------------------------------------------------
set -euo pipefail

# (1) 프로젝트 루트 결정 ---------------------------------------------
#    ① SLURM 환경이면  $SLURM_SUBMIT_DIR   (제출 위치) 사용
#    ② 아니면 스크립트 상대 경로(BASH_SOURCE)로 계산
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    ROOT_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd -P)"
else
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
fi
cd "$ROOT_DIR" || { echo "[ERROR] cd $ROOT_DIR failed"; exit 1; }

# 안전: 그래도 환경변수로 덮어쓰기 허용
if [ -n "${PROJECT_ROOT:-}" ]; then
    ROOT_DIR="${PROJECT_ROOT}"
    cd "$ROOT_DIR"
fi

# 저장 경로 환경변수는 외부에서 설정 가능하며,
# 기본값은 "${HOME}/.asmb_kd" 하위 디렉터리입니다.
export ASMB_KD_ROOT="${ASMB_KD_ROOT:-$HOME/.asmb_kd}"

# (2) 출력·체크포인트 디렉터리를 "집" 밑으로 옮긴다
JOB_ID=${SLURM_JOB_ID:-local}
CKPT_DIR="${ASMB_KD_ROOT:-$HOME/.asmb_kd}/checkpoints"      # ← 쓰기 가능한 위치
OUT_ROOT="${ASMB_KD_ROOT:-$HOME/.asmb_kd}/outputs"
OUT_DIR="$OUT_ROOT/ibkd_${JOB_ID}"

# 실제로 만들기
mkdir -p "$CKPT_DIR" "$OUT_DIR" "$OUT_ROOT/slurm"

# 0) 디버그용 정보
echo "[DEBUG] PWD=$(pwd)"
echo "[DEBUG] HOST=$(hostname)"

## 1) Conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tlqkf

# (선택) W&B API Key – 있을 때만 ↓ 라인 유지, 없으면 그냥 지워두세요
export WANDB_API_KEY="ca52ce9b353498922ae0cd78cbb5ae0673494e6b"

# 디버그 – 지금 파이썬이 conda env 것인지 확인
"$CONDA_PREFIX/bin/python" - <<'PY'
import torch, sys; print("[DEBUG] torch", torch.__version__, "| CUDA =", torch.cuda.is_available())
PY

# 1) 체크포인트 경로는 위에서 지정된 디렉터리 사용
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
        "$CONDA_PREFIX/bin/python" "$ROOT_DIR/scripts/fine_tuning.py" \
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

srun --chdir="$ROOT_DIR" "$CONDA_PREFIX/bin/python" "$ROOT_DIR/scripts/launcher.py" "$@"

# ➜ 주의: 다른 인자(실험 id, 추가 override 등)는
#     ./run_launcher.sh --batch_size 256 처럼 이어서 넘기면 됩니다.
trap 'pkill -P $$ || true' EXIT   # 자식 프로세스(예: TensorBoard) 강제 종료
