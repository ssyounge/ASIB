#!/bin/bash
cd "$(dirname "$0")"
# run_finetune.sh ── 교사 2종을 ‘한 번에’ 돌리고 싶으면 SLURM 배열 유지,
#                ‘하나씩’ 돌리고 싶으면 아래처럼 파라미터화.

#SBATCH --job-name=ft_teacher
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=logs/ft_%j.log          # 배열 안 쓰면 %a 필요 X
#SBATCH --error=logs/ft_%j.err

# ──────────────────────────────────────────────────────────
# ① CONFIG 매핑 (배열 or 인수)
CONFIGS=(resnet152_cifar32 efficientnet_b2_cifar32)

#   ● 방법 B: sbatch 시 인수로 넘기기 ── 예:  sbatch run_finetune.sh 1
IDX=${1:-0}                     # 인수가 없으면 기본 0
CFG_NAME=${CONFIGS[$IDX]}       # 범위 체크는 아래에서

# ② 범위 확인 (예외 처리)
if [ -z "$CFG_NAME" ]; then
  echo "❌  잘못된 인덱스 $IDX  (0~$((${#CONFIGS[@]}-1)) 중 하나여야 함)" >&2
  exit 1
fi

# ③ 환경
source ~/.bashrc
conda activate tlqkf
export PYTHONPATH="$(pwd):$PYTHONPATH"

# ④ 실행
echo "▶ Fine-tuning $CFG_NAME …"
python scripts/fine_tuning.py \
  --config-path configs/finetune \
  --config-name "$CFG_NAME"
