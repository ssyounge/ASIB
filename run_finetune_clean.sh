#!/usr/bin/env bash
#SBATCH --job-name=ft_teacher
#SBATCH --partition=dell_rtx3090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=logs/ft_%j.log
#SBATCH --error=logs/ft_%j.err
# ---------------------------------------------------------
# Fine-tune teacher checkpoints (clean - no SLURM required)
# 스크립트를 어디서 실행하든 레포 루트에서 시작하도록 고정
# ---------------------------------------------------------
set -euo pipefail

# 1) 리포 최상위로 이동
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# 2) PYTHONPATH 추가 (내부 모듈 import 용)
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# 3) 인자: <yaml basename> (default=resnet152_cifar32)
CFG_NAME="${1:-resnet152_cifar32}"
# efficientnet_l2_cifar32, resnet152_cifar32
shift || true         # 나머지 인자 → Hydra override

# 4) 실행
python scripts/fine_tuning.py \
    --config-path configs \
    --config-name "finetune/$CFG_NAME" \
    "$@"

echo "[run_finetune_clean.sh] ✅ finished – $CFG_NAME"
