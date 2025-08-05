#!/usr/bin/env bash
#SBATCH --job-name=ft_single_teacher
#SBATCH --partition=suma_a6000
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=experiments/logs/ft_%j.log
#SBATCH --error=experiments/logs/ft_%j.err
# ---------------------------------------------------------
# Fine-tune single teacher checkpoint
# 개별 teacher 모델 하나만 finetune
# ---------------------------------------------------------
set -euo pipefail

# 1) 리포 최상위로 이동
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# 2) PYTHONPATH 추가 (내부 모듈 import 용)
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PATH="/home/suyoung425/.local/bin:$PATH"

# 3) 인자: <yaml basename> (default=convnext_s_cifar100)
CFG_NAME="${1:-convnext_s_cifar100}"
# convnext_s_cifar100, convnext_s_imagenet32, convnext_l_cifar100, convnext_l_imagenet32, efficientnet_l2_cifar100, efficientnet_l2_imagenet32, resnet152_cifar100, resnet152_imagenet32
shift || true         # 나머지 인자 → Hydra override

# 4) 실행
python scripts/training/fine_tuning.py \
    --config-name "finetune/$CFG_NAME" \
    "$@"

echo "[run_finetune_single.sh] ✅ finished – $CFG_NAME"
