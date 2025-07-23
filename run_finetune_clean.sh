#!/bin/bash
#SBATCH --job-name=ft_teacher
#SBATCH --partition=dell_rtx3090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ft_%j.log
#SBATCH --error=logs/ft_%j.err

cd "$(dirname "$0")"
source ~/.bashrc
conda activate tlqkf
export PYTHONPATH="$(pwd):$PYTHONPATH"

CONFIGS=(resnet152_cifar32 efficientnet_b2_cifar32)
IDX=${1:-0}
CFG_NAME=${CONFIGS[$IDX]}

if [ -z "$CFG_NAME" ]; then
  echo "❌  잘못된 인덱스 $IDX" >&2
  exit 1
fi

echo "▶ Fine-tuning config: $CFG_NAME"
python scripts/fine_tuning.py \
  --config-path configs/finetune \
  --config-name "$CFG_NAME"
