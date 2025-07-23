#!/bin/bash
# run_finetune.sh  ── SLURM array로 교사 2 종 fine-tune
#SBATCH --job-name=ft_teacher
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --array=0-1            # 0: ResNet-152, 1: EfficientNet-B2
#SBATCH --output=logs/ft_%A_%a.log
#SBATCH --error=logs/ft_%A_%a.log

# ① array-id → config 이름 매핑
CONFIGS=(resnet152_cifar32 efficientnet_b2_cifar32)
CFG_NAME=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

# ② 환경
source ~/.bashrc
conda activate tlqkf
export PYTHONPATH="$(pwd):$PYTHONPATH"   # 로컬 모듈 찾도록

# ③ 실행
echo "▶ Fine-tuning $CFG_NAME …"
python scripts/fine_tuning.py \
  --config-path configs/finetune \
  --config-name "$CFG_NAME"
