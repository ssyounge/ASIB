#!/usr/bin/env bash
#SBATCH --job-name=ft_all_teachers
#SBATCH --partition=suma_rtx4090
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/logs/ft_all_%j.log
#SBATCH --error=experiments/logs/ft_all_%j.err
# ---------------------------------------------------------
# Fine-tune ALL teacher checkpoints automatically
# 모든 teacher 모델들을 순차적으로 finetune
# ---------------------------------------------------------
set -euo pipefail

# 1) 리포 최상위로 이동
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# 2) PYTHONPATH 추가 (내부 모듈 import 용)
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PATH="/home/suyoung425/.local/bin:$PATH"

# 3) 모든 teacher finetune 설정들
TEACHERS=(
    #"convnext_s_cifar100"     # ConvNeXt-Small (CIFAR-100)
    #"convnext_s_imagenet32"   # ConvNeXt-Small (ImageNet-32)
    #"convnext_l_cifar100"     # ConvNeXt-Large (CIFAR-100)
    #"convnext_l_imagenet32"   # ConvNeXt-Large (ImageNet-32)
    #"efficientnet_l2_cifar100" # EfficientNet-L2 (CIFAR-100)
    "efficientnet_l2_imagenet32" # EfficientNet-L2 (ImageNet-32)
    #"resnet152_cifar100"       # ResNet152 (CIFAR-100)
    #"resnet152_imagenet32"     # ResNet152 (ImageNet-32)
)

# 4) 각 teacher 순차적으로 finetune
for teacher in "${TEACHERS[@]}"; do
    echo "🚀 Starting finetune for: $teacher"
    echo "=================================================="
    
    # finetune 실행
    python scripts/training/fine_tuning.py \
        --config-name "finetune/$teacher" \
        "$@"
    
    echo "✅ Finished finetune for: $teacher"
    echo "=================================================="
    echo ""
done

echo "🎉 All teacher finetuning completed!"
echo "📁 Checkpoints saved in: experiments/checkpoints/"
echo "📊 Results saved in: experiments/outputs/finetune/" 