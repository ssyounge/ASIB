#!/usr/bin/env bash
#SBATCH --job-name=ft_all_teachers
#SBATCH --partition=base_suma_rtx3090
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

# Python 환경 설정
echo "🔧 Setting up Python environment..."
export PATH="$HOME/anaconda3/envs/tlqkf/bin:$PATH"
echo "✅ Python environment setup completed"
echo ""

# 1) 리포 최상위로 이동
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# 2) PYTHONPATH 추가 (내부 모듈 import 용)
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PATH="$HOME/.local/bin:$PATH"

# 3) GPU 할당 확인 및 설정
echo "🔍 Checking GPU allocation..."
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    # GPU 인덱스를 0부터 시작하도록 조정
    if [ "$SLURM_GPUS_ON_NODE" = "1" ]; then
        export CUDA_VISIBLE_DEVICES=0
        echo "✅ CUDA_VISIBLE_DEVICES set to: 0 (mapped from SLURM_GPUS_ON_NODE=1)"
    else
        export CUDA_VISIBLE_DEVICES=0
        echo "✅ CUDA_VISIBLE_DEVICES set to: 0 (default for any GPU allocation)"
    fi
else
    echo "⚠️  SLURM_GPUS_ON_NODE not set, using default GPU 0"
    export CUDA_VISIBLE_DEVICES=0
fi

# CUDA 컨텍스트 초기화 (segmentation fault 방지)
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# PyTorch CUDA 12.4 라이브러리 사용
export LD_LIBRARY_PATH="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"
export CUDA_HOME="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib"

# PyTorch CUDA 설정
export TORCH_CUDA_ARCH_LIST="8.6"

# CUDA 환경변수 (PyTorch 내장 CUDA 12.4 사용)
export CUDA_PATH="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib"
export CUDA_ROOT="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib"

# GPU 정보 출력
echo "🔍 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits



# 4) 모든 teacher finetune 설정들
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

# 5) 각 teacher 순차적으로 finetune
for teacher in "${TEACHERS[@]}"; do
    echo "🚀 Starting finetune for: $teacher"
    echo "=================================================="
    
    # finetune 실행
    python scripts/training/fine_tuning.py -cn="finetune/$teacher" \
        "$@"
    
    echo "✅ Finished finetune for: $teacher"
    echo "=================================================="
    echo ""
done

echo "🎉 All teacher finetuning completed!"
echo "📁 Checkpoints saved in: experiments/checkpoints/"
echo "📊 Results saved in: experiments/finetune/results/" 