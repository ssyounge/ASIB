#!/usr/bin/env bash
#SBATCH --job-name=test_gpu_allocation
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:10:00
#SBATCH --output=experiments/logs/test_gpu_%j.log
#SBATCH --error=experiments/logs/test_gpu_%j.err

set -euo pipefail

# Python 환경 설정
echo "🔧 Setting up Python environment..."
export PATH="$HOME/anaconda3/envs/tlqkf/bin:$PATH"
echo "✅ Python environment setup completed"
echo ""

echo "=== GPU Allocation Test ==="
echo "Time: $(date)"
echo ""

# SLURM 환경변수 확인
echo "🔍 SLURM Environment Variables:"
echo "SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE:-'Not set'}"
echo "SLURM_GPUS_PER_NODE: ${SLURM_GPUS_PER_NODE:-'Not set'}"
echo "SLURM_GPUS_PER_TASK: ${SLURM_GPUS_PER_TASK:-'Not set'}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-'Not set'}"
echo ""

# GPU 할당 확인 및 설정
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
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CUDA 환경변수 (PyTorch 내장 CUDA 12.4 사용)
export CUDA_PATH="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib"
export CUDA_ROOT="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib"
echo ""

# GPU 정보 출력
echo "🔍 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits
echo ""

# Python으로 GPU 확인 (상세 진단)
echo "🔍 Python GPU Check:"
python -c "
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
    print(f'CUDA_HOME: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Not available\"}')
    
    # 환경변수 확인
    import os
    print(f'CUDA_HOME env: {os.environ.get(\"CUDA_HOME\", \"Not set\")}')
    print(f'LD_LIBRARY_PATH: {os.environ.get(\"LD_LIBRARY_PATH\", \"Not set\")[:100]}...')
    print(f'CUDA_VISIBLE_DEVICES: {os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"Not set\")}')
    
    if torch.cuda.is_available():
        print(f'CUDA device count: {torch.cuda.device_count()}')
        print(f'Current device: {torch.cuda.current_device()}')
        print(f'Device name: {torch.cuda.get_device_name(0)}')
        print('✅ CUDA is working!')
    else:
        print('❌ CUDA not available in PyTorch')
        print('Debugging info:')
        print('1. PyTorch CUDA support: Check if PyTorch was compiled with CUDA')
        print('2. CUDA libraries: Check if CUDA libraries are in LD_LIBRARY_PATH')
        print('3. GPU memory: Check if GPU has enough memory')
        print('4. CUDA_VISIBLE_DEVICES: Check if GPU is visible to PyTorch')
except Exception as e:
    print(f'⚠️  Python torch import failed: {e}')
    import traceback
    traceback.print_exc()
"
echo ""

echo "✅ GPU allocation test completed!"
echo "Time: $(date)"
