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
echo "Node: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-'Not set'}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME:-'Not set'}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-'Not set'}"
echo ""

# SLURM 환경변수 확인
echo "🔍 SLURM Environment Variables:"
echo "SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE:-'Not set'}"
echo "SLURM_GPUS_PER_NODE: ${SLURM_GPUS_PER_NODE:-'Not set'}"
echo "SLURM_GPUS_PER_TASK: ${SLURM_GPUS_PER_TASK:-'Not set'}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-'Not set'}"
echo ""

# GPU 할당 확인 (수동 설정 없음: Slurm의 자동 매핑 사용)
echo "🔍 Checking GPU allocation (no manual override)..."
echo "SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE:-'Not set'}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-'Not set'}"
echo "SLURM_STEP_GPUS: ${SLURM_STEP_GPUS:-'Not set'}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-'Not set'}"
echo ""

# CUDA 컨텍스트 초기화 (segmentation fault 방지)
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# PyTorch CUDA 12.4 라이브러리 사용 (경로가 존재할 때만 추가)
PYTORCH_CUDA_LIB_DIR="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib"
if [ -d "$PYTORCH_CUDA_LIB_DIR" ]; then
    unset LD_LIBRARY_PATH || true
    export CUDA_HOME="$PYTORCH_CUDA_LIB_DIR"
    export CUDA_PATH="$PYTORCH_CUDA_LIB_DIR"
    export CUDA_ROOT="$PYTORCH_CUDA_LIB_DIR"
fi
echo ""

# Python/환경 확인
echo "Python: $(python -V)"
echo "Python path: $(which python)"
echo ""

# GPU 정보 출력
echo "🔍 GPU Information:"
nvidia-smi -L
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits
echo ""

# Python으로 GPU 확인 (상세 진단)
echo "🔍 Python GPU Check:"
python -c "
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
    print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Not available\"}')
    
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
