#!/usr/bin/env bash
#SBATCH --job-name=asib_class_overlap
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=experiments/logs/class_overlap_%j.log
#SBATCH --error=experiments/logs/class_overlap_%j.err
# ---------------------------------------------------------
# ASIB Class Overlap Analysis 실험
# 100% Class Overlap 상황에서의 ASIB 성능 분석
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

# 2) PYTHONPATH 추가
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

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



# 4) ASIB Class Overlap Analysis 실험들
EXPERIMENTS=(
    "overlap_100"           # 100% Class Overlap Analysis
)

# 5) 각 실험 순차적으로 실행
for exp in "${EXPERIMENTS[@]}"; do
    echo "🚀 Starting ASIB Class Overlap Analysis experiment: $exp"
    echo "=================================================="
    echo "Time: $(date)"
    echo "Experiment: $exp"
    echo "ASIB Class Overlap Analysis"
    echo "=================================================="
    
    # 실험 실행
    python main.py \
        --config-name "experiment/$exp" \
        "$@"
    
    echo "✅ Finished ASIB Class Overlap Analysis experiment: $exp"
    echo "=================================================="
    echo "Time: $(date)"
    echo ""
done

echo "🎉 ASIB Class Overlap Analysis completed!"
echo "📁 Results saved in: experiments/overlap/results/"
echo "📊 All ASIB experiments completed! Ready for analysis" 