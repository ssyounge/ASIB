#!/usr/bin/env bash
#SBATCH --job-name=asib_ablation_study
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=experiments/ablation/logs/ablation_%j.log
#SBATCH --error=experiments/ablation/logs/ablation_%j.err

set -euo pipefail
trap 'echo "❌ Job failed at $(date)"; exit 1' ERR

# Python 환경 설정
echo "🔧 Setting up Python environment..."
export PATH="$HOME/anaconda3/envs/tlqkf/bin:$PATH"
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
echo "✅ Python environment setup completed"
echo ""

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# Ensure SLURM log directory exists (best to create before sbatch submission)
mkdir -p "$ROOT/experiments/ablation/logs" || true

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "🔍 Checking GPU allocation..."
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0
  echo "✅ CUDA_VISIBLE_DEVICES set to: 0"
else
  echo "⚠️  SLURM_GPUS_ON_NODE not set, using default GPU 0"
  export CUDA_VISIBLE_DEVICES=0
fi

# CUDA 컨텍스트 초기화 (segmentation fault 방지)
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}

# PyTorch CUDA 라이브러리 경로 가드 (노드별 경로 차이 대응)
TORCH_LIB_DIR="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib"
if [ -d "$TORCH_LIB_DIR" ]; then
  export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$LD_LIBRARY_PATH"
  export CUDA_HOME="$TORCH_LIB_DIR"
fi

# PyTorch CUDA 설정 (아키텍처 리스트 필요 시)
export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_PATH="${CUDA_HOME:-${CUDA_PATH:-}}"
export CUDA_ROOT="${CUDA_HOME:-${CUDA_ROOT:-}}"

# GPU 정보 출력
echo "🔍 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits



# 실행할 실험 선택 (기본: ablation_baseline)
CFG_NAME="${ABLATION_CFG:-ablation_cccp}"

# CFG 유효성 체크(친절한 에러)
CFG_PATH="$ROOT/configs/experiment/${CFG_NAME}.yaml"
if [ ! -f "$CFG_PATH" ]; then
  echo "❌ Invalid ABLATION_CFG='${CFG_NAME}'."
  echo "   Valid options: ablation_baseline, ablation_ib, ablation_cccp, ablation_full, ablation_tadapt"
  exit 1
fi

echo "🚀 Starting ASIB ablation experiment: ${CFG_NAME}"
echo "Time: $(date)"

python -u main.py -cn="experiment/${CFG_NAME}"

echo "✅ Finished ASIB ablation experiment"
echo "Time: $(date)"