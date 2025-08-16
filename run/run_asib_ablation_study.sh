#!/usr/bin/env bash
#SBATCH --job-name=asib_ablation_study
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=experiments/ablation/logs/ablation_%j.log
#SBATCH --error=experiments/ablation/logs/ablation_%j.err

set -euo pipefail
trap 'echo "❌ Job failed at $(date)"; exit 1' ERR

echo "🔧 Setting up Python environment..."
# conda non-interactive 활성화 (set -u로 인한 activate hook 에러 방지)
set +u
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate tlqkf || {
  export PATH="$HOME/anaconda3/envs/tlqkf/bin:$PATH"
}
set -u
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
echo "✅ Python environment setup completed"
echo ""

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

mkdir -p "$ROOT/experiments/ablation/logs" || true
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "🔍 Checking GPU allocation..."
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo "✅ CUDA_VISIBLE_DEVICES set to: ${CUDA_VISIBLE_DEVICES:-<slurm_default>}"

# CUDA libs guard (노드별 경로 차이 대응)
TORCH_LIB_DIR="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib"
if [ -d "$TORCH_LIB_DIR" ]; then
  export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$LD_LIBRARY_PATH"
  export CUDA_HOME="${CUDA_HOME:-$TORCH_LIB_DIR}"
fi

echo "🔍 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || true

# 실행 설정
# ABLATION_CFG: 단일 실험 키 또는 'ladder'
#   유효 키: L0_baseline, L1_ib, L2_cccp, L3_ib_cccp_tadapt, L4_full, side_cccp_ppf (또는 구형 ablation_* 동의어)
# SEEDS: 공백 구분 다중 seed (예: "42 87 1337") 없으면 42 한 번
# EXTRA: 추가 Hydra 오버라이드 (예: 'experiment.schedule.type=cosine')
CFG_NAME="${ABLATION_CFG:-L0_baseline}"
SEEDS_STR="${SEEDS:-42}"
EXTRA_OVR="${EXTRA:-}"

CFG_DIR="$ROOT/configs/experiment"

# Ladder 모드 구성 (정규화된 키)
LADDER_LIST=("L0_baseline" "L1_ib" "L2_cccp" "L3_ib_cccp_tadapt" "L4_full")

# 구형 키 → 정규화된 키 매핑
map_cfg() {
  case "$1" in
    ablation_baseline) echo "L0_baseline";;
    ablation_ib) echo "L1_ib";;
    ablation_cccp) echo "L2_cccp";;
    ablation_ib_cccp_tadapt) echo "L3_ib_cccp_tadapt";;
    ablation_full) echo "L4_full";;
    ablation_cccp_ppf) echo "side_cccp_ppf";;
    *) echo "$1";;
  esac
}

CANON_NAME="$(map_cfg "$CFG_NAME")"

# 유효성 검사
if [ "$CANON_NAME" != "ladder" ]; then
  CFG_PATH="${CFG_DIR}/${CANON_NAME}.yaml"
  if [ ! -f "$CFG_PATH" ]; then
    echo "❌ Invalid ABLATION_CFG='${CFG_NAME}' (canon='${CANON_NAME}')."
    echo "   Valid options:"
    echo "   - L0_baseline (ablation_baseline)"
    echo "   - L1_ib (ablation_ib)"
    echo "   - L2_cccp (ablation_cccp)"
    echo "   - L3_ib_cccp_tadapt (ablation_ib_cccp_tadapt)"
    echo "   - L4_full (ablation_full)"
    echo "   - side_cccp_ppf (ablation_cccp_ppf)"
    echo "   - ladder (runs full ladder set)"
    exit 1
  fi
fi

echo "🔗 Repo: $(git rev-parse --short HEAD) | Branch: $(git rev-parse --abbrev-ref HEAD)"

run_one() {
  local cfg="$1"
  local seed="$2"
  echo ""
  echo "🚀 Starting ASIB ablation experiment: ${cfg} | seed=${seed}"
  echo "Time: $(date)"
  # Hydra 호출(권장): -cn으로 experiment 그룹 선택, 오버라이드는 루트 키로 전달
  echo "OVERRIDES: ${EXTRA_OVR}"
  # root struct에는 seed가 없으므로 '+'로 추가하여 Hydra strict 오류를 회피
  python -u main.py -cn="experiment/${cfg}" +seed="${seed}" ${EXTRA_OVR}
  echo "✅ Finished: ${cfg} | seed=${seed} | Time: $(date)"
}

if [ "$CANON_NAME" = "ladder" ]; then
  echo "📚 Running Ladder sequence: ${LADDER_LIST[*]}"
  for s in ${SEEDS_STR}; do
    for cfg in "${LADDER_LIST[@]}"; do
      if [ ! -f "${CFG_DIR}/${cfg}.yaml" ]; then
        echo "⚠️  Skipping missing config: ${cfg}"
        continue
      fi
      run_one "${cfg}" "${s}"
    done
  done
else
  for s in ${SEEDS_STR}; do
    run_one "${CANON_NAME}" "${s}"
  done
fi

echo ""
echo "✅ All runs completed at $(date)"