#!/usr/bin/env bash
#SBATCH --job-name=asib_ablation_study
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --array=0-4%5            # sbatch 시 --array=0-(N-1)%K 로 덮어써서 사용
#SBATCH --output=experiments/ablation/logs/ablation_%A_%a.log
#SBATCH --error=experiments/ablation/logs/ablation_%A_%a.err

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

# Array → 샤딩 자동 매핑
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  SHARD_IDX="${SLURM_ARRAY_TASK_ID}"
fi
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
  SHARD_N="${SLURM_ARRAY_TASK_COUNT}"
elif [[ -n "${SLURM_ARRAY_TASK_MAX:-}" && -n "${SLURM_ARRAY_TASK_MIN:-}" ]]; then
  SHARD_N="$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))"
fi
SHARD_N="${SHARD_N:-1}"
SHARD_IDX="${SHARD_IDX:-0}"
echo "🔀 Sharding: SHARD_IDX=${SHARD_IDX} / SHARD_N=${SHARD_N}"
run_idx=0

# CUDA libs guard (노드별 경로 차이 대응)
TORCH_LIB_DIR="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib"
if [ -d "$TORCH_LIB_DIR" ]; then
  unset LD_LIBRARY_PATH || true
  export CUDA_HOME="${CUDA_HOME:-$TORCH_LIB_DIR}"
fi

echo "🔍 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || true

# 실행 설정
# ABLATION_CFG: 단일 실험 키 또는 'ladder'
#   유효 키: L0_baseline, L1_ib, L2_cccp, L3_asib_cccp, L4_full (또는 구형 ablation_* 동의어)
# SEEDS: 공백 구분 다중 seed (예: "42 87 1337") 없으면 42 한 번
# EXTRA: 추가 Hydra 오버라이드 (예: 'experiment.schedule.type=cosine')
CFG_NAME="${ABLATION_CFG:-ladder}"
DRY_RUN="${DRY_RUN:-0}"
SEEDS_STR="${SEEDS:-42}"
EXTRA_OVR="${EXTRA:-}"

CFG_DIR="$ROOT/configs/experiment"

# Ladder 모드 구성 (정규화된 키)
LADDER_LIST=("L0_baseline" "L1_ib" "L2_cccp" "L3_asib_cccp" "L4_full")

# 구형 키 → 정규화된 키 매핑
map_cfg() {
  case "$1" in
    ablation_baseline) echo "L0_baseline";;
    ablation_ib) echo "L1_ib";;
    ablation_cccp) echo "L2_cccp";;
    ablation_ib_cccp_tadapt) echo "L3_asib_cccp";;
    ablation_full) echo "L4_full";;
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
    echo "   - L3_asib_cccp (ablation_ib_cccp_tadapt)"
    echo "   - L4_full (ablation_full)"
    echo "   - ladder (runs full ladder set)"
    exit 1
  fi
fi

echo "🔗 Repo: $(git rev-parse --short HEAD) | Branch: $(git rev-parse --abbrev-ref HEAD)"

# DataLoader workers 자동 스케일 (경고 방지):
# - 기본 상한은 4로 제한 (환경변수 PYTORCH_WORKERS_MAX로 조정 가능)
calc_workers() {
  local cpus="${SLURM_CPUS_PER_TASK:-4}"
  local maxw="${PYTORCH_WORKERS_MAX:-4}"
  # 최소 1 보장, 상한은 maxw로 제한
  if (( cpus < 1 )); then cpus=1; fi
  if (( cpus > maxw )); then echo "$maxw"; else echo "$cpus"; fi
}

WORKERS="$(calc_workers)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$WORKERS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$WORKERS}"
echo "🧵 num_workers(auto)=${WORKERS}, OMP=${OMP_NUM_THREADS}, MKL=${MKL_NUM_THREADS}"

run_one() {
  local cfg="$1"
  local seed="$2"
  local workers_override=""
  # 자동 주입: EXTRA/CLI 둘 다에 num_workers 없을 때만
  local has_nw=0
  if [[ " ${EXTRA_OVR} " == *"experiment.dataset.num_workers="* || " ${EXTRA_OVR} " == *"dataset.num_workers="* ]]; then
    has_nw=1
  fi
  for arg in "$@"; do
    if [[ "$arg" == experiment.dataset.num_workers=* || "$arg" == dataset.num_workers=* ]]; then
      has_nw=1
    fi
  done
  if [[ $has_nw -eq 0 ]]; then
    # Strict struct: append under experiment.dataset
    workers_override="+experiment.dataset.num_workers=${WORKERS}"
  fi
  echo ""
  echo "🚀 Starting ASIB ablation experiment: ${cfg} | seed=${seed}"
  echo "Time: $(date)"
  # Hydra 호출(권장): -cn으로 experiment 그룹 선택, 오버라이드는 루트 키로 전달
  echo "OVERRIDES: ${EXTRA_OVR}"
  # root struct에는 seed가 없으므로 '+'로 추가하여 Hydra strict 오류를 회피
  # Hydra-safe CLI만 패스스루 (key=value, +key=value, 또는 하이픈 옵션)
  local passthru_args=()
  for a in "$@"; do
    if [[ "$a" == -* || "$a" == *=* || "$a" == +*=* ]]; then
      passthru_args+=("$a")
    fi
  done
  python -u main.py -cn="experiment/${cfg}" +seed="${seed}" ${EXTRA_OVR} ${workers_override} "${passthru_args[@]}"
  echo "✅ Finished: ${cfg} | seed=${seed} | Time: $(date)"
}

# ───────────────────────────── Job Array 로직 ─────────────────────────────
# 1) 실행할 (cfg,seed) 조합들을 RUNS 배열로 구성
RUNS=()
if [ "$CANON_NAME" = "ladder" ]; then
  echo "📚 Ladder sequence: ${LADDER_LIST[*]}"
  for s in ${SEEDS_STR}; do
    for cfg in "${LADDER_LIST[@]}"; do
      if [ -f "${CFG_DIR}/${cfg}.yaml" ]; then
        RUNS+=("${cfg}|${s}")
      else
        echo "⚠️  Skipping missing config: ${cfg}"
      fi
    done
  done
else
  if [ -f "${CFG_DIR}/${CANON_NAME}.yaml" ]; then
    for s in ${SEEDS_STR}; do
      RUNS+=("${CANON_NAME}|${s}")
    done
  else
    echo "❌ Invalid ABLATION_CFG='${CFG_NAME}' (canon='${CANON_NAME}')."
    exit 1
  fi
fi

N_RUNS=${#RUNS[@]}
echo "🧮 Planned runs: ${N_RUNS}"

# 배열 크기에 맞게 RUNS 자동 확장 (seed 증가로 유니크 보장)
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
  TARGET=${SLURM_ARRAY_TASK_COUNT}
  if (( N_RUNS < TARGET && N_RUNS > 0 )); then
    NEW_RUNS=()
    for (( i=0; i<TARGET; i++ )); do
      base_idx=$(( i % N_RUNS ))
      IFS='|' read -r CFG_I SEED_I <<< "${RUNS[$base_idx]}"
      NEW_SEED=$(( SEED_I + i ))
      NEW_RUNS+=("${CFG_I}|${NEW_SEED}")
    done
    RUNS=("${NEW_RUNS[@]}")
    N_RUNS=${#RUNS[@]}
    echo "🧩 Auto-expanded RUNS to match array: ${N_RUNS}"
  fi
fi

if (( DRY_RUN > 0 )); then
  echo "📋 This shard will run:"
  for i in $(seq 0 $((N_RUNS-1))); do
    if (( SHARD_N > 1 )); then
      if (( (i % SHARD_N) != SHARD_IDX )); then continue; fi
    fi
    echo "  - [$i] ${RUNS[$i]}"
  done
  echo "🔎 Tip: sbatch --array=0-$((SHARD_N-1))%$SHARD_N run/run_asib_ablation_study.sh ABLATION_CFG=${CFG_NAME} SEEDS=\"${SEEDS_STR}\""
  exit 0
fi

# 2) Array index → run index 매핑 (각 태스크가 정확히 1개 런 수행)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  idx="${SLURM_ARRAY_TASK_ID}"
  if (( idx >= N_RUNS )); then
    echo "ℹ️  Array index ${idx} >= N_RUNS ${N_RUNS} → nothing to do."; exit 0
  fi
  IFS='|' read -r CFG_I SEED_I <<< "${RUNS[$idx]}"
  run_one "${CFG_I}" "${SEED_I}"
else
  for rs in "${RUNS[@]}"; do
    IFS='|' read -r CFG_I SEED_I <<< "$rs"
    run_one "${CFG_I}" "${SEED_I}"
  done
fi

echo ""
echo "✅ All runs completed at $(date)"