#!/usr/bin/env bash
#SBATCH --job-name=asib_class_overlap
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --array=0-10%1                      # 제출 시 --array=0-(N-1)%K 로 덮어쓰기
#SBATCH --output=experiments/logs/class_overlap_%A_%a.log
#SBATCH --error=experiments/logs/class_overlap_%A_%a.err
# ---------------------------------------------------------
# ASIB Class Overlap Analysis (Job Array + mod-sharding, no srun)
# ---------------------------------------------------------
set -euo pipefail
trap 'echo "❌ Job failed at $(date)"; exit 1' ERR

# Python env
echo "🔧 Setting up Python environment..."
set +u
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate tlqkf || export PATH="$HOME/anaconda3/envs/tlqkf/bin:$PATH"
set -u
export HYDRA_FULL_ERROR=1

# Repo root
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
mkdir -p "$ROOT/experiments/logs" || true
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Slurm GPU 바인딩 안내 (수동 고정 안 함)
echo "🔍 Slurm will set CUDA_VISIBLE_DEVICES per array task."

# Torch libs guard (노드별 차이 대응)
TORCH_LIB_DIR="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib"
if [ -d "$TORCH_LIB_DIR" ]; then
  unset LD_LIBRARY_PATH || true
  export CUDA_HOME="${CUDA_HOME:-$TORCH_LIB_DIR}"
fi

# num_workers 자동 스케일: SLURM_CPUS_PER_TASK와 상한(PYTORCH_WORKERS_MAX, 기본 4)
calc_workers() {
  local cpus="${SLURM_CPUS_PER_TASK:-4}"
  local maxw="${PYTORCH_WORKERS_MAX:-4}"
  (( cpus < 1 )) && cpus=1
  if (( maxw < cpus )); then echo "$maxw"; else echo "$cpus"; fi
}
WORKERS="$(calc_workers)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$WORKERS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$WORKERS}"
echo "🧵 num_workers(auto)=${WORKERS}, OMP=${OMP_NUM_THREADS}, MKL=${MKL_NUM_THREADS}"

# GPU info (optional)
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || true

# ========================== 설정 ==========================
# 여러 실험/시드를 Job Array로 나눠서 돌릴 수 있도록 RUNS 구성
# EXPS: 공백 구분 실험 이름 리스트 (기본: overlap_100)
# SEEDS: 공백 구분 시드 리스트 (기본: 42)
EXPS_STR="${EXPS:-overlap_100}"
SEEDS_STR="${SEEDS:-42}"
EXTRA_OVR="${EXTRA:-}"
DRY_RUN="${DRY_RUN:-0}"

# Slurm Array → 샤딩 변수
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then SHARD_IDX="${SLURM_ARRAY_TASK_ID}"; fi
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
  SHARD_N="${SLURM_ARRAY_TASK_COUNT}"
elif [[ -n "${SLURM_ARRAY_TASK_MAX:-}" && -n "${SLURM_ARRAY_TASK_MIN:-}" ]]; then
  SHARD_N="$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))"
fi
SHARD_N="${SHARD_N:-1}"
SHARD_IDX="${SHARD_IDX:-0}"
echo "🔀 Sharding: SHARD_IDX=${SHARD_IDX} / SHARD_N=${SHARD_N}"

# RUNS = (exp | seed) 조합
RUNS=()
for exp in ${EXPS_STR}; do
  for s in ${SEEDS_STR}; do
    RUNS+=("${exp}|${s}")
  done
done
N_RUNS=${#RUNS[@]}
echo "🧮 Planned runs: ${N_RUNS}"

# 배열 크기에 맞게 RUNS 자동 확장 (seed 증가로 유니크 보장)
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
  TARGET=${SLURM_ARRAY_TASK_COUNT}
  if (( N_RUNS < TARGET && N_RUNS > 0 )); then
    NEW_RUNS=()
    for (( i=0; i<TARGET; i++ )); do
      base_idx=$(( i % N_RUNS ))
      IFS='|' read -r EXP_B SEED_B <<< "${RUNS[$base_idx]}"
      NEW_SEED=$(( SEED_B + i ))
      NEW_RUNS+=("${EXP_B}|${NEW_SEED}")
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
  echo "🔎 Tip: sbatch --array=0-$((SHARD_N-1))%$SHARD_N run/run_asib_class_overlap.sh EXPS=\"${EXPS_STR}\" SEEDS=\"${SEEDS_STR}\""
  exit 0
fi

# ========================== 실행 ==========================
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  idx="${SLURM_ARRAY_TASK_ID}"
  if (( idx >= N_RUNS )); then
    echo "ℹ️  Array index ${idx} >= N_RUNS ${N_RUNS} → nothing to do."; exit 0
  fi
  IFS='|' read -r EXP_NAME SEED_VAL <<< "${RUNS[$idx]}"
  echo ""
  echo "🚀 Starting: ${EXP_NAME} | seed=${SEED_VAL}"
  echo "Time: $(date)"
  echo "OVERRIDES: ${EXTRA_OVR}"

  HAS_NW_OVR=0
  if [[ " ${EXTRA_OVR} " == *"experiment.dataset.num_workers="* || " ${EXTRA_OVR} " == *"dataset.num_workers="* ]]; then
    HAS_NW_OVR=1
  fi
  for arg in "$@"; do
    if [[ "$arg" == experiment.dataset.num_workers=* || "$arg" == dataset.num_workers=* ]]; then
      HAS_NW_OVR=1
    fi
  done
  NW_OVR=()
  if [[ $HAS_NW_OVR -eq 0 ]]; then
    NW_OVR=(+experiment.dataset.num_workers="${WORKERS}")
  fi

  PASSTHRU_ARGS=()
  for a in "$@"; do
    if [[ "$a" == -* || "$a" == *=* || "$a" == +*=* ]]; then
      PASSTHRU_ARGS+=("$a")
    fi
  done
  python -u main.py -cn="experiment/${EXP_NAME}" +seed="${SEED_VAL}" ${EXTRA_OVR} "${NW_OVR[@]}" "${PASSTHRU_ARGS[@]}"

  echo "✅ Finished: ${EXP_NAME} | seed=${SEED_VAL} | Time: $(date)"
else
  for rs in "${RUNS[@]}"; do
    IFS='|' read -r EXP_NAME SEED_VAL <<< "$rs"
    echo ""
    echo "🚀 Starting: ${EXP_NAME} | seed=${SEED_VAL}"
    echo "Time: $(date)"
    echo "OVERRIDES: ${EXTRA_OVR}"

    HAS_NW_OVR=0
    if [[ " ${EXTRA_OVR} " == *"experiment.dataset.num_workers="* || " ${EXTRA_OVR} " == *"dataset.num_workers="* ]]; then
      HAS_NW_OVR=1
    fi
    for arg in "$@"; do
      if [[ "$arg" == experiment.dataset.num_workers=* || "$arg" == dataset.num_workers=* ]]; then
        HAS_NW_OVR=1
      fi
    done
    NW_OVR=()
    if [[ $HAS_NW_OVR -eq 0 ]]; then
      NW_OVR=(+experiment.dataset.num_workers="${WORKERS}")
    fi

    PASSTHRU_ARGS=()
    for a in "$@"; do
      if [[ "$a" == -* || "$a" == *=* || "$a" == +*=* ]]; then
        PASSTHRU_ARGS+=("$a")
      fi
    done
    python -u main.py -cn="experiment/${EXP_NAME}" +seed="${SEED_VAL}" ${EXTRA_OVR} "${NW_OVR[@]}" "${PASSTHRU_ARGS[@]}"

    echo "✅ Finished: ${EXP_NAME} | seed=${SEED_VAL} | Time: $(date)"
  done
fi

echo ""
echo "🎉 ASIB Class Overlap Analysis completed!"
echo "📁 Results in: experiments/overlap/results/"