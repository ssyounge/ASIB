#!/usr/bin/env bash
#SBATCH --job-name=asib_sota_comparison
#SBATCH -D /home/suyoung425/ASIB
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-10%6
#SBATCH --output=experiments/sota/logs/sota_comparison_%A_%a.out
#SBATCH --error=experiments/sota/logs/sota_comparison_%A_%a.err

# Local SOTA comparison runner (separate from run_asib_methods_sanity.sh)
# - Keeps teacher/student fixed sets and sweeps methods
# - Designed for quick local loops; use the Slurm file for array jobs

set -euo pipefail
trap 'echo "❌ Job failed at $(date)"; exit 1' ERR

echo "🔧 Setting up Python environment..."
PYTHON_BIN="$HOME/anaconda3/envs/tlqkf/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi
export HYDRA_FULL_ERROR=1
export CONDA_SOLVER=${CONDA_SOLVER:-classic}
export PYTHONDONTWRITEBYTECODE=1
unset LD_PRELOAD || true
unset CONDA_EXE CONDA_PYTHON_EXE CONDA_SHLVL || true
export CUDA_LAUNCH_BLOCKING=1
export NVIDIA_TF32_OVERRIDE=0
export PYTORCH_NVFUSER_DISABLE=1
export TORCH_USE_CUDA_DSA=0
export CUDA_MODULE_LOADING=LAZY
LIB_MODE=${LIB_MODE:-unset}
if [[ "$LIB_MODE" == "conda" ]]; then
  export LD_LIBRARY_PATH="$HOME/anaconda3/envs/tlqkf/lib"
else
  unset LD_LIBRARY_PATH || true
fi
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
echo "✅ Python environment setup completed"
echo ""

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-none} ARRAY_ID=${SLURM_ARRAY_TASK_ID:-none}"

calc_workers() {
  local cpus="${SLURM_CPUS_PER_TASK:-4}"; local maxw="${PYTORCH_WORKERS_MAX:-4}"
  (( cpus < 1 )) && cpus=1; if (( maxw < cpus )); then echo "$maxw"; else echo "$cpus"; fi
}
WORKERS="$(calc_workers)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$WORKERS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$WORKERS}"
echo "🧵 num_workers(auto)=${WORKERS}, OMP=${OMP_NUM_THREADS}, MKL=${MKL_NUM_THREADS}"

PY=${PY:-"$PYTHON_BIN"}
CFG=${CFG:-experiment/sota_generic}
RESULTS_ROOT=${RESULTS_ROOT:-$ROOT/experiments/sota/results}

# Optional filters
ONLY_METHODS=${ONLY_METHODS:-}
ONLY_STUDENTS=${ONLY_STUDENTS:-}
ONLY_PAIRS=${ONLY_PAIRS:-}
SEEDS_STR=${SEEDS_STR:-42}

# Common overrides respected across methods
BASE_OVR=(
  +experiment.use_amp=true
  +experiment.amp_dtype=bfloat16
  +experiment.use_distillation_adapter=true
  +experiment.compute_teacher_eval=false
)

# Teacher pairs and students
TEACHER_PAIRS=(
  "resnet152,convnext_s"
  "convnext_l,efficientnet_l2"
  "resnet152,resnet152"
)

declare -A CKPT=(
  [resnet152]="checkpoints/teachers/resnet152_cifar100.pth"
  [convnext_s]="checkpoints/teachers/convnext_s_cifar100.pth"
  [convnext_l]="checkpoints/teachers/convnext_l_cifar100.pth"
  [efficientnet_l2]="checkpoints/teachers/efficientnet_l2_cifar100.pth"
)

STUDENTS=(mobilenet_v2 resnet50 efficientnet_b0 efficientnet_b2 shufflenet_v2 resnet101)

METHODS=(
  asib_stage
  asib_fair
  vanilla_kd
  avg_kd
  dkd
  ab
  at
  fitnet
  ft
  reviewkd
  crd
  simkd
  sskd
)

IFS=',' read -r -a SEEDS <<< "$SEEDS_STR"

contains() { local x="$1"; shift; for a in "$@"; do [[ "$a" == "$x" ]] && return 0; done; return 1; }

filter_ok() {
  local item="$1" list="$2"; [[ -z "$list" ]] && return 0; IFS=',' read -r -a arr <<< "$list"; contains "$item" "${arr[@]}"
}

method_overrides() {
  local m="$1"; case "$m" in
    asib_stage)
      echo "experiment/method@experiment.method=asib_stage" ;;
    asib_fair)
      echo "experiment/method@experiment.method=asib_fair" ;;
    vanilla_kd)
      echo "experiment/method@experiment.method=vanilla_kd \
        +experiment.kd_target=teacher \
        +experiment.ce_alpha=0.65 +experiment.kd_alpha=0.35" ;;
    avg_kd)
      echo "experiment/method@experiment.method=vanilla_kd \
        +experiment.kd_target=avg \
        +experiment.ce_alpha=0.65 +experiment.kd_alpha=0.35" ;;
    dkd)
      echo "experiment/method@experiment.method=dkd \
        +experiment.ce_alpha=0.65 +experiment.kd_alpha=0.35" ;;
    ab)
      echo "experiment/method@experiment.method=ab" ;;
    at)
      echo "experiment/method@experiment.method=at" ;;
    fitnet)
      echo "experiment/method@experiment.method=fitnet" ;;
    ft)
      echo "experiment/method@experiment.method=ft" ;;
    reviewkd)
      echo "experiment/method@experiment.method=reviewkd" ;;
    crd)
      echo "experiment/method@experiment.method=crd" ;;
    simkd)
      echo "experiment/method@experiment.method=simkd" ;;
    sskd)
      echo "experiment/method@experiment.method=sskd" ;;
    *) echo ""; return 1 ;;
  esac
}

# Map CLI student name to Hydra group key (configs/model/student/*)
student_key_from() {
  case "$1" in
    mobilenet_v2|resnet50|resnet101|efficientnet_b0|efficientnet_b2|shufflenet_v2) echo "${1}_scratch" ;;
    *) echo "$1" ;;
  esac
}

# Choose distillation adapter dim per student
distill_dim_for_student() {
  case "$1" in
    mobilenet_v2_scratch|efficientnet_b0_scratch|efficientnet_b2_scratch|shufflenet_v2_scratch) echo 256 ;;
    *) echo 512 ;;
  esac
}

# Pair/method specific overrides: kd_teacher_index or teacher_weights
pair_overrides() {
  local T1="$1"; local T2="$2"; local method="$3"; local ovr="";
  # Single-teacher family → pick the stronger teacher (index 0 by our pair ordering)
  if [[ "$method" =~ ^(vanilla_kd|dkd)$ ]]; then
    if [[ "$T1" == "resnet152" && "$T2" == "convnext_s" ]]; then
      ovr+=" +experiment.kd_teacher_index=0"
    elif [[ "$T1" == "convnext_l" && "$T2" == "efficientnet_l2" ]]; then
      ovr+=" +experiment.kd_teacher_index=0"
    fi
  fi
  # Avg/ASIB family → teacher weights
  if [[ "$method" =~ ^(avg_kd|asib_stage|asib_fair)$ ]]; then
    if [[ "$T1" == "resnet152" && "$T2" == "convnext_s" ]]; then
      ovr+=" +experiment.teacher_weights=[0.7,0.3]"
    elif [[ "$T1" == "convnext_l" && "$T2" == "efficientnet_l2" ]]; then
      ovr+=" +experiment.teacher_weights=[0.6,0.4]"
    elif [[ "$T1" == "resnet152" && "$T2" == "resnet152" ]]; then
      ovr+=" +experiment.teacher_weights=[0.5,0.5]"
    fi
  fi
  echo "$ovr"
}

mkdir -p "$RESULTS_ROOT"

# Optional: enable a single teacher-eval smoke run (env toggle)
TEACHER_EVAL_SMOKE=${TEACHER_EVAL_SMOKE:-0}
TEACHER_EVAL_MAX_BATCHES=${TEACHER_EVAL_MAX_BATCHES:-10}
did_eval_smoke=0

RUNS=()
for pair in "${TEACHER_PAIRS[@]}"; do
  filter_ok "$pair" "$ONLY_PAIRS" || continue
  for student in "${STUDENTS[@]}"; do
    filter_ok "$student" "$ONLY_STUDENTS" || continue
    for method in "${METHODS[@]}"; do
      filter_ok "$method" "$ONLY_METHODS" || continue
      for seed in "${SEEDS[@]}"; do
        RUNS+=("${pair}|${student}|${method}|${seed}")
      done
    done
  done
done

echo "Planned runs: ${#RUNS[@]}"

# Optional dry-run (index→spec mapping only)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Planned runs: ${#RUNS[@]}"
  for i in "${!RUNS[@]}"; do
    echo "IDX=$i | ${RUNS[$i]}"
  done
  exit 0
fi

# Slurm array mode: run only the indexed item
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  IDX="${SLURM_ARRAY_TASK_ID}"
  TOTAL="${#RUNS[@]}"
  echo "[INFO] Array index ${IDX} / ${TOTAL}"
  if (( IDX < 0 || IDX >= TOTAL )); then
    echo "[INFO] Index out of range. Nothing to do. Exit 0."
    exit 0
  fi
  RUNS=( "${RUNS[$IDX]}" )
fi

for spec in "${RUNS[@]}"; do
  IFS='|' read -r pair student method seed <<< "$spec"
  IFS=',' read -r T1 T2 <<< "$pair"

  OVRS="$(method_overrides "$method")"
  if [[ -z "$OVRS" ]]; then echo "Skip unknown method: $method"; continue; fi
  read -r -a OV_ARR <<< "$OVRS"

  # Resolve student Hydra key and adapter dim
  STU_KEY="$(student_key_from "$student")"
  D_DIM="$(distill_dim_for_student "$STU_KEY")"

  # Check teacher checkpoints exist; skip run if missing
  for tk in "$T1" "$T2"; do
    ck="${CKPT[$tk]:-}"
    if [[ -z "$ck" || ! -f "$ck" ]]; then
      echo "Missing CKPT: $tk → $ck (skipping)"
      continue 2
    fi
  done

  OUT_DIR="$RESULTS_ROOT/${T1}-${T2}/${student}/${method}/seed${seed}"
  mkdir -p "$OUT_DIR"

  CMD=("$PY" -u main.py -cn="$CFG"
    model/teacher@experiment.teacher1="$T1"
    model/teacher@experiment.teacher2="$T2"
    model/student@experiment.model.student="$STU_KEY"
    +experiment.teacher1.pretrained=true
    +experiment.teacher2.pretrained=true
    +experiment.teacher1_ckpt="${CKPT[$T1]:-null}"
    +experiment.teacher2_ckpt="${CKPT[$T2]:-null}"
    +experiment.results_dir="$OUT_DIR"
    +experiment.exp_id="sota__${method}__${T1}-${T2}__${student}__s${seed}"
    +seed="$seed"
    +experiment.distill_out_dim="$D_DIM"
  )

  CMD+=("${BASE_OVR[@]}")
  CMD+=("${OV_ARR[@]}")
  # Pair overrides appended last
  PAIR_OVR="$(pair_overrides "$T1" "$T2" "$method")"
  if [[ -n "$PAIR_OVR" ]]; then
    read -r -a PAIR_OVR_ARR <<< "$PAIR_OVR"
    CMD+=("${PAIR_OVR_ARR[@]}")
  fi

  # Debug: show pair overrides for visibility
  printf "[pair_overrides] %s,%s %s -> %s\n" "$T1" "$T2" "$method" "$PAIR_OVR"

  # Optional DataLoader workers override via env
  if [[ -n "${NUM_WORKERS:-}" ]]; then
    CMD+=(+experiment.dataset.num_workers="${NUM_WORKERS}")
  fi

  # One-time teacher-eval smoke if enabled
  if [[ "$TEACHER_EVAL_SMOKE" == "1" && "$did_eval_smoke" == "0" ]]; then
    CMD+=(
      +experiment.compute_teacher_eval=true
      +experiment.teacher_eval_max_batches="$TEACHER_EVAL_MAX_BATCHES"
      +experiment.teacher_eval_on_gpu=true
      +experiment.teacher_eval_amp=true
      +experiment.teacher_eval_batch_size=128
    )
    did_eval_smoke=1
  fi

  # Optional: 1-epoch smoke via env EPOCHS
  if [[ -n "${EPOCHS:-}" ]]; then
    CMD+=(+experiment.student_epochs="${EPOCHS}")
    if [[ "${EPOCHS}" == "1" && ( "$method" == "asib_stage" || "$method" == "asib" ) ]]; then
      CMD+=(+experiment.teacher_adapt_epochs=1)
    fi
  fi

  printf "\n🚀 %s\n\n" "${CMD[*]}"
  set +e
  "${CMD[@]}"; ret=$?
  set -e
  if [[ $ret -ne 0 ]]; then
    echo "❌ Failed: $OUT_DIR (exit=$ret) — continue"
    echo "{\"status\":\"failed\",\"code\":$ret}" > "$OUT_DIR/failed.json"
    continue
  fi
done

printf "\n🎉 All runs finished. See %s/\n" "${RESULTS_ROOT}"


