#!/usr/bin/env bash
#SBATCH --job-name=asib_methods_sanity
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-10%6
#SBATCH --output=experiments/sota/logs/methods_sanity_%A_%a.log
#SBATCH --error=experiments/sota/logs/methods_sanity_%A_%a.err

set -euo pipefail
trap 'echo "❌ Job failed at $(date)"; exit 1' ERR

echo "🔧 Setting up Python environment..."
# Use direct interpreter instead of conda activate to avoid libmamba/libstdc++ issues
PYTHON_BIN="$HOME/anaconda3/envs/tlqkf/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi
export HYDRA_FULL_ERROR=1
# Harden env to avoid libstdc++/solver crashes
export CONDA_SOLVER=${CONDA_SOLVER:-classic}
export PYTHONDONTWRITEBYTECODE=1
unset LD_PRELOAD || true
# Also avoid accidental conda entrypoint usage
unset CONDA_EXE CONDA_PYTHON_EXE CONDA_SHLVL || true
export CUDA_LAUNCH_BLOCKING=1
export NVIDIA_TF32_OVERRIDE=0
export PYTORCH_NVFUSER_DISABLE=1
export TORCH_USE_CUDA_DSA=0
export CUDA_MODULE_LOADING=LAZY
# LD_LIBRARY_PATH A/B toggle (unset | conda)
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

# Repo root
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
mkdir -p "$ROOT/experiments/sota/logs" || true
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# num_workers 자동 스케일
calc_workers() {
  local cpus="${SLURM_CPUS_PER_TASK:-4}"; local maxw="${PYTORCH_WORKERS_MAX:-4}"
  (( cpus < 1 )) && cpus=1; if (( maxw < cpus )); then echo "$maxw"; else echo "$cpus"; fi
}
WORKERS="$(calc_workers)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$WORKERS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$WORKERS}"
echo "🧵 num_workers(auto)=${WORKERS}, OMP=${OMP_NUM_THREADS}, MKL=${MKL_NUM_THREADS}"

# Fixed teachers, student set, and parameters; vary only method
TEACHER_PAIR="${TEACHER_PAIR:-resnet152,convnext_s}"
IFS=',' read -r T1 T2 <<< "$TEACHER_PAIR"

declare -A CKPT=(
  [resnet152]="checkpoints/teachers/resnet152_cifar100.pth"
  [convnext_s]="checkpoints/teachers/convnext_s_cifar100.pth"
  [convnext_l]="checkpoints/teachers/convnext_l_cifar100.pth"
  [efficientnet_l2]="checkpoints/teachers/efficientnet_l2_cifar100.pth"
)

# 동일한 학생 1종, 동일한 하이퍼, 메소드만 변경
# 환경변수 STUDENT 로 지정(기본: mobilenet_v2)
STUDENT="${STUDENT:-mobilenet_v2}"
# 지원 메소드 전부 포함 (필요 시 ONLY_METHODS 환경변수로 필터) — avg_kd 제외
METHODS=(ce_sgd_sanity ce_same_recipe vanilla_kd kd_single_teacher kd_weighted_conf dkd asib_stage asib_fair at fitnet crd)
SEED=42

method_overrides() {
  case "$1" in
    ce_sgd_sanity) echo "experiment/method@experiment.experiment.method=ce_sgd_sanity" ;;
    ce_same_recipe) echo "experiment/method@experiment.experiment.method=ce_same_recipe" ;;
    vanilla_kd) echo "experiment/method@experiment.experiment.method=vanilla_kd" ;;
    asib_stage) echo "experiment/method@experiment.experiment.method=asib_stage" ;;
    asib_fair)  echo "experiment/method@experiment.experiment.method=asib_fair" ;;
    dkd)        echo "experiment/method@experiment.experiment.method=dkd" ;;
    at)         echo "experiment/method@experiment.experiment.method=at" ;;
    fitnet)     echo "experiment/method@experiment.experiment.method=fitnet" ;;
    ft)         echo "experiment/method@experiment.experiment.method=ft" ;;
    reviewkd)   echo "experiment/method@experiment.experiment.method=reviewkd" ;;
    crd)        echo "experiment/method@experiment.experiment.method=crd" ;;
    ab)         echo "experiment/method@experiment.experiment.method=ab" ;;
    kd_single_teacher) echo "experiment/method@experiment.experiment.method=kd_single_teacher" ;;
    kd_weighted_conf)  echo "experiment/method@experiment.experiment.method=kd_weighted_conf" ;;
    *) echo ""; return 1 ;;
  esac
}

if [[ -n "${ONLY_METHODS:-}" ]]; then IFS=',' read -r -a METHODS <<< "$ONLY_METHODS"; fi
RUNS=()
for m in "${METHODS[@]}"; do RUNS+=("$m"); done

N_RUNS=${#RUNS[@]}
echo "🧮 Planned runs: ${N_RUNS}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  idx="${SLURM_ARRAY_TASK_ID}"
  if (( idx >= N_RUNS )); then echo "ℹ️  Array index ${idx} >= N_RUNS ${N_RUNS} → nothing to do."; exit 0; fi
  METHODS=("${RUNS[$idx]}")
fi

# Run
student="$STUDENT"
for method in "${METHODS[@]}"; do
    RESULTS_BASE="experiments/sota/results/${T1}-${T2}/${student}/${method}"
    RESULTS_DIR="${RESULTS_BASE}/seed${SEED}"
    # If create-new is requested and base run exists, allocate a new indexed run dir
    if [[ -d "$RESULTS_DIR" && "${CREATE_NEW_RUN:-0}" == "1" ]]; then
      idx=1
      while [[ -d "${RESULTS_BASE}/seed${SEED}_run${idx}" ]]; do idx=$((idx+1)); done
      RESULTS_DIR="${RESULTS_BASE}/seed${SEED}_run${idx}"
      RUN_IDX="$idx"
    else
      RUN_IDX=""
    fi
    mkdir -p "$RESULTS_DIR"
    LATEST="$RESULTS_DIR/latest.json"
    if [[ -f "$LATEST" ]]; then
      if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
        echo "♻️  FORCE_RERUN=1 → removing $LATEST and re-running ${method}"
        rm -f "$LATEST"
      else
        # default: keep existing run intact and auto-create a new indexed run dir
        idx=1
        while [[ -d "${RESULTS_BASE}/seed${SEED}_run${idx}" ]]; do idx=$((idx+1)); done
        RESULTS_DIR="${RESULTS_BASE}/seed${SEED}_run${idx}"
        RUN_IDX="$idx"
        mkdir -p "$RESULTS_DIR"
        LATEST="$RESULTS_DIR/latest.json"
        echo "➕ Existing run found. Creating new run directory: $RESULTS_DIR"
      fi
    fi

    OVRS="$(method_overrides "$method")"
    if [[ -z "$OVRS" ]]; then echo "Unknown method: $method"; exit 1; fi

    # Minimal overrides only; no dataset/optimizer/etc.

    # Compose exp_id with optional run index suffix
    EXP_ID="sota__${method}__${T1}-${T2}__${STUDENT}__s${SEED}"
    if [[ -n "${RUN_IDX}" ]]; then EXP_ID="${EXP_ID}__r${RUN_IDX}"; fi

    # Use conda-run to isolate environment and avoid libstdc++/loader conflicts
    # Use fixed interpreter to avoid conda entrypoint
    CLEAN_ENV=${CLEAN_ENV:-1}
    if [[ "$CLEAN_ENV" == "1" ]]; then
      CLEAN_PREFIX=(env -i PATH="$PATH" HOME="$HOME" USER="$USER" PYTHONPATH="$PYTHONPATH" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        CUDA_LAUNCH_BLOCKING=1 NVIDIA_TF32_OVERRIDE=0 PYTORCH_NVFUSER_DISABLE=1 \
        CUDA_MODULE_LOADING=LAZY NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1 \
        NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN TORCH_USE_CUDA_DSA=0)
    else
      CLEAN_PREFIX=()
    fi
    PY_CMD=("${CLEAN_PREFIX[@]}" "$PYTHON_BIN" -u)
    CMD=("${PY_CMD[@]}" main.py -cn=experiment/sota_generic
      +results_dir="$RESULTS_DIR"
      +exp_id="$EXP_ID"
      +seed="$SEED"
    )

    # Unique Hydra run/output dir per task to avoid FileExistsError on config.yaml
    HYDRA_DIR="${RESULTS_DIR}/.hydra_runs/${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}_$(date +%s%N)"
    mkdir -p "$HYDRA_DIR"
    CMD+=(hydra.run.dir="$HYDRA_DIR" hydra.output_subdir=null)
    # Safe/Perf mode flags
    SAFE_MODE=${SAFE_MODE:-0}
    if [[ "$SAFE_MODE" == "1" ]]; then
      # Reduce crash probability (rely on YAML for AMP/channels_last)
      CMD+=(+experiment.use_safe_mode=true)
    else
      # Performance mode: rely on YAML for AMP/channels_last; only set workers here
      CMD+=(+experiment.use_safe_mode=false)
      # Ensure multi-worker if not overridden elsewhere
      CMD+=(+experiment.dataset.num_workers=${WORKERS})
    fi
    # Disable NCCL transports to avoid rare init crashes
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export NCCL_SHM_DISABLE=1
    export NCCL_ASYNC_ERROR_HANDLING=1
    export NCCL_DEBUG=WARN

    # method overrides only
    echo "OVRS: $OVRS"
    # Optional CPU smoke import to verify torch import without CUDA
    if [[ "${CPU_SMOKE:-0}" == "1" ]]; then
      echo "[CPU SMOKE] Verifying torch import..."
      set +e; "${CLEAN_PREFIX[@]}" "$PYTHON_BIN" -u - <<'PY'
import torch, sys
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("done")
PY
      set -e
    fi
    # Optional GPU smoke to verify CUDA visibility and cuDNN
    if [[ "${GPU_SMOKE:-0}" == "1" ]]; then
      echo "[GPU SMOKE] Verifying CUDA..."
      set +e; "${CLEAN_PREFIX[@]}" "$PYTHON_BIN" -u - <<'PY'
import os, torch
print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device0:", torch.cuda.get_device_name(0))
    import torch.backends.cudnn as cudnn
    print("cudnn.enabled:", cudnn.enabled, "version:", cudnn.version())
PY
      set -e
    fi
    read -r -a OV_ARR <<< "$OVRS"
    CMD+=("${OV_ARR[@]}")
    # Ensure PPF/BN enabled for ASIB runs regardless of base defaults (namespaced under experiment.*)
    # Fairness: method YAMLs control policy; avoid forcing leaf overrides
    # CE-only: speed up by skipping teacher eval
    case "$method" in
      ce_only|ce_sgd_sanity|ce_same_recipe)
        CMD+=(+experiment.compute_teacher_eval=false)
        ;;
    esac
    if [[ "$method" == "ce_only" ]]; then
      # MobileNetV2 scratch에서 안정 수렴을 위해 LR 0.05로 시작
      CMD+=(+experiment.student_lr=0.05)
    fi
    # Optional safe mode (default ON): reduce runtime crashes by using single-worker
    if [[ "$SAFE_MODE" == "1" ]]; then
      # Add dataset workers flag safely; do NOT override AMP by default
      CMD+=("+experiment.dataset.num_workers=0")
      # Optional: allow forcing AMP off via env if needed
      if [[ "${FORCE_AMP_OFF:-0}" == "1" ]]; then
        CMD+=(+experiment.use_amp=false)
      fi
    fi
    # Rely on method YAMLs for AMP/channels_last/two_view stop/KD gating/cooldown

    # CKPT/평가 오버라이드는 CLI에서 제거 (strict struct 충돌 방지)

    echo "🚀 ${CMD[*]}"
    # Temporarily disable ERR trap around Python execution to allow loop continuation
    trap - ERR
    set +e; "${CMD[@]}"; ret=$?; set -e
    trap 'echo "❌ Job failed at $(date)"; exit 1' ERR
    if [[ $ret -ne 0 ]]; then
      echo "❌ Failed: $RESULTS_DIR (exit=$ret) — continue"
      echo "{\"status\":\"failed\",\"code\":$ret}" > "$RESULTS_DIR/failed.json"
      continue
    fi
    echo "✅ Done: $RESULTS_DIR"
done

echo "🎉 Methods sanity sweep finished."


