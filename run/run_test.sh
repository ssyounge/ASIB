#!/usr/bin/env bash
#SBATCH --job-name=run_tests
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
# Suppress SLURM's default slurm-%j.{out,err} files entirely
# (we manage our own grouped logs below).
# It is acceptable to use /dev/null here to avoid stray files.
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# Simple unified test runner on Linux/SLURM
set -euo pipefail

# Determine repo root (prefer SLURM submission directory)
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR"
  ROOT="$(pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  cd "$SCRIPT_DIR/.."
  ROOT="$(pwd)"
fi

# Ensure logs and results directories exist
mkdir -p "$ROOT/experiments/test/logs"
mkdir -p "$ROOT/experiments/test/results"

# Cleanup any legacy .status files from older runs (no longer used)
rm -f "$ROOT/experiments/test/logs"/*.status 2>/dev/null || true



# Python/conda environment
export PATH="$HOME/anaconda3/envs/tlqkf/bin:$PATH"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# CUDA & PyTorch runtime settings
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset LD_LIBRARY_PATH || true
export CUDA_HOME="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib"
export CUDA_PATH="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib"
export CUDA_ROOT="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib"
export TORCH_CUDA_ARCH_LIST="8.6"

# Overall timing
OVERALL_START_EPOCH=$(date +%s)
OVERALL_START_HUMAN=$(date +'%F %T')

# Default to parallel group execution unless explicitly disabled
: "${PARALLEL:=1}"
: "${MAX_PARALLEL:=${SLURM_CPUS_ON_NODE:-3}}"

# GPU allocation mapping (prefer SLURM_GPUS_ON_NODE)
echo "🔍 Checking GPU allocation..."
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  if [[ "${SLURM_GPUS_ON_NODE}" == "1" ]]; then
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

# GPU info
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "🔍 GPU Information:"
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || true
fi

# Helper to run a group and write to a named log
declare -a PIDS=()
declare -a GROUP_NAMES=()
declare -a GROUP_LOGS=()
declare -a GROUP_EXITCODES=()
declare -a GROUP_START=()
declare -a GROUP_END=()
declare -A PID_INDEX=()
run_group() {
  local log_name="$1"; shift
  local patterns=("$@")
  echo "[run_test] Running group: ${log_name} -> ${patterns[*]}"
  local LOG_FILE="$ROOT/experiments/test/logs/${log_name}.log"
  GROUP_NAMES+=("$log_name")
  GROUP_LOGS+=("$LOG_FILE")
  GROUP_EXITCODES+=("-1")
  GROUP_START+=("$(date +%s)")
  GROUP_END+=("-1")
  local idx=$((${#GROUP_NAMES[@]} - 1))
  if [[ "${PARALLEL:-0}" == "1" ]]; then
    # Run in background, redirecting output to its own log
    (
      set +e
      python -m pytest -v "${patterns[@]}" > "$LOG_FILE" 2>&1
      exit $?
    ) &
    local pid=$!
    PIDS+=($pid)
    PID_INDEX[$pid]="$idx"
    # Optional throttle via MAX_PARALLEL
    if [[ -n "${MAX_PARALLEL:-}" ]]; then
      while [[ $(jobs -p | wc -l) -ge ${MAX_PARALLEL} ]]; do
        wait -n || true
      done
    fi
  else
    set +e
    python -m pytest -v "${patterns[@]}" | tee "$LOG_FILE"
    # Capture exit code of first cmd in the pipeline (python)
    local ec=${PIPESTATUS[0]}
    set -e
    GROUP_EXITCODES[$idx]="$ec"
    GROUP_END[$idx]="$(date +%s)"
  fi
}

finalize_summary() {
  local SUMMARY_FILE="$ROOT/experiments/test/results/summary.log"
  {
    echo "==== Test Summary ($(date +'%F %T')) ===="
    echo "Root: $ROOT"
    echo "Parallel: ${PARALLEL:-0}, Max Parallel: ${MAX_PARALLEL:-N/A}"
    echo "Start: ${OVERALL_START_HUMAN}"
    local END_EPOCH=$(date +%s)
    local DURATION=$((END_EPOCH - OVERALL_START_EPOCH))
    echo "End: $(date +'%F %T')"
    echo "Duration(s): ${DURATION}"

    # GPU / CUDA info
    local CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-unset}"
    local CUDA_AVAIL="unknown"
    local CUDA_VERSION="unknown"
    local GPU_NAMES="unknown"
    local KERNEL_INFO="$(uname -r)"
    local HOSTNAME_INFO="$(hostname)"
    local PY_VERSION="unknown"
    local TORCH_VERSION="unknown"
    local CUDNN_VERSION="unknown"
    # torch cuda availability
    CUDA_AVAIL=$(python - <<'PY'
try:
 import torch
 print('True' if torch.cuda.is_available() else 'False')
except Exception:
 print('Error')
PY
)
    # Python/Torch/cuDNN versions
    readarray -t PYT_INFO < <(python - <<'PY'
import sys
try:
 import torch
 v_torch = getattr(torch, '__version__', 'unknown')
 v_cudnn = getattr(getattr(torch.backends, 'cudnn', None), 'version', None)
 v_cudnn = v_cudnn() if callable(v_cudnn) else (v_cudnn or 'unknown')
except Exception:
 v_torch = 'error'
 v_cudnn = 'error'
print(sys.version.split()[0])
print(v_torch)
print(v_cudnn)
PY
)
    PY_VERSION="${PYT_INFO[0]:-unknown}"
    TORCH_VERSION="${PYT_INFO[1]:-unknown}"
    CUDNN_VERSION="${PYT_INFO[2]:-unknown}"
    # CUDA version (if nvcc exists)
    if command -v nvcc >/dev/null 2>&1; then
      CUDA_VERSION=$(nvcc --version | tail -n1 | sed 's/^.*release \([^,]*\),.*$/\1/')
    fi
    # GPU names via nvidia-smi if available
    if command -v nvidia-smi >/dev/null 2>&1; then
      if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
        GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1{print; exit}')
      else
        GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | paste -sd ',')
      fi
    fi
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_DEVICES}"
    echo "CUDA available (torch): ${CUDA_AVAIL}"
    echo "CUDA version: ${CUDA_VERSION}"
    echo "GPU: ${GPU_NAMES}"
    echo "Host: ${HOSTNAME_INFO}"
    echo "Kernel: ${KERNEL_INFO}"
    echo "Python: ${PY_VERSION}"
    echo "PyTorch: ${TORCH_VERSION}"
    echo "cuDNN: ${CUDNN_VERSION}"
    echo

    # Totals per group and aggregated test counts
    local total_groups=${#GROUP_NAMES[@]}
    local groups_passed=0
    local groups_failed=0
    local tests_passed=0
    local tests_failed=0
    local tests_skipped=0
    local tests_xfailed=0
    local tests_xpassed=0
    local tests_errors=0
    local tests_total=0

    # helper to parse pytest summary numbers from a log
    parse_summary_counts() {
      local log_file="$1"
      local last_summary
      last_summary=$(grep -E "[0-9]+ (passed|failed|skipped|xfailed|xpassed|error|errors)" "$log_file" | tail -n1)
      # Extract numbers if present
      echo "$last_summary"
    }

    if [[ ${#GROUP_NAMES[@]} -eq 0 ]]; then
      # Fallback: discover logs and summarize even if arrays are empty
      shopt -s nullglob
      for log in "$ROOT/experiments/test/logs"/*.log; do
        [[ "$(basename "$log")" == "summary.log" ]] && continue
        local name
        name=$(basename "$log" .log)
        local status="UNKNOWN"
        local p=0 f=0 s=0 xf=0 xp=0 e=0
        local summary_line
        summary_line=$(parse_summary_counts "$log")
        p=$(echo "$summary_line" | grep -Eo "[0-9]+ passed"   | awk '{print $1}' | tail -n1 || echo 0)
        f=$(echo "$summary_line" | grep -Eo "[0-9]+ failed"   | awk '{print $1}' | tail -n1 || echo 0)
        s=$(echo "$summary_line" | grep -Eo "[0-9]+ skipped"  | awk '{print $1}' | tail -n1 || echo 0)
        xf=$(echo "$summary_line"| grep -Eo "[0-9]+ xfailed"  | awk '{print $1}' | tail -n1 || echo 0)
        xp=$(echo "$summary_line"| grep -Eo "[0-9]+ xpassed"  | awk '{print $1}' | tail -n1 || echo 0)
        e=$(echo "$summary_line" | grep -Eo "[0-9]+ errors?"  | awk '{print $1}' | tail -n1 || echo 0)
        tests_passed=$((tests_passed + p))
        tests_failed=$((tests_failed + f))
        tests_skipped=$((tests_skipped + s))
        tests_xfailed=$((tests_xfailed + xf))
        tests_xpassed=$((tests_xpassed + xp))
        tests_errors=$((tests_errors + e))
        tests_total=$((tests_total + p + f + s + xf + xp + e))
        # Heuristic: pass if no failed/errors
        if [[ "$f" == "0" && "$e" == "0" ]]; then
          status="PASS"; ((groups_passed++))
        else
          status="FAIL"; ((groups_failed++))
        fi
        echo "- ${name}: ${status} (log: ${log})"
        echo "  tests: ${p} passed, ${f} failed, ${e} errors, ${s} skipped, ${xf} xfailed, ${xp} xpassed"
        if [[ "$status" == "FAIL" && -f "$log" ]]; then
          echo "  tail of log:"
          tail -n 40 "$log" | sed 's/^/    /'
        fi
        echo
      done
      shopt -u nullglob
      total_groups=$((groups_passed + groups_failed))
    else
      for i in "${!GROUP_NAMES[@]}"; do
        local name="${GROUP_NAMES[$i]}"
        local log="${GROUP_LOGS[$i]}"
      local status="UNKNOWN"
        local p=0 f=0 s=0 xf=0 xp=0 e=0
        if [[ -f "$log" ]]; then
          local summary_line
          summary_line=$(parse_summary_counts "$log")
          p=$(echo "$summary_line" | grep -Eo "[0-9]+ passed"   | awk '{print $1}' | tail -n1 || echo 0)
          f=$(echo "$summary_line" | grep -Eo "[0-9]+ failed"   | awk '{print $1}' | tail -n1 || echo 0)
          s=$(echo "$summary_line" | grep -Eo "[0-9]+ skipped"  | awk '{print $1}' | tail -n1 || echo 0)
          xf=$(echo "$summary_line"| grep -Eo "[0-9]+ xfailed"  | awk '{print $1}' | tail -n1 || echo 0)
          xp=$(echo "$summary_line"| grep -Eo "[0-9]+ xpassed"  | awk '{print $1}' | tail -n1 || echo 0)
          e=$(echo "$summary_line" | grep -Eo "[0-9]+ errors?"  | awk '{print $1}' | tail -n1 || echo 0)
        fi
        tests_passed=$((tests_passed + p))
        tests_failed=$((tests_failed + f))
        tests_skipped=$((tests_skipped + s))
        tests_xfailed=$((tests_xfailed + xf))
        tests_xpassed=$((tests_xpassed + xp))
        tests_errors=$((tests_errors + e))
        tests_total=$((tests_total + p + f + s + xf + xp + e))
        local ec="${GROUP_EXITCODES[$i]}"
        if [[ "$ec" == "0" ]]; then
          status="PASS"; ((groups_passed++))
        elif [[ "$ec" == "-1" ]]; then
          status="UNKNOWN"
        else
          status="FAIL"; ((groups_failed++))
        fi
        # Duration if available
        local dur=""
        if [[ "${GROUP_START[$i]:--1}" != "-1" && "${GROUP_END[$i]:--1}" != "-1" ]]; then
          dur=$(( GROUP_END[$i] - GROUP_START[$i] ))
        fi
        echo "- ${name}: ${status} (log: ${log})${dur:+, duration(s): ${dur}}"
        echo "  tests: ${p} passed, ${f} failed, ${e} errors, ${s} skipped, ${xf} xfailed, ${xp} xpassed"
        if [[ "$status" == "FAIL" && -f "$log" ]]; then
          echo "  tail of log:"
          tail -n 40 "$log" | sed 's/^/    /'
        fi
        echo
      done
    fi
    echo "Group totals: ${groups_passed} passed, ${groups_failed} failed, ${total_groups} total"
    echo "Test totals: ${tests_passed} passed, ${tests_failed} failed, ${tests_errors} errors, ${tests_skipped} skipped, ${tests_xfailed} xfailed, ${tests_xpassed} xpassed, ${tests_total} total"
  } > "$SUMMARY_FILE"
  echo "[run_test] Wrote summary to $SUMMARY_FILE"
}

# Quick mode by default; pass FULL=1 to run everything
if [[ "${FULL:-0}" != "1" ]]; then
  # Core functionality
  run_group core_functionality_test \
    tests/test_core.py tests/test_core_utils.py tests/test_utils_common.py

  # Model & Module
  run_group model_module_test \
    tests/test_models*.py tests/test_modules*.py

  # Data & Config
  run_group data_config_test \
    tests/test_data.py tests/test_configs.py tests/test_finetune_configs.py

  # Training & Experiment
  run_group training_experiment_test \
    tests/test_main*.py tests/test_asib*.py tests/test_training*.py

  # Integration & Validation
  run_group integration_validation_test \
    tests/test_integration.py tests/test_final_validation.py

  # Analysis & Scripts
  run_group analysis_script_test \
    tests/test_scripts.py

  # Experiment Configs & Execution
  run_group experiment_config_test \
    tests/test_experiment_configs.py tests/test_experiment_execution.py tests/test_hydra_configs.py

  # Registry Comprehensive
  run_group registry_comprehensive_test \
    tests/test_registry_comprehensive.py

  # Specialized Components
  run_group specialized_component_test \
    tests/test_mbm_tensor_shapes.py tests/test_ib_mbm_shapes.py tests/test_kd_methods.py tests/test_models_advanced.py

  # Utility
  run_group utility_test \
    tests/test_utils.py tests/test_disagreement.py

  # Error Prevention & Robustness
  run_group error_prevention_test \
    tests/test_error_prevention.py tests/test_framework_robustness.py

  # Overlap Dataset
  run_group overlap_dataset_test \
    tests/test_overlap_dataset.py

  # PyCIL Integration (optional heavy)
  run_group pycil_integration_test \
    tests/test_pycil_integration.py tests/test_pycil_models.py tests/test_cl_experiments.py
else
  # Full suite
  run_group full_suite tests -v
fi

# Wait for all background jobs if PARALLEL=1
if [[ "${PARALLEL:-0}" == "1" ]]; then
  echo "[run_test] Waiting for parallel groups to finish..."
  for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
      ec=0
    else
      ec=$?
    fi
    idx="${PID_INDEX[$pid]}"
    GROUP_EXITCODES[$idx]="$ec"
    GROUP_END[$idx]="$(date +%s)"
  done
  echo "[run_test] All groups completed."
fi

# Always produce a summary at the end
finalize_summary