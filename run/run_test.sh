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

# Ensure logs directory exists
mkdir -p "$ROOT/experiments/test/logs"



# Python/conda environment
export PATH="$HOME/anaconda3/envs/tlqkf/bin:$PATH"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# CUDA & PyTorch runtime settings
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"
export CUDA_HOME="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib"
export CUDA_PATH="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib"
export CUDA_ROOT="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib"
export TORCH_CUDA_ARCH_LIST="8.6"

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
run_group() {
  local log_name="$1"; shift
  local patterns=("$@")
  echo "[run_test] Running group: ${log_name} -> ${patterns[*]}"
  if [[ "${PARALLEL:-0}" == "1" ]]; then
    # Run in background, redirecting output to its own log
    (
      python -m pytest -v "${patterns[@]}" > "$ROOT/experiments/test/logs/${log_name}.log" 2>&1
    ) &
    PIDS+=($!)
    # Optional throttle via MAX_PARALLEL
    if [[ -n "${MAX_PARALLEL:-}" ]]; then
      while [[ $(jobs -p | wc -l) -ge ${MAX_PARALLEL} ]]; do
        wait -n || true
      done
    fi
  else
    python -m pytest -v "${patterns[@]}" | tee "$ROOT/experiments/test/logs/${log_name}.log"
  fi
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

  # Specialized Components
  run_group specialized_component_test \
    tests/test_mbm_tensor_shapes.py tests/test_ib_mbm_shapes.py tests/test_kd_methods.py tests/test_models_advanced.py

  # Utility
  run_group utility_test \
    tests/test_utils.py tests/test_disagreement.py

  # Error Prevention & Robustness
  run_group error_prevention_test \
    tests/test_error_prevention.py tests/test_framework_robustness.py
else
  # Full suite
  run_group full_suite tests -v
fi

# Wait for all background jobs if PARALLEL=1
if [[ "${PARALLEL:-0}" == "1" ]]; then
  echo "[run_test] Waiting for parallel groups to finish..."
  for pid in "${PIDS[@]}"; do
    wait "$pid" || true
  done
  echo "[run_test] All groups completed."
fi