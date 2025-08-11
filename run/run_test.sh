#!/usr/bin/env bash
#SBATCH --job-name=run_tests
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --chdir=/home/suyoung425/ASIB
#SBATCH --output=/home/suyoung425/ASIB/experiments/test/logs/slurm-%j.out
# Simple unified test runner on Linux/SLURM
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Ensure logs directory exists
mkdir -p "$ROOT/experiments/test/logs"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# If running under SLURM and CUDA_VISIBLE_DEVICES not set, default to first GPU
if [[ -n "${SLURM_JOB_ID:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

# Quick mode by default; pass FULL=1 to run everything
ARGS=(tests -v)
if [[ "${FULL:-0}" != "1" ]]; then
  ARGS+=(
    --ignore=tests/test_asib_cl.py
    --ignore-glob=tests/test_pycil_*.py
    --ignore=tests/test_cl_experiments.py
    --ignore=tests/test_experiment_execution.py
    --ignore=tests/test_modules_partial_freeze.py
    --deselect=tests/test_final_validation.py::TestFinalValidation::test_framework_robustness
    --deselect=tests/test_final_validation.py::TestFinalValidation::test_experiment_scripts_executable
    --deselect=tests/test_integration.py::TestCompletePipeline::test_logging_pipeline
  )
fi

python -m pytest "${ARGS[@]}" | tee "$ROOT/experiments/test/logs/core_functionality_test.log"