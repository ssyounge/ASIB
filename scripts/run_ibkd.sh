#!/usr/bin/env bash
# scripts/run_ibkd.sh
set -e

# Ensure repo root for PYTHONPATH
export PYTHONPATH="$(pwd):${PYTHONPATH}"

MODE="loop"
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 [--mode MODE] --output_dir DIR" >&2
  exit 1
fi

# Use same conda environment as other scripts if available
source ~/.bashrc
conda activate facil_env

# Delegate to run_experiments.sh for actual training logic
bash scripts/run_experiments.sh --mode "$MODE" --output_dir "$OUTPUT_DIR"
