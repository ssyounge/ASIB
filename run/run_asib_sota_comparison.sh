#!/usr/bin/env bash
#SBATCH --job-name=asib_sota_comparison
#SBATCH --partition=suma_a6000
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=experiments/logs/phase2_%j.log
#SBATCH --error=experiments/logs/phase2_%j.err
# ---------------------------------------------------------
# ASIB SOTA Comparison 실험
# ASIB vs State-of-the-Art Methods 비교
# ---------------------------------------------------------
set -euo pipefail

# 1) 리포 최상위로 이동
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# 2) PYTHONPATH 추가
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# 3) ASIB SOTA Comparison 실험들
EXPERIMENTS=(
    "sota_scenario_a"       # ASIB vs SOTA Methods
)

# 4) 각 실험 순차적으로 실행
for exp in "${EXPERIMENTS[@]}"; do
    echo "🚀 Starting ASIB SOTA Comparison experiment: $exp"
    echo "=================================================="
    echo "Time: $(date)"
    echo "Experiment: $exp"
    echo "ASIB vs SOTA Methods"
    echo "=================================================="
    
    # 실험 실행
    python main.py \
        --config-name "experiment/$exp" \
        "$@"
    
    echo "✅ Finished ASIB SOTA Comparison experiment: $exp"
    echo "=================================================="
    echo "Time: $(date)"
    echo ""
done

echo "🎉 ASIB SOTA Comparison completed!"
echo "📁 Results saved in: outputs/sota/"
echo "📊 Next step: Run ASIB Class Overlap Analysis" 