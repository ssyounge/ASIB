#!/usr/bin/env bash
#SBATCH --job-name=asib_class_overlap
#SBATCH --partition=suma_a6000
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=experiments/logs/phase3_%j.log
#SBATCH --error=experiments/logs/phase3_%j.err
# ---------------------------------------------------------
# ASIB Class Overlap Analysis 실험
# 100% Class Overlap 상황에서의 ASIB 성능 분석
# ---------------------------------------------------------
set -euo pipefail

# 1) 리포 최상위로 이동
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# 2) PYTHONPATH 추가
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# 3) ASIB Class Overlap Analysis 실험들
EXPERIMENTS=(
    "overlap_100"           # 100% Class Overlap Analysis
)

# 4) 각 실험 순차적으로 실행
for exp in "${EXPERIMENTS[@]}"; do
    echo "🚀 Starting ASIB Class Overlap Analysis experiment: $exp"
    echo "=================================================="
    echo "Time: $(date)"
    echo "Experiment: $exp"
    echo "ASIB Class Overlap Analysis"
    echo "=================================================="
    
    # 실험 실행
    python main.py \
        --config-name "experiment/$exp" \
        "$@"
    
    echo "✅ Finished ASIB Class Overlap Analysis experiment: $exp"
    echo "=================================================="
    echo "Time: $(date)"
    echo ""
done

echo "🎉 ASIB Class Overlap Analysis completed!"
echo "📁 Results saved in: outputs/overlap/"
echo "📊 All ASIB experiments completed! Ready for analysis" 