#!/usr/bin/env bash
#SBATCH --job-name=asib_ablation_study
#SBATCH --partition=suma_a6000
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=experiments/logs/ablation_%j.log
#SBATCH --error=experiments/logs/ablation_%j.err
# ---------------------------------------------------------
# ASIB Complete Ablation Study 실행
# ASIB 구성 요소들의 점진적 추가 실험 (5단계)
# ---------------------------------------------------------
set -euo pipefail

# 1) 리포 최상위로 이동
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# 2) PYTHONPATH 추가
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# 3) ASIB Complete Ablation Study 실험들
EXPERIMENTS=(
    "ablation_baseline"      # (1) Baseline: MBM + E2E + Fixed Teachers
    "ablation_ib"           # (2) +IB: Information Bottleneck
    "ablation_cccp"         # (3) +IB +CCCP: Stage-wise 학습
    "ablation_tadapt"       # (4) +IB +CCCP +T-Adapt: Teacher Adaptation
    "ablation_full"         # (5) ASIB Full: Progressive Partial Freezing
)

# 4) 각 실험 순차적으로 실행
for exp in "${EXPERIMENTS[@]}"; do
    echo "🚀 Starting ASIB ablation experiment: $exp"
    echo "=================================================="
    echo "Time: $(date)"
    echo "Experiment: $exp"
    echo "ASIB Ablation Study"
    echo "=================================================="
    
    # 실험 실행
    python main.py \
        --config-name "experiment/$exp" \
        "$@"
    
    echo "✅ Finished ASIB ablation experiment: $exp"
    echo "=================================================="
    echo "Time: $(date)"
    echo ""
done

echo "🎉 ASIB ablation experiments completed!"
echo "📁 Results saved in: outputs/ablation/"
echo "📊 Next step: Run ASIB SOTA comparison"
echo "   bash run/run_asib_sota_comparison.sh" 