#!/bin/bash
#SBATCH --job-name=sensitivity_analysis
#SBATCH --partition=suma_a600
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=outputs/sensitivity_%j.log
#SBATCH --error=outputs/sensitivity_%j.err

# 환경 설정
source ~/.bashrc
conda activate tlqkf  # 또는 사용하는 conda 환경명

# 작업 디렉토리로 이동
cd /home/suyoung425/ASMB_KD

# Sensitivity analysis 실행
echo "🎯 Starting Sensitivity Analysis..."
echo "Time: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# 자동화 스크립트 실행
python scripts/sensitivity_analysis.py

echo "✅ Sensitivity Analysis completed!"
echo "Time: $(date)"