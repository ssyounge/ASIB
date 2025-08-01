#!/bin/bash
#SBATCH --job-name=overlap_analysis
#SBATCH --partition=suma_a600
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=outputs/overlap_%j.log
#SBATCH --error=outputs/overlap_%j.err

# 환경 설정
source ~/.bashrc
conda activate tlqkf  # 또는 사용하는 conda 환경명

# 작업 디렉토리로 이동
cd /home/suyoung425/ASMB_KD

# Overlap analysis 실행
echo "🎯 Starting Overlap Analysis..."
echo "Time: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# 자동화 스크립트 실행
python scripts/overlap_analysis.py

echo "✅ Overlap Analysis completed!"
echo "Time: $(date)" 