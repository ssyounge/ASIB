# Scripts

이 폴더는 ASIB-KD 프레임워크의 다양한 유틸리티 스크립트들을 포함합니다.

## 📁 구조

```
scripts/
├── analysis/           # 분석 스크립트들
│   ├── sensitivity_analysis.py  # 기능별 민감도 분석
│   └── overlap_analysis.py      # 클래스 중복도 분석
├── training/           # 학습 관련 스크립트들
│   ├── fine_tuning.py           # Teacher 파인튜닝
│   └── train_student_baseline.py # Student 베이스라인
└── setup/              # 설정 스크립트들
    └── setup_tests.sh           # 테스트 환경 설정
```

## 🚀 사용법

### 분석 스크립트
```bash
# Sensitivity Analysis
python scripts/analysis/sensitivity_analysis.py

# Overlap Analysis  
python scripts/analysis/overlap_analysis.py
```

### 학습 스크립트
```bash
# Teacher Fine-tuning
python scripts/training/fine_tuning.py

# Student Baseline
python scripts/training/train_student_baseline.py
```

### 설정 스크립트
```bash
# Test Setup
bash scripts/setup/setup_tests.sh
```

## 📋 SLURM 실행

실제 실험은 `run/` 폴더의 SLURM 스크립트를 사용하세요:

```bash
# 메인 실험
sbatch run/run.sh

# Sensitivity Analysis
sbatch run/run_sensitivity.sh

# Overlap Analysis
sbatch run/run_overlap.sh

# Fine-tuning
sbatch run/run_finetune_clean.sh
``` 