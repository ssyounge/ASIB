# Run Scripts

이 폴더는 SLURM 클러스터에서 실행할 수 있는 배치 스크립트들을 포함합니다.

## 📁 스크립트 목록

| 스크립트 | 용도 | 실행 시간 | GPU |
|---------|------|----------|-----|
| `run.sh` | 메인 ASIB 실험 | ~2-4시간 | 1 |
| `run_sensitivity.sh` | 기능별 민감도 분석 | ~6-12시간 | 1 |
| `run_overlap.sh` | 클래스 중복도 분석 | ~24-48시간 | 1 |
| `run_finetune_clean.sh` | Teacher 파인튜닝 | ~1-2시간 | 1 |

## 🚀 사용법

### 기본 실행
```bash
# 메인 실험 (res152_convnext_effi)
sbatch run/run.sh

# Sensitivity Analysis
sbatch run/run_sensitivity.sh

# Overlap Analysis
sbatch run/run_overlap.sh

# Teacher Fine-tuning
sbatch run/run_finetune_clean.sh
```

### 상태 확인
```bash
# 작업 상태 확인
squeue -u $USER

# 로그 확인
tail -f outputs/run_*.log
```

### 작업 취소
```bash
# 특정 작업 취소
scancel <job_id>

# 모든 작업 취소
scancel -u $USER
```

## ⚙️ 설정 변경

각 스크립트에서 다음 설정을 필요에 따라 수정하세요:

- `--partition`: GPU 파티션 (suma_a600, suma_v100 등)
- `--time`: 최대 실행 시간
- `--gres=gpu`: GPU 개수
- `--cpus-per-task`: CPU 코어 수

## 📊 결과 확인

실험 결과는 다음 위치에서 확인할 수 있습니다:

- **로그**: `outputs/run_*.log`
- **결과**: `outputs/res152_convnext_effi/`
- **체크포인트**: `checkpoints/` 