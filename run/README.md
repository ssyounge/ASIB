# Run Scripts

이 폴더는 SLURM 클러스터에서 실행할 수 있는 배치 스크립트들을 포함합니다.

## 📁 스크립트 목록

| 스크립트 | 용도 | 실행 시간 | GPU |
|---------|------|----------|-----|
| `run_ablation_study.sh` | Phase 1: Ablation Study | ~8-12시간 | 1 |
| `run_finetune_single.sh` | Teacher 파인튜닝 (단일) | ~1-2시간 | 1 |
| `run_finetune_all_teachers.sh` | Teacher 파인튜닝 (전체) | ~4-6시간 | 1 |

## 🚀 사용법

### 체계적 실험 실행
```bash
# Phase 1: Ablation Study (모든 단계)
sbatch run/run_ablation_study.sh

# Teacher Fine-tuning
sbatch run/run_finetune_single.sh convnext_s_cifar32
sbatch run/run_finetune_single.sh convnext_l_cifar32
sbatch run/run_finetune_single.sh efficientnet_l2_cifar32
sbatch run/run_finetune_single.sh resnet152_cifar32

# 또는 전체 Teacher 파인튜닝
sbatch run/run_finetune_all_teachers.sh
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
- **결과**: `outputs/ablation_*/`, `outputs/sota_*/`, `outputs/overlap_*/`
- **체크포인트**: `checkpoints/`

## 🔄 정리된 파일들

다음 파일들은 새로운 체계적 실험 계획에 맞춰 정리되었습니다:

### 제거된 파일들
- `run.sh` → 새로운 config 기반 실험으로 대체
- `run_sensitivity.sh` → `run_ablation_study.sh`로 통합
- `run_overlap.sh` → 새로운 `overlap_*.yaml` config로 대체

### 새로운 실험 계획
- **Phase 1**: Ablation Study (`ablation_*.yaml`)
- **Phase 2**: SOTA Comparison (`sota_*.yaml`) 
- **Phase 3**: Overlap Analysis (`overlap_*.yaml`)

자세한 내용은 `EXPERIMENT_PLAN.md`를 참조하세요. 