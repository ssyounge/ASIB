# Run Scripts

이 폴더는 Windows(로컬)와 Linux(데이터 서버/SLURM) 모두에서 실행 가능한 스크립트를 제공합니다.

## 📁 스크립트 목록

| 스크립트 | 용도 | 실행 시간 | GPU |
|---------|------|----------|-----|
| `run_ablation_study.sh` | Phase 1: Ablation Study | ~8-12시간 | 1 |
| `run_finetune_single.sh` | Teacher 파인튜닝 (단일) | ~1-2시간 | 1 |
| `run_finetune_all_teachers.sh` | Teacher 파인튜닝 (전체) | ~4-6시간 | 1 |

## 🚀 사용법

### Windows (PowerShell)
```powershell
# 테스트 (빠른 모드; 일부 외부/느린 테스트 제외)
./run/run_test.ps1

# 전체 테스트
./run/run_test.ps1 -Full

# 파인튜닝 (단일)
./run/run_finetune_single.ps1 -Name convnext_s_cifar100
# 또는 YAML 경로 사용(레거시)
./run/run_finetune_single.ps1 -Config configs/finetune/convnext_s_cifar100.yaml

# 파인튜닝 (여러 개)
./run/run_finetune_all_teachers.ps1 -Names convnext_s_cifar100,convnext_l_cifar100

# 실험: SOTA 비교
./run/run_asib_sota_comparison.ps1 -Experiments sota_scenario_a

# 실험: Class Overlap
./run/run_asib_class_overlap.ps1 -Experiments overlap_100

# GPU 확인
./run/test_gpu_allocation.ps1
```

실행 정책 차단 시:
```powershell
powershell -ExecutionPolicy Bypass -File .\run\run_test.ps1
```

환경/GPU 지정은 모든 스크립트에 `-Env asib -GPU 0` 형태로 공통 지원합니다.

### Linux (SLURM)
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