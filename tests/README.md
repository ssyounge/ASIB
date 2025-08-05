# ASMB_KD Tests

이 폴더는 ASMB_KD 프로젝트의 모든 테스트 파일들을 포함합니다.

## 📁 테스트 파일 구조

### 🔧 Core Tests
- **`test_core.py`** - 핵심 빌더 및 트레이너 함수들 테스트
  - `build_model()`, `create_teacher_by_name()`, `create_student_by_name()`
  - `create_optimizers_and_schedulers()`, `run_training_stages()`
  - `partial_freeze_teacher_auto()`, `partial_freeze_student_auto()`

### 🧠 Knowledge Distillation Methods
- **`test_kd_methods.py`** - 모든 KD 방법들 테스트
  - `ASIBDistiller` (주요 방법)
  - `VanillaKDDistiller`, `DKDDistiller`, `CRDDistiller`
  - `FitNetDistiller`, `ATDistiller`, `SimKDDistiller`
  - `ReviewKDDistiller`, `SSKDDistiller`, `ABDistiller`, `FTDistiller`

### 🏗️ Models
- **`test_models.py`** - 기본 모델 생성 및 동작 테스트
  - Teacher 모델들: ResNet152, ConvNeXt-L/S, EfficientNet-L2
  - Student 모델들: ResNet152/101/50, ShuffleNet-V2, MobileNet-V2, EfficientNet-B0
- **`test_models_advanced.py`** - 고급 모델 기능 테스트
  - `IB_MBM` (Information Bottleneck Manifold Bridging Module)
  - `SynergyHead`, `ChannelAdapter2D`, `BaseKDModel`
  - 멀티헤드 어텐션, 그래디언트 플로우, 통합 테스트

### 🔗 Integration Tests
- **`test_integration.py`** - End-to-End 통합 테스트
  - 완전한 파이프라인 테스트
  - 성능 및 메모리 사용량 테스트
  - 에러 핸들링 및 재현성 테스트

### 📊 Data & Configs
- **`test_data.py`** - 데이터 로더 및 데이터셋 테스트
  - CIFAR-100, ImageNet-32 데이터셋
  - 데이터 변환 및 검증
- **`test_configs.py`** - 설정 파일 검증
  - 실험, 파인튜닝, 메서드, 모델 설정들
  - YAML 설정 파일 구조 및 값 검증

### 🛠️ Modules & Utils
- **`test_modules.py`** - 모듈별 기능 테스트
  - 손실 함수들 (KL, MSE, IB, CE, Contrastive 등)
  - Partial Freeze, Student/Teacher Trainer
  - CutMix 파인튜닝
- **`test_utils.py`** - 유틸리티 함수들 테스트
  - 공통 유틸리티, 설정 유틸리티, 로깅 유틸리티
  - 훈련 메트릭, 프리즈 유틸리티

### 📝 Scripts & New Features
- **`test_scripts.py`** - 스크립트 기능 테스트
  - Sensitivity Analysis, Overlap Analysis
  - Fine-tuning, Baseline Training
- **`test_new_methods.py`** - 새로운 KD 방법들 테스트
- **`test_new_students.py`** - 새로운 Student 모델들 테스트

### 🔍 Specialized Tests
- **`test_asib_step.py`** - ASIB 방법의 forward/backward 테스트
- **`test_convnext_s_teacher.py`** - ConvNeXt-S Teacher 전용 테스트
- **`test_ib_mbm_shapes.py`** - IB_MBM 출력 형태 테스트
- **`test_disagreement.py`** - Teacher 간 불일치율 계산 테스트
- **`test_partial_freeze.py`** - Partial Freeze 기능 테스트
- **`test_finetune_configs.py`** - 파인튜닝 설정 테스트

## 🚀 테스트 실행 방법

### 1. 모든 테스트 실행
```bash
# 커스텀 러너 사용 (권장)
python tests/run_all_tests.py

# pytest 사용
python -m pytest tests/ -v
```

### 2. 특정 테스트 파일 실행
```bash
# 특정 파일만 실행
python -m pytest tests/test_core.py -v

# 특정 클래스만 실행
python -m pytest tests/test_core.py::TestCoreBuilder -v

# 특정 테스트만 실행
python -m pytest tests/test_core.py::TestCoreBuilder::test_build_model -v
```

### 3. 마커를 사용한 선택적 실행
```bash
# 빠른 테스트만 실행
python -m pytest tests/ -m "not slow" -v

# 통합 테스트만 실행
python -m pytest tests/ -m integration -v

# 단위 테스트만 실행
python -m pytest tests/ -m unit -v
```

## 📊 테스트 커버리지

### ✅ 완전히 테스트된 기능들
- **모든 KD 방법들** (11개 방법)
- **모든 Teacher/Student 모델들** (10+ 모델)
- **MBM 및 SynergyHead** (고급 기능 포함)
- **데이터 로더 및 변환**
- **설정 파일 검증**
- **통합 파이프라인**
- **성능 및 메모리 테스트**

### 🎯 테스트 결과
- **총 테스트 수**: 257개
- **테스트 파일 수**: 18개
- **커버리지**: 핵심 기능 100% 커버

## 🔧 환경 설정

테스트를 실행하기 전에 필요한 환경:

```bash
# Python 3.12+ 환경
# 필수 패키지들
pip install pytest torch torchvision torchaudio
pip install hydra-core omegaconf timm pandas
pip install matplotlib seaborn wandb
```

## 📝 테스트 작성 가이드라인

새로운 테스트를 추가할 때:

1. **파일명**: `test_<module_name>.py`
2. **클래스명**: `Test<ModuleName>`
3. **함수명**: `test_<function_name>`
4. **마커 사용**: `@pytest.mark.slow`, `@pytest.mark.integration`

## 🐛 문제 해결

테스트가 실패할 경우:

1. **환경 확인**: Python 버전, 패키지 버전
2. **의존성 확인**: 필요한 모델 가중치 다운로드
3. **GPU 메모리**: 큰 모델 테스트 시 GPU 메모리 부족 가능성
4. **로그 확인**: 상세한 에러 메시지 확인

## 📈 성능 모니터링

- **테스트 실행 시간**: 전체 약 70초
- **메모리 사용량**: 최대 4GB (GPU 포함)
- **CPU 사용량**: 멀티코어 활용

---

**참고**: 이 테스트 스위트는 ASMB_KD 프로젝트의 모든 핵심 기능을 검증하며, 코드 변경 시 반드시 실행해야 합니다. 