# Tests Directory

이 폴더는 ASMB_KD 프로젝트의 모든 테스트 파일들을 포함합니다.

## 📊 **Test Suite Overview**

- **Total Test Files**: 42 files
- **Test Categories**: 12 categories
- **Coverage**: All major components
- **Execution**: Parallel GPU-accelerated testing

## 테스트 실행 방법

### 1. GPU 가속 테스트 (권장)
```bash
# 모든 테스트를 GPU에서 병렬로 실행
sbatch run/run_test.sh
```

### 2. 로컬 테스트
```bash
# 모든 테스트 실행
python -m pytest tests/ -v

# 특정 테스트 파일 실행
python -m pytest tests/test_asib_cl.py -v

# 특정 테스트 함수 실행
python -m pytest tests/test_asib_cl.py::test_asib_cl_initialization -v
```

### 3. Python 스크립트로 실행
```bash
# run/run_test.sh 사용 (권장 - GPU 가속, 병렬 실행)
bash run/run_test.sh
```

## 테스트 파일 구조

### 🔥 **Core ASIB Tests** (2 files)
- `test_asib_cl.py` - ASIB-CL 모델 테스트
- `test_asib_step.py` - ASIB Step 기능 테스트

### 🔗 **PyCIL Integration Tests** (2 files)
- `test_pycil_integration.py` - PyCIL 통합 테스트
- `test_pycil_models.py` - PyCIL 모델들 테스트

### 📊 **Data & Utils Tests** (7 files)
- `test_data.py` - 데이터 로더 및 변환 테스트
- `test_utils.py` - 유틸리티 함수 테스트
- `test_core.py` - 핵심 기능 테스트
- `test_dataset_attributes.py` - 데이터셋 속성 테스트
- `test_dataset_fix.py` - 데이터셋 수정 테스트
- `test_overlap_dataset.py` - 오버랩 데이터셋 테스트
- `test_main_dataset_loading.py` - 메인 데이터 로딩 테스트

### 🤖 **Model Tests** (4 files)
- `test_models.py` - 기본 모델 테스트
- `test_models_advanced.py` - 고급 모델 테스트
- `test_new_methods.py` - 새로운 방법들 테스트
- `test_new_students.py` - 새로운 학생 모델 테스트

### ⚙️ **Config & Experiment Tests** (5 files)
- `test_configs.py` - 설정 파일 테스트
- `test_finetune_configs.py` - 파인튜닝 설정 테스트
- `test_cl_experiments.py` - CL 실험 테스트
- `test_experiment_configs.py` - 실험 설정 테스트
- `test_registry_comprehensive.py` - 레지스트리 종합 테스트

### 🔧 **Script & Integration Tests** (3 files)
- `test_scripts.py` - 스크립트 테스트
- `test_integration.py` - 통합 테스트
- `test_modules.py` - 모듈 테스트

### 🧠 **KD & Special Tests** (5 files)
- `test_kd_methods.py` - 지식 증류 방법 테스트
- `test_disagreement.py` - 불일치 계산 테스트
- `test_ib_mbm_shapes.py` - IB MBM 형태 테스트
- `test_partial_freeze.py` - 부분 고정 테스트
- `test_mbm_tensor_shapes.py` - MBM 텐서 형태 테스트

### 🛡️ **Framework Robustness Tests** (3 files)
- `test_framework_robustness.py` - 프레임워크 견고성 테스트
- `test_error_prevention.py` - 오류 방지 테스트
- `test_final_validation.py` - 최종 검증 테스트

### 🚀 **Experiment Execution Tests** (3 files)
- `test_experiment_execution.py` - 실험 실행 테스트
- `test_training_pipeline.py` - 훈련 파이프라인 테스트
- `test_main_py_integration.py` - 메인 파이썬 통합 테스트

### 🛠️ **Utility Function Tests** (3 files)
- `test_auto_set_mbm_query_dim.py` - MBM 쿼리 차원 자동 설정 테스트
- `test_renorm_ce_kd.py` - 재정규화 CE KD 테스트
- `test_setup_partial_freeze_schedule.py` - 부분 고정 스케줄 설정 테스트

### 🎯 **Main Integration Tests** (4 files)
- `test_main.py` - 메인 모듈 테스트
- `test_main_step_by_step.py` - 단계별 메인 테스트
- `test_main_training.py` - 메인 훈련 테스트
- `test_training_simple.py` - 간단한 훈련 테스트

### 🔍 **Dataset Problem Tests** (1 file)
- `test_actual_dataset_problem.py` - 실제 데이터셋 문제 테스트

### 📋 **Configuration Files**
- `conftest.py` - pytest 공통 설정 및 fixtures (42개 fixture 제공)

## 테스트 결과 확인

### GPU 테스트 결과
```bash
# 실시간 로그 확인
tail -f experiments/logs/test_<JOBID>.log

# 요약 결과 확인
cat experiments/test_results/summary.log

# 개별 테스트 결과 확인
ls experiments/test_results/*.log

# 테스트 그룹별 결과
cat experiments/test_results/core_asib_test.log      # Core ASIB Tests
cat experiments/test_results/pycil_test.log          # PyCIL Tests
cat experiments/test_results/data_utils_test.log     # Data & Utils Tests
cat experiments/test_results/models_test.log         # Model Tests
cat experiments/test_results/configs_test.log        # Config & Experiment Tests
cat experiments/test_results/scripts_test.log        # Script & Integration Tests
cat experiments/test_results/kd_test.log             # KD & Special Tests
cat experiments/test_results/robustness_test.log     # Framework Robustness Tests
cat experiments/test_results/execution_test.log      # Experiment Execution Tests
cat experiments/test_results/utility_test.log        # Utility Function Tests
cat experiments/test_results/main_integration_test.log # Main Integration Tests
cat experiments/test_results/dataset_problem_test.log # Dataset Problem Tests
```

### 로컬 테스트 결과
```bash
# 상세 결과 확인
python -m pytest tests/ -v --tb=long

# HTML 리포트 생성
python -m pytest tests/ --html=test_report.html
```

## 테스트 작성 가이드

### 새로운 테스트 추가
1. `test_<module_name>.py` 형식으로 파일명 지정
2. `conftest.py`의 fixtures 활용
3. GPU/CPU 호환성 고려
4. 적절한 assertion 사용

### Fixtures 활용
```python
def test_example(device, sample_args, dummy_network):
    # conftest.py에서 제공하는 fixtures 사용
    pass

def test_main_integration(main_config, training_config):
    # main.py 테스트용 설정 사용
    pass
```

### Available Fixtures
- `device`: 테스트용 디바이스 (CUDA/CPU)
- `sample_args`: 기본 테스트 설정
- `test_config`: 테스트용 설정
- `dummy_network`: 더미 네트워크 클래스
- `temp_config_file`: 임시 설정 파일
- `registry_configs`: 레지스트리 설정
- `registry_validation`: 레지스트리 검증 함수
- `main_config`: main.py 테스트용 설정
- `training_config`: 훈련 테스트용 설정

## 주의사항

- `conftest.py`는 pytest 설정 파일이므로 직접 실행하지 마세요
- GPU 테스트는 `run/run_test.sh`를 사용하는 것이 가장 효율적입니다
- 로컬 테스트는 개발 중 빠른 피드백용으로 사용하세요
- 모든 42개 테스트 파일이 `run/run_test.sh`에서 병렬로 실행됩니다
- 테스트 결과는 `experiments/test_results/` 폴더에 저장됩니다

## 🎯 **Test Coverage Summary**

| Category | Files | Description |
|----------|-------|-------------|
| 🔥 Core ASIB | 2 | 핵심 ASIB 기능 테스트 |
| 🔗 PyCIL Integration | 2 | PyCIL 통합 테스트 |
| 📊 Data & Utils | 7 | 데이터 및 유틸리티 테스트 |
| 🤖 Models | 4 | 모델 관련 테스트 |
| ⚙️ Config & Experiments | 5 | 설정 및 실험 테스트 |
| 🔧 Script & Integration | 3 | 스크립트 및 통합 테스트 |
| 🧠 KD & Special | 5 | 지식 증류 및 특수 테스트 |
| 🛡️ Framework Robustness | 3 | 프레임워크 견고성 테스트 |
| 🚀 Experiment Execution | 3 | 실험 실행 테스트 |
| 🛠️ Utility Functions | 3 | 유틸리티 함수 테스트 |
| 🎯 Main Integration | 4 | 메인 통합 테스트 |
| 🔍 Dataset Problems | 1 | 데이터셋 문제 테스트 |
| **Total** | **42** | **모든 컴포넌트 커버** | 