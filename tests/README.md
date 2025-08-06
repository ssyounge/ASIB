# Tests Directory

이 폴더는 ASMB_KD 프로젝트의 모든 테스트 파일들을 포함합니다.

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

# run/run_test.sh 사용 (권장 - GPU 가속, 병렬 실행)
bash run/run_test.sh
```

## 테스트 파일 구조

### 핵심 기능 테스트
- `test_asib_cl.py` - ASIB-CL 모델 테스트
- `test_asib_step.py` - ASIB Step 기능 테스트
- `test_core.py` - 핵심 기능 테스트

### PyCIL 통합 테스트
- `test_pycil_integration.py` - PyCIL 통합 테스트
- `test_pycil_models.py` - PyCIL 모델들 테스트

### 데이터 및 유틸리티 테스트
- `test_data.py` - 데이터 로더 및 변환 테스트
- `test_utils.py` - 유틸리티 함수 테스트

### 모델 테스트
- `test_models.py` - 기본 모델 테스트
- `test_models_advanced.py` - 고급 모델 테스트
- `test_new_methods.py` - 새로운 방법들 테스트
- `test_new_students.py` - 새로운 학생 모델 테스트

### 설정 및 실험 테스트
- `test_configs.py` - 설정 파일 테스트
- `test_finetune_configs.py` - 파인튜닝 설정 테스트
- `test_cl_experiments.py` - CL 실험 테스트

### 스크립트 및 통합 테스트
- `test_scripts.py` - 스크립트 테스트
- `test_integration.py` - 통합 테스트
- `test_modules.py` - 모듈 테스트

### KD 및 특수 테스트
- `test_kd_methods.py` - 지식 증류 방법 테스트
- `test_disagreement.py` - 불일치 계산 테스트
- `test_convnext_s_teacher.py` - ConvNeXt-S 교사 테스트
- `test_ib_mbm_shapes.py` - IB MBM 형태 테스트
- `test_partial_freeze.py` - 부분 고정 테스트

### 설정 파일
- `conftest.py` - pytest 공통 설정 및 fixtures

## 테스트 결과 확인

### GPU 테스트 결과
```bash
# 실시간 로그 확인
tail -f experiments/logs/test_<JOBID>.log

# 요약 결과 확인
cat experiments/test_results/summary.log

# 개별 테스트 결과 확인
ls experiments/test_results/*.log
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
```

## 주의사항

- `conftest.py`는 pytest 설정 파일이므로 직접 실행하지 마세요
- GPU 테스트는 `run/run_test.sh`를 사용하는 것이 가장 효율적입니다
- 로컬 테스트는 개발 중 빠른 피드백용으로 사용하세요 