# PyCIL & ASIB-CL 테스트 가이드

## 📖 개요

이 문서는 PyCIL 프레임워크 통합과 ASIB-CL 모델 구현에 대한 포괄적인 테스트 가이드입니다. 단위 테스트부터 통합 테스트까지 모든 테스트 시나리오를 다룹니다.

## 🎯 테스트 목표

- **PyCIL 통합 검증**: 프레임워크가 올바르게 통합되었는지 확인
- **ASIB-CL 모델 검증**: 모델이 정상적으로 작동하는지 확인
- **실험 환경 검증**: 전체 실험 파이프라인이 정상 작동하는지 확인
- **성능 검증**: 예상된 성능이 나오는지 확인

## 📁 테스트 파일 구조

```
tests/
├── test_pycil_integration.py      # PyCIL 통합 테스트
├── test_asib_cl.py                # ASIB-CL 모델 테스트
├── test_cl_experiments.py         # CL 실험 테스트
├── test_data_loading.py           # 데이터 로딩 테스트
├── test_config_validation.py      # 설정 파일 검증 테스트
└── conftest.py                    # pytest 설정 및 공통 fixture
```

## 🧪 테스트 실행 방법

### 1. 전체 테스트 실행
```bash
# 모든 테스트 실행
pytest tests/ -v

# 특정 테스트 파일 실행
pytest tests/test_asib_cl.py -v

# 특정 테스트 함수 실행
pytest tests/test_asib_cl.py::test_asib_cl_initialization -v
```

### 2. 커버리지와 함께 실행
```bash
pytest tests/ --cov=PyCIL --cov-report=html
```

### 3. 병렬 실행
```bash
pytest tests/ -n auto
```

## 🔧 테스트 환경 설정

### 1. pytest 설정 (conftest.py)
```python
import pytest
import torch
import numpy as np
import sys
import os

# PyCIL 경로 추가
sys.path.append('./PyCIL')

@pytest.fixture(scope="session")
def device():
    """테스트용 디바이스 설정"""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

@pytest.fixture(scope="session")
def sample_args():
    """테스트용 기본 설정"""
    return {
        "prefix": "test",
        "dataset": "cifar100",
        "memory_size": 100,
        "memory_per_class": 10,
        "fixed_memory": False,
        "shuffle": True,
        "init_cls": 5,
        "increment": 5,
        "model_name": "asib_cl",
        "convnet_type": "resnet32",
        "device": ["0"],
        "seed": [1993],
        "ib_beta": 0.1
    }

@pytest.fixture(scope="session")
def small_dataset():
    """작은 테스트 데이터셋"""
    # 실제 데이터 대신 더미 데이터 사용
    return torch.randn(100, 3, 32, 32), torch.randint(0, 10, (100,))
```

## 📋 테스트 케이스 상세

### 1. PyCIL 통합 테스트 (test_pycil_integration.py)

#### 기본 통합 검증
```python
def test_pycil_import():
    """PyCIL 모듈들이 정상적으로 import되는지 확인"""
    try:
        from PyCIL.models.base import BaseLearner
        from PyCIL.utils.factory import get_model
        from PyCIL.utils.data_manager import DataManager
        assert True
    except ImportError as e:
        pytest.fail(f"PyCIL import 실패: {e}")

def test_factory_registration():
    """ASIB-CL이 factory에 정상적으로 등록되었는지 확인"""
    from PyCIL.utils.factory import get_model
    from PyCIL.models.asib_cl import ASIB_CL
    
    args = sample_args()
    model = get_model("asib_cl", args)
    assert isinstance(model, ASIB_CL)
```

#### 설정 파일 검증
```python
def test_config_file_exists():
    """ASIB-CL 설정 파일이 존재하는지 확인"""
    config_path = "PyCIL/exps/asib_cl.json"
    assert os.path.exists(config_path), f"설정 파일이 없습니다: {config_path}"

def test_config_file_valid():
    """설정 파일이 유효한 JSON인지 확인"""
    import json
    config_path = "PyCIL/exps/asib_cl.json"
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    required_keys = ["model_name", "convnet_type", "dataset", "ib_beta"]
    for key in required_keys:
        assert key in config, f"필수 키가 없습니다: {key}"
    
    assert config["model_name"] == "asib_cl"
    assert config["ib_beta"] == 0.1
```

### 2. ASIB-CL 모델 테스트 (test_asib_cl.py)

#### 모델 초기화 테스트
```python
def test_asib_cl_initialization(sample_args):
    """ASIB-CL 모델이 정상적으로 초기화되는지 확인"""
    from PyCIL.models.asib_cl import ASIB_CL
    
    model = ASIB_CL(sample_args)
    
    # 기본 속성 확인
    assert hasattr(model, '_network')
    assert hasattr(model, '_old_network')
    assert hasattr(model, '_ib_beta')
    assert hasattr(model, '_ib_encoder')
    assert hasattr(model, '_ib_decoder')
    
    # 초기값 확인
    assert model._ib_beta == 0.1
    assert model._old_network is None
    assert model._ib_encoder is None
    assert model._ib_decoder is None

def test_ib_modules_initialization(sample_args):
    """IB 모듈이 정상적으로 초기화되는지 확인"""
    from PyCIL.models.asib_cl import ASIB_CL
    
    model = ASIB_CL(sample_args)
    
    # 더미 네트워크 생성
    class DummyNetwork:
        def __init__(self):
            self.feature_dim = 512
    
    model._old_network = DummyNetwork()
    model._init_ib_modules(512)
    
    # IB 모듈 구조 확인
    assert model._ib_encoder is not None
    assert model._ib_decoder is not None
    
    # 입력 차원 확인
    dummy_input = torch.randn(10, 512)
    encoder_output = model._ib_encoder(dummy_input)
    assert encoder_output.shape == (10, 256)  # latent_dim * 2
    
    # 디코더 테스트
    latent_dim = 512 // 4  # 128
    dummy_latent = torch.randn(10, latent_dim)
    decoder_output = model._ib_decoder(dummy_latent)
    assert decoder_output.shape == (10, 512)
```

#### IB 손실 함수 테스트
```python
def test_ib_distillation_loss(sample_args):
    """IB 기반 지식 증류 손실이 정상적으로 계산되는지 확인"""
    from PyCIL.models.asib_cl import ASIB_CL
    
    model = ASIB_CL(sample_args)
    
    # 더미 데이터 생성
    batch_size = 16
    feature_dim = 512
    student_features = torch.randn(batch_size, feature_dim)
    teacher_features = torch.randn(batch_size, feature_dim)
    
    # IB 모듈 초기화
    model._old_network = type('DummyNetwork', (), {'feature_dim': feature_dim})()
    model._init_ib_modules(feature_dim)
    
    # 손실 계산
    loss = model._ib_distillation_loss(student_features, teacher_features)
    
    # 손실이 스칼라이고 양수인지 확인
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # 스칼라
    assert loss.item() > 0  # 양수

def test_reparameterization_trick(sample_args):
    """Reparameterization trick이 정상적으로 작동하는지 확인"""
    from PyCIL.models.asib_cl import ASIB_CL
    
    model = ASIB_CL(sample_args)
    
    # 더미 mu, logvar 생성
    batch_size = 16
    latent_dim = 128
    mu = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim)
    
    # Reparameterization
    z = model._reparameterize(mu, logvar)
    
    # 출력 형태 확인
    assert z.shape == (batch_size, latent_dim)
    assert isinstance(z, torch.Tensor)
```

#### 메모리 관리 테스트
```python
def test_memory_management(sample_args):
    """메모리 관리 기능이 정상적으로 작동하는지 확인"""
    from PyCIL.models.asib_cl import ASIB_CL
    
    model = ASIB_CL(sample_args)
    
    # 초기 메모리 상태 확인
    assert len(model._data_memory) == 0
    assert len(model._targets_memory) == 0
    
    # 더미 메모리 데이터 추가
    dummy_data = np.random.randn(50, 3, 32, 32)
    dummy_targets = np.random.randint(0, 10, 50)
    
    model._data_memory = dummy_data
    model._targets_memory = dummy_targets
    
    # 메모리 반환 테스트
    memory = model._get_memory()
    assert memory is not None
    assert len(memory) == 2
    assert memory[0].shape == dummy_data.shape
    assert memory[1].shape == dummy_targets.shape
```

### 3. CL 실험 테스트 (test_cl_experiments.py)

#### 실험 설정 검증
```python
def test_experiment_configs():
    """모든 실험 설정 파일이 유효한지 확인"""
    import json
    import glob
    
    config_files = glob.glob("PyCIL/exps/*.json")
    
    for config_file in config_files:
        with open(config_file, 'r') as f:
            try:
                config = json.load(f)
                # 기본 필수 키 확인
                required_keys = ["model_name", "convnet_type", "dataset"]
                for key in required_keys:
                    assert key in config, f"{config_file}: {key} 키가 없습니다"
            except json.JSONDecodeError as e:
                pytest.fail(f"{config_file}: JSON 파싱 실패 - {e}")

def test_asib_cl_config_specific():
    """ASIB-CL 설정의 특정 값들이 올바른지 확인"""
    import json
    
    with open("PyCIL/exps/asib_cl.json", 'r') as f:
        config = json.load(f)
    
    # ASIB-CL 특화 설정 확인
    assert config["model_name"] == "asib_cl"
    assert "ib_beta" in config
    assert isinstance(config["ib_beta"], (int, float))
    assert 0 < config["ib_beta"] < 1
```

#### 데이터 로딩 테스트
```python
def test_data_manager_initialization(sample_args):
    """DataManager가 정상적으로 초기화되는지 확인"""
    from PyCIL.utils.data_manager import DataManager
    
    data_manager = DataManager(
        dataset=sample_args["dataset"],
        shuffle=sample_args["shuffle"],
        seed=sample_args["seed"][0],
        init_cls=sample_args["init_cls"],
        increment=sample_args["increment"]
    )
    
    assert data_manager.nb_tasks > 0
    assert data_manager.total_classes == 100  # CIFAR-100

def test_dataset_loading(sample_args):
    """데이터셋이 정상적으로 로드되는지 확인"""
    from PyCIL.utils.data_manager import DataManager
    
    data_manager = DataManager(
        dataset=sample_args["dataset"],
        shuffle=sample_args["shuffle"],
        seed=sample_args["seed"][0],
        init_cls=sample_args["init_cls"],
        increment=sample_args["increment"]
    )
    
    # 첫 번째 태스크 데이터 로드
    train_dataset = data_manager.get_dataset(
        np.arange(0, sample_args["init_cls"]),
        source="train",
        mode="train"
    )
    
    assert len(train_dataset) > 0
    
    # 데이터 샘플 확인
    sample_data, sample_target = train_dataset[0]
    assert sample_data.shape == (3, 32, 32)  # CIFAR-100 이미지 크기
    assert 0 <= sample_target < sample_args["init_cls"]
```

### 4. 성능 테스트 (test_performance.py)

#### 메모리 사용량 테스트
```python
def test_memory_usage(sample_args):
    """모델의 메모리 사용량이 합리적인지 확인"""
    import psutil
    import torch
    
    from PyCIL.models.asib_cl import ASIB_CL
    
    # 메모리 사용량 측정 시작
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # 모델 생성
    model = ASIB_CL(sample_args)
    
    # 더미 데이터로 forward pass
    dummy_input = torch.randn(32, 3, 32, 32)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
        model._network = model._network.cuda()
    
    with torch.no_grad():
        _ = model._network(dummy_input)
    
    # 메모리 사용량 측정
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # 메모리 증가량이 합리적인지 확인 (예: 2GB 이하)
    assert memory_increase < 2048, f"메모리 사용량이 너무 큽니다: {memory_increase:.2f}MB"
```

#### 학습 속도 테스트
```python
def test_training_speed(sample_args):
    """학습 속도가 합리적인지 확인"""
    import time
    from PyCIL.models.asib_cl import ASIB_CL
    
    model = ASIB_CL(sample_args)
    
    # 더미 데이터 생성
    batch_size = 32
    dummy_input = torch.randn(batch_size, 3, 32, 32)
    dummy_target = torch.randint(0, 10, (batch_size,))
    
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
        dummy_target = dummy_target.cuda()
        model._network = model._network.cuda()
    
    # 학습 시간 측정
    start_time = time.time()
    
    model._network.train()
    optimizer = torch.optim.SGD(model._network.parameters(), lr=0.01)
    
    for _ in range(10):  # 10번의 forward/backward pass
        optimizer.zero_grad()
        outputs = model._network(dummy_input)
        if isinstance(outputs, dict):
            outputs = outputs['logits']
        loss = torch.nn.functional.cross_entropy(outputs, dummy_target)
        loss.backward()
        optimizer.step()
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # 학습 시간이 합리적인지 확인 (예: 10초 이하)
    assert training_time < 10, f"학습 시간이 너무 깁니다: {training_time:.2f}초"
```

## 🔍 디버깅 테스트

### 1. 오류 상황 테스트
```python
def test_error_handling():
    """오류 상황에서 적절히 처리되는지 확인"""
    from PyCIL.models.asib_cl import ASIB_CL
    
    # 잘못된 설정으로 모델 생성 시도
    invalid_args = {"invalid_key": "invalid_value"}
    
    with pytest.raises(Exception):
        model = ASIB_CL(invalid_args)

def test_edge_cases():
    """엣지 케이스에서 정상적으로 작동하는지 확인"""
    from PyCIL.models.asib_cl import ASIB_CL
    
    # 빈 배치 테스트
    sample_args = {
        "prefix": "test",
        "dataset": "cifar100",
        "memory_size": 0,  # 빈 메모리
        "memory_per_class": 0,
        "fixed_memory": False,
        "shuffle": True,
        "init_cls": 1,  # 최소 클래스 수
        "increment": 1,
        "model_name": "asib_cl",
        "convnet_type": "resnet32",
        "device": ["0"],
        "seed": [1993],
        "ib_beta": 0.1
    }
    
    model = ASIB_CL(sample_args)
    assert model is not None
```

### 2. 로깅 테스트
```python
def test_logging_functionality():
    """로깅 기능이 정상적으로 작동하는지 확인"""
    import logging
    from PyCIL.models.asib_cl import ASIB_CL
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 로그 메시지 캡처
    log_messages = []
    
    class TestHandler(logging.Handler):
        def emit(self, record):
            log_messages.append(record.getMessage())
    
    logger.addHandler(TestHandler())
    
    # 모델 초기화 (로깅 발생)
    sample_args = {
        "prefix": "test",
        "dataset": "cifar100",
        "memory_size": 100,
        "memory_per_class": 10,
        "fixed_memory": False,
        "shuffle": True,
        "init_cls": 5,
        "increment": 5,
        "model_name": "asib_cl",
        "convnet_type": "resnet32",
        "device": ["0"],
        "seed": [1993],
        "ib_beta": 0.1
    }
    
    model = ASIB_CL(sample_args)
    
    # 로그 메시지가 생성되었는지 확인
    assert len(log_messages) > 0
```

## 📊 테스트 결과 분석

### 1. 테스트 커버리지
```bash
# 커버리지 리포트 생성
pytest tests/ --cov=PyCIL --cov-report=html --cov-report=term-missing
```

### 2. 성능 벤치마크
```python
def benchmark_performance():
    """성능 벤치마크 실행"""
    import time
    import torch
    
    # 다양한 배치 크기로 테스트
    batch_sizes = [16, 32, 64, 128]
    results = {}
    
    for batch_size in batch_sizes:
        start_time = time.time()
        
        # 모델 생성 및 forward pass
        dummy_input = torch.randn(batch_size, 3, 32, 32)
        # ... 테스트 코드 ...
        
        end_time = time.time()
        results[batch_size] = end_time - start_time
    
    return results
```

## 🚀 CI/CD 통합

### 1. GitHub Actions 설정
```yaml
# .github/workflows/test.yml
name: PyCIL & ASIB-CL Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install torch torchvision
        pip install pytest pytest-cov
        pip install scipy quadprog POT
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=PyCIL --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

### 2. 테스트 자동화 스크립트
```bash
#!/bin/bash
# run_tests.sh

echo "🧪 PyCIL & ASIB-CL 테스트 시작"

# 1. 기본 테스트
echo "📋 기본 테스트 실행..."
pytest tests/ -v

# 2. 커버리지 테스트
echo "📊 커버리지 테스트 실행..."
pytest tests/ --cov=PyCIL --cov-report=html

# 3. 성능 테스트
echo "⚡ 성능 테스트 실행..."
python -m pytest tests/test_performance.py -v

# 4. 결과 요약
echo "📈 테스트 결과 요약..."
echo "커버리지 리포트: htmlcov/index.html"
echo "테스트 완료!"
```

## 🔍 문제 해결

### 1. 일반적인 테스트 실패

#### Import 오류
```bash
# PyCIL 경로 문제
export PYTHONPATH="${PYTHONPATH}:./PyCIL"
```

#### CUDA 메모리 부족
```python
# CPU에서 테스트 실행
@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")
```

#### 데이터셋 다운로드 실패
```python
# 더미 데이터 사용
@pytest.fixture(scope="session")
def mock_dataset():
    return torch.randn(100, 3, 32, 32), torch.randint(0, 10, (100,))
```

### 2. 디버깅 팁

#### 상세한 로그 출력
```bash
pytest tests/ -v -s --tb=long
```

#### 특정 테스트만 실행
```bash
pytest tests/test_asib_cl.py::test_ib_distillation_loss -v -s
```

#### 메모리 프로파일링
```bash
pip install memory-profiler
python -m memory_profiler tests/test_performance.py
```

## 📚 참고 자료

1. **pytest 공식 문서**: https://docs.pytest.org/
2. **PyTorch 테스트 가이드**: https://pytorch.org/docs/stable/testing.html
3. **Python 테스트 모범 사례**: https://realpython.com/python-testing/

## 🤝 기여 가이드

테스트 개선 제안이나 새로운 테스트 케이스 추가는 이슈를 생성해 주세요.

---

**마지막 업데이트**: 2025-08-05
**버전**: 1.0 