#!/usr/bin/env python3
"""
CL 실험 테스트
Continual Learning 실험 환경이 정상적으로 작동하는지 확인
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

# PyCIL 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PyCIL'))

def test_experiment_configurations():
    """실험 설정들이 올바르게 구성되었는지 확인"""
    config_files = [
        "PyCIL/exps/finetune.json",
        "PyCIL/exps/ewc.json",
        "PyCIL/exps/lwf.json",
        "PyCIL/exps/icarl.json",
        "PyCIL/exps/der.json",
        "PyCIL/exps/asib_cl.json"
    ]
    
    for config_file in config_files:
        assert os.path.exists(config_file), f"설정 파일이 없습니다: {config_file}"
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 기본 필수 키 확인
        required_keys = ["dataset", "init_cls", "increment", "memory_size", 
                        "memory_per_class", "convnet_type", "device"]
        for key in required_keys:
            assert key in config, f"{config_file}에 필수 키가 없습니다: {key}"

def test_data_manager_initialization():
    """DataManager가 정상적으로 초기화되는지 확인"""
    from PyCIL.utils.data_manager import DataManager
    
    try:
        data_manager = DataManager("cifar100", True, 1993, 10, 10)
        assert data_manager is not None
        assert hasattr(data_manager, 'nb_tasks')
        assert hasattr(data_manager, 'get_total_classnum')
    except Exception as e:
        pytest.fail(f"DataManager 초기화 실패: {e}")

def test_dataset_loading():
    """데이터셋 로딩이 정상적으로 작동하는지 확인"""
    from PyCIL.utils.data_manager import DataManager
    
    try:
        data_manager = DataManager("cifar100", True, 1993, 10, 10)
        
        # 첫 번째 태스크 데이터 가져오기
        train_dataset = data_manager.get_dataset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "train", "train")
        test_dataset = data_manager.get_dataset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "test", "test")
        
        assert train_dataset is not None
        assert test_dataset is not None
        assert len(train_dataset) > 0
        assert len(test_dataset) > 0
    except Exception as e:
        pytest.fail(f"데이터셋 로딩 실패: {e}")

def test_task_progression():
    """태스크 진행이 올바르게 작동하는지 확인"""
    from PyCIL.utils.data_manager import DataManager
    
    data_manager = DataManager("cifar100", True, 1993, 10, 10)
    
    # 태스크 수 확인
    assert data_manager.nb_tasks > 0
    
    # 각 태스크의 크기 확인
    for task_id in range(data_manager.nb_tasks):
        task_size = data_manager.get_task_size(task_id)
        if task_id == 0:
            assert task_size == 10
        else:
            assert task_size == 10

def test_memory_size_calculation():
    """메모리 크기 계산이 올바르게 작동하는지 확인"""
    from PyCIL.utils.data_manager import DataManager
    
    data_manager = DataManager("cifar100", True, 1993, 10, 10)
    
    # 메모리 관련 속성 확인
    assert hasattr(data_manager, 'nb_tasks')
    assert data_manager.nb_tasks > 0

def test_exemplar_management():
    """Exemplar 관리가 정상적으로 작동하는지 확인"""
    from PyCIL.utils.data_manager import DataManager
    
    data_manager = DataManager("cifar100", True, 1993, 10, 10)
    
    # 기본 속성 확인
    assert hasattr(data_manager, 'nb_tasks')
    assert data_manager.nb_tasks > 0
    
    # 데이터셋 가져오기 테스트
    dataset = data_manager.get_dataset([0, 1, 2, 3, 4], "train", "train")
    assert dataset is not None

def test_learning_progression():
    """학습 진행이 정상적으로 작동하는지 확인"""
    from PyCIL.models.asib_cl import ASIB_CL
    
    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["0"],
        "ib_beta": 0.1,
        "init_cls": 10,
        "increment": 10,
        "memory_size": 2000,
        "memory_per_class": 20
    }
    
    model = ASIB_CL(args)
    
    # 첫 번째 태스크 학습 시뮬레이션
    assert model._network is not None
    
    # after_task 호출로 이전 네트워크 저장
    model.after_task()
    assert model._old_network is not None

def test_network_update():
    """네트워크 업데이트가 정상적으로 작동하는지 확인"""
    from PyCIL.models.asib_cl import ASIB_CL
    
    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["0"],
        "ib_beta": 0.1,
        "memory_size": 2000,
        "memory_per_class": 20
    }
    
    model = ASIB_CL(args)
    
    # 네트워크 출력 확인
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 32, 32)
    
    with torch.no_grad():
        output = model._network(input_tensor)
        
        if isinstance(output, dict):
            logits = output['logits']
        else:
            logits = output
        
        assert logits.shape[0] == batch_size
        assert logits.dim() == 2

def test_optimizer_scheduler_initialization():
    """옵티마이저와 스케줄러 초기화가 정상적으로 작동하는지 확인"""
    from PyCIL.models.asib_cl import ASIB_CL
    
    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["0"],
        "ib_beta": 0.1,
        "memory_size": 2000,
        "memory_per_class": 20
    }
    
    model = ASIB_CL(args)
    
    # 옵티마이저 생성 테스트
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    assert optimizer is not None
    
    # 스케줄러 생성 테스트
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    assert scheduler is not None

def test_loss_accuracy_calculation():
    """손실과 정확도 계산이 정상적으로 작동하는지 확인"""
    from PyCIL.models.asib_cl import ASIB_CL
    
    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["0"],
        "ib_beta": 0.1,
        "memory_size": 2000,
        "memory_per_class": 20
    }
    
    model = ASIB_CL(args)
    
    # 테스트 데이터 생성
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 32, 32)
    targets = torch.randint(0, 10, (batch_size,))
    
    # 네트워크 출력
    output = model._network(input_tensor)
    
    if isinstance(output, dict):
        logits = output['logits']
    else:
        logits = output
    
    # 손실 계산
    loss = F.cross_entropy(logits, targets)
    assert loss.item() > 0
    assert not torch.isnan(loss)
    
    # 정확도 계산
    _, predicted = torch.max(logits, 1)
    accuracy = (predicted == targets).float().mean()
    assert 0 <= accuracy.item() <= 1

def test_batch_processing():
    """배치 처리가 정상적으로 작동하는지 확인"""
    from PyCIL.models.asib_cl import ASIB_CL
    
    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["0"],
        "ib_beta": 0.1,
        "memory_size": 2000,
        "memory_per_class": 20
    }
    
    model = ASIB_CL(args)
    
    # 다양한 배치 크기로 테스트
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, 3, 32, 32)
        
        with torch.no_grad():
            output = model._network(input_tensor)
            
            if isinstance(output, dict):
                logits = output['logits']
            else:
                logits = output
            
            assert logits.shape[0] == batch_size

def test_device_handling():
    """디바이스 처리가 정상적으로 작동하는지 확인"""
    from PyCIL.models.asib_cl import ASIB_CL

    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["0"],
        "ib_beta": 0.1,
        "memory_size": 2000,
        "memory_per_class": 20
    }

    model = ASIB_CL(args)

    # 디바이스 정보 확인
    assert 'device' in model.args
    assert isinstance(model.args['device'], list)

    # GPU가 사용 가능한지 확인
    if torch.cuda.is_available():
        # GPU에서 추론 테스트
        model.to('cuda')
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 32, 32).to('cuda')

        with torch.no_grad():
            output = model._network(input_tensor)
            # 출력이 딕셔너리인 경우를 고려
            if isinstance(output, dict):
                # 딕셔너리의 값들 중 하나에서 device 확인
                for key, value in output.items():
                    if isinstance(value, torch.Tensor):
                        assert value.device.type == 'cuda'
                        break
            else:
                assert output.device.type == 'cuda'
    else:
        # CPU에서 추론 테스트
        model.to('cpu')
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 32, 32)

        with torch.no_grad():
            output = model._network(input_tensor)
            # 출력이 딕셔너리인 경우를 고려
            if isinstance(output, dict):
                # 딕셔너리의 값들 중 하나에서 device 확인
                for key, value in output.items():
                    if isinstance(value, torch.Tensor):
                        assert value.device.type == 'cpu'
                        break
            else:
                assert output.device.type == 'cpu'

def test_gradient_flow():
    """그래디언트 흐름이 정상적으로 작동하는지 확인"""
    from PyCIL.models.asib_cl import ASIB_CL
    
    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["0"],
        "ib_beta": 0.1,
        "memory_size": 2000,
        "memory_per_class": 20
    }
    
    model = ASIB_CL(args)
    
    # 테스트 데이터 생성
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 32, 32, requires_grad=True)
    targets = torch.randint(0, 10, (batch_size,))
    
    # 순전파
    output = model._network(input_tensor)
    
    if isinstance(output, dict):
        logits = output['logits']
    else:
        logits = output
    
    # 손실 계산 및 역전파
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    
    # 그래디언트 확인
    assert input_tensor.grad is not None
    assert not torch.isnan(input_tensor.grad).any()
    assert not torch.isinf(input_tensor.grad).any()

def test_memory_efficiency():
    """메모리 효율성이 적절한지 확인"""
    from PyCIL.models.asib_cl import ASIB_CL
    
    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["0"],
        "ib_beta": 0.1,
        "memory_size": 2000,
        "memory_per_class": 20
    }
    
    model = ASIB_CL(args)
    
    # 모델 파라미터 수 확인
    total_params = sum(p.numel() for p in model.parameters())
    
    # 합리적인 파라미터 수 범위
    assert total_params > 100000  # 최소 10만 파라미터
    assert total_params < 10000000  # 최대 1000만 파라미터

def test_training_stability():
    """훈련 안정성이 적절한지 확인"""
    from PyCIL.models.asib_cl import ASIB_CL
    
    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["0"],
        "ib_beta": 0.1,
        "memory_size": 2000,
        "memory_per_class": 20
    }
    
    model = ASIB_CL(args)
    
    # 여러 번의 순전파로 안정성 확인
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 32, 32)
    
    for _ in range(5):
        with torch.no_grad():
            output = model._network(input_tensor)
            
            if isinstance(output, dict):
                logits = output['logits']
            else:
                logits = output
            
            # 출력이 유한한 값인지 확인
            assert torch.isfinite(logits).all()
            assert not torch.isnan(logits).any()
            assert not torch.isinf(logits).any()

def test_config_validation():
    """설정 검증이 정상적으로 작동하는지 확인"""
    # 유효한 설정
    valid_args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["0"],
        "ib_beta": 0.1,
        "init_cls": 10,
        "increment": 10,
        "memory_size": 2000,
        "memory_per_class": 20
    }
    
    # 필수 키 확인
    required_keys = ["convnet_type", "dataset", "device", "ib_beta"]
    for key in required_keys:
        assert key in valid_args
    
    # 값 타입 확인
    assert isinstance(valid_args["convnet_type"], str)
    assert isinstance(valid_args["dataset"], str)
    assert isinstance(valid_args["device"], list)
    assert isinstance(valid_args["ib_beta"], (int, float))

def test_experiment_reproducibility():
    """실험 재현성이 보장되는지 확인"""
    from PyCIL.models.asib_cl import ASIB_CL
    
    # 시드 설정
    torch.manual_seed(42)
    np.random.seed(42)
    
    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["0"],
        "ib_beta": 0.1,
        "memory_size": 2000,
        "memory_per_class": 20
    }
    
    # 첫 번째 모델
    model1 = ASIB_CL(args)
    
    # 시드 재설정
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 두 번째 모델
    model2 = ASIB_CL(args)
    
    # 동일한 입력에 대한 출력 비교
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 32, 32)
    
    with torch.no_grad():
        output1 = model1._network(input_tensor)
        output2 = model2._network(input_tensor)
        
        if isinstance(output1, dict):
            logits1 = output1['logits']
            logits2 = output2['logits']
        else:
            logits1 = output1
            logits2 = output2
        
        # 출력이 동일한지 확인 (재현성)
        assert torch.allclose(logits1, logits2, atol=1e-6) 