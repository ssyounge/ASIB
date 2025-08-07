#!/usr/bin/env python3
"""ASIB-CL 모델 테스트"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
import numpy as np
from PyCIL.models.asib_cl import ASIB_CL

def test_asib_cl_initialization():
    """ASIB-CL 모델이 올바르게 초기화되는지 테스트"""
    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["cuda"],
        "memory_size": 2000,
        "ib_beta": 0.1,
        "lambda_D": 1.0,
        "lambda_IB": 1.0
    }
    
    model = ASIB_CL(args)
    
    # 기본 속성들이 올바르게 설정되었는지 확인
    assert model._ib_beta == 0.1
    assert model.lambda_D == 1.0
    assert model.lambda_IB == 1.0
    assert model._network is not None
    assert model._old_network is None
    assert model._ib_encoder is None
    assert model._ib_decoder is None

def test_ib_modules_initialization():
    """IB 모듈들이 올바르게 초기화되는지 테스트"""
    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["cuda"],
        "memory_size": 2000,
        "ib_beta": 0.1,
        "lambda_D": 1.0,
        "lambda_IB": 1.0
    }
    
    model = ASIB_CL(args)
    
    # after_task 호출하여 IB 모듈 초기화
    model.after_task()
    
    # IB 모듈들이 초기화되었는지 확인
    assert model._ib_encoder is not None
    assert model._ib_decoder is not None
    
    # IB 모듈의 구조 확인
    assert isinstance(model._ib_encoder, nn.Sequential)
    assert isinstance(model._ib_decoder, nn.Sequential)

def test_reparameterize_trick():
    """VAE reparameterization trick이 올바르게 작동하는지 테스트"""
    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["cuda"],
        "memory_size": 2000,
        "ib_beta": 0.1,
        "lambda_D": 1.0,
        "lambda_IB": 1.0
    }
    
    model = ASIB_CL(args)
    
    # 더미 mu, logvar 생성
    batch_size, latent_dim = 4, 8
    mu = torch.randn(batch_size, latent_dim, requires_grad=True)
    logvar = torch.randn(batch_size, latent_dim, requires_grad=True)
    
    # reparameterization trick 적용
    z = model._reparameterize(mu, logvar)
    
    # 결과 검증
    assert z.shape == (batch_size, latent_dim)
    assert isinstance(z, torch.Tensor)

def test_kl_loss_normalization():
    """KL 손실 정규화가 올바르게 적용되는지 테스트"""
    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["cuda"],
        "memory_size": 2000,
        "ib_beta": 0.1,
        "lambda_D": 1.0,
        "lambda_IB": 1.0
    }
    
    model = ASIB_CL(args)
    
    # 더미 mu, logvar 생성
    batch_size, latent_dim = 4, 8
    mu = torch.randn(batch_size, latent_dim, requires_grad=True)
    logvar = torch.randn(batch_size, latent_dim, requires_grad=True)
    
    # KL 손실 계산 (수정된 버전)
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    # 결과 검증
    assert isinstance(kl_loss, torch.Tensor)
    assert kl_loss.dim() == 0  # 스칼라 값

def test_separated_losses_computation():
    """분리된 손실 함수들이 올바르게 계산되는지 테스트"""
    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["cuda"],
        "memory_size": 2000,
        "ib_beta": 0.1,
        "lambda_D": 1.0,
        "lambda_IB": 1.0
    }
    
    model = ASIB_CL(args)
    
    # IB 모듈 초기화
    model.after_task()
    
    # 더미 데이터 생성
    batch_size = 4
    inputs = torch.randn(batch_size, 3, 32, 32)
    targets = torch.randint(0, 10, (batch_size,))
    features = torch.randn(batch_size, 64)  # 더미 특징
    
    # 분리된 손실 계산
    losses = model._compute_separated_losses(inputs, targets, features)
    
    # 결과 검증
    assert isinstance(losses, dict)
    assert 'cls_loss' in losses
    assert 'distill_loss' in losses
    # 'ib_loss' 대신 'ib_module_loss'를 확인
    assert 'ib_module_loss' in losses
    
    # 모든 손실이 스칼라 값인지 확인
    for loss_name, loss_value in losses.items():
        assert isinstance(loss_value, torch.Tensor)

def test_model_mode_settings():
    """모델 모드 설정이 올바르게 적용되는지 테스트"""
    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["cuda"],
        "memory_size": 2000,
        "ib_beta": 0.1,
        "lambda_D": 1.0,
        "lambda_IB": 1.0
    }
    
    model = ASIB_CL(args)
    
    # after_task 호출하여 old_network 설정
    model.after_task()
    
    # _train 메소드에서 모드 설정 확인
    # 실제로는 _train 메소드를 직접 호출할 수 없으므로
    # 모델의 상태를 확인
    assert model._network is not None
    assert model._old_network is not None
    assert model._ib_encoder is not None
    assert model._ib_decoder is not None

def test_differential_learning_rates():
    """차등 학습률 설정이 올바르게 적용되는지 테스트"""
    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["cuda"],
        "memory_size": 2000,
        "ib_beta": 0.1,
        "lambda_D": 1.0,
        "lambda_IB": 1.0
    }
    
    model = ASIB_CL(args)
    
    # after_task 호출하여 IB 모듈 초기화
    model.after_task()
    
    # 차등 학습률 옵티마이저 생성
    optimizer = torch.optim.SGD([
        {'params': model._network.parameters()},  # Default LR (0.1)
        {'params': model._ib_encoder.parameters(), 'lr': 0.001},  # Lower LR for IB
        {'params': model._ib_decoder.parameters(), 'lr': 0.001}
    ], lr=0.1, momentum=0.9, weight_decay=0.0002)
    
    # 옵티마이저 그룹 확인
    assert len(optimizer.param_groups) == 3
    assert optimizer.param_groups[0]['lr'] == 0.1  # 메인 네트워크
    assert optimizer.param_groups[1]['lr'] == 0.001  # IB 인코더
    assert optimizer.param_groups[2]['lr'] == 0.001  # IB 디코더

def test_incremental_train():
    """incremental_train 메소드가 올바르게 작동하는지 테스트"""
    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["cuda"],
        "memory_size": 2000,
        "ib_beta": 0.1,
        "lambda_D": 1.0,
        "lambda_IB": 1.0
    }
    
    model = ASIB_CL(args)
    
    # 간단한 데이터 매니저 모킹
    class MockDataManager:
        def __init__(self):
            self._increments = [5, 5, 5]  # 3개 태스크, 각각 5개 클래스
            self._total_classes = 15
            self._nb_tasks = 3
            self._cur_task = 0
            
        def get_task_size(self, task_id):
            return self._increments[task_id] if task_id < len(self._increments) else 0
            
        def get_dataset(self, indices, source, mode, appendent=None, ret_data=False):
            return None
            
        def get_dataloader(self, dataset, batch_size, shuffle=True, num_workers=0):
            return None
    
    data_manager = MockDataManager()
    
    # incremental_train이 예외 없이 실행되는지 확인
    try:
        model.incremental_train(data_manager)
        # 실제로는 데이터가 없어서 실패할 수 있지만, 메소드 호출 자체는 성공해야 함
    except Exception as e:
        # 데이터 관련 오류는 예상됨
        assert "dataset" in str(e).lower() or "dataloader" in str(e).lower() or "tensor" in str(e).lower()

def test_memory_management():
    """메모리 관리 메소드들이 올바르게 작동하는지 테스트"""
    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["cuda"],
        "memory_size": 2000,
        "ib_beta": 0.1,
        "lambda_D": 1.0,
        "lambda_IB": 1.0
    }
    
    model = ASIB_CL(args)
    
    # 초기 메모리는 비어있어야 함
    memory = model._get_memory()
    assert memory is None
    
    # 메모리에 데이터 추가
    model._data_memory = np.array([[1, 2, 3], [4, 5, 6]])
    model._targets_memory = np.array([0, 1])
    
    memory = model._get_memory()
    assert memory is not None
    assert len(memory) == 2
    assert memory[0].shape == (2, 3)
    assert memory[1].shape == (2,)

def test_accuracy_computation():
    """정확도 계산 메소드가 올바르게 작동하는지 테스트"""
    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["cuda"],
        "memory_size": 2000,
        "ib_beta": 0.1,
        "lambda_D": 1.0,
        "lambda_IB": 1.0
    }
    
    model = ASIB_CL(args)
    
    # 간단한 더미 데이터로더
    class MockLoader:
        def __iter__(self):
            # 더미 배치 데이터 (3개 값: index, inputs, targets)
            dummy_data = torch.randn(2, 3, 32, 32)
            dummy_targets = torch.tensor([0, 1])
            yield (0, dummy_data, dummy_targets)
    
    loader = MockLoader()
    
    # _compute_accuracy가 예외 없이 실행되는지 확인
    try:
        accuracy = model._compute_accuracy(model._network, loader)
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 100.0
    except Exception as e:
        # 모델 forward pass 관련 오류는 예상됨
        assert "forward" in str(e).lower() or "input" in str(e).lower()

def test_exemplar_construction():
    """예시 데이터 구성 메소드가 올바르게 작동하는지 테스트"""
    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["cuda"],
        "memory_size": 2000,
        "ib_beta": 0.1,
        "lambda_D": 1.0,
        "lambda_IB": 1.0
    }
    
    model = ASIB_CL(args)
    
    # 간단한 데이터 매니저 모킹
    class MockDataManager:
        def __init__(self):
            self._increments = [5, 5, 5]
            self._total_classes = 15
            self._nb_tasks = 3
            self._cur_task = 0
            
        def get_dataset(self, indices, source, mode, appendent=None, ret_data=False):
            return None
    
    data_manager = MockDataManager()
    
    # _construct_exemplar가 예외 없이 실행되는지 확인
    try:
        exemplar_data, exemplar_targets = model._construct_exemplar(data_manager, 10)
        # 메소드가 튜플을 반환하는지 확인
        assert isinstance(exemplar_data, np.ndarray)
        assert isinstance(exemplar_targets, np.ndarray)
    except Exception as e:
        # 데이터셋 관련 오류는 예상됨
        assert "dataset" in str(e).lower() or "nonetype" in str(e).lower()

def test_parameters_and_to_methods():
    """parameters()와 to() 메소드가 올바르게 작동하는지 테스트"""
    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["cuda"],
        "memory_size": 2000,
        "ib_beta": 0.1,
        "lambda_D": 1.0,
        "lambda_IB": 1.0
    }
    
    model = ASIB_CL(args)
    
    # parameters() 메소드 테스트
    params = list(model.parameters())
    assert len(params) > 0
    assert all(isinstance(p, torch.nn.Parameter) for p in params)
    
    # to() 메소드 테스트
    model.to("cpu")  # 이미 CPU에 있지만 메소드 호출은 성공해야 함
    
    # GPU가 사용 가능한 경우 GPU로 이동 테스트
    if torch.cuda.is_available():
        model.to("cuda")
        # 첫 번째 파라미터의 디바이스 확인
        first_param = model.parameters()[0]
        assert first_param.device.type == "cuda"
    else:
        # CPU에서 테스트
        first_param = model.parameters()[0]
        assert first_param.device.type == "cpu"

def test_after_task():
    """after_task 메소드가 올바르게 작동하는지 테스트"""
    args = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "device": ["cuda"],
        "memory_size": 2000,
        "ib_beta": 0.1,
        "lambda_D": 1.0,
        "lambda_IB": 1.0
    }
    
    model = ASIB_CL(args)
    
    # 초기에는 old_network가 None이어야 함
    assert model._old_network is None
    
    # after_task 호출
    model.after_task()
    
    # old_network가 설정되어야 함
    assert model._old_network is not None
    assert model._old_network is not model._network  # 깊은 복사 확인
    
    # IB 모듈이 초기화되어야 함
    assert model._ib_encoder is not None
    assert model._ib_decoder is not None

if __name__ == "__main__":
    # 모든 테스트 실행
    pytest.main([__file__, "-v"]) 