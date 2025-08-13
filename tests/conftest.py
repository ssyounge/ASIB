#!/usr/bin/env python3
"""Common fixtures for all tests"""

import pytest
import torch
import numpy as np
import sys
import os
import json

# Ensure project root is importable for tests
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# PyCIL 관련 의존성 제거됨 – 경로 추가 없음

@pytest.fixture(scope="session")
def device():
    """테스트용 디바이스 설정"""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cuda")

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
        "ib_beta": 0.1,
        "lambda_D": 1.0,
        "lambda_IB": 1.0
    }

@pytest.fixture(scope="session")
def small_dataset():
    """작은 테스트 데이터셋"""
    # 실제 데이터 대신 더미 데이터 사용
    return torch.randn(100, 3, 32, 32), torch.randint(0, 10, (100,))

@pytest.fixture(scope="session")
def test_config():
    """테스트용 설정 파일"""
    return {
        "prefix": "test",
        "dataset": "cifar100",
        "memory_size": 50,
        "memory_per_class": 5,
        "fixed_memory": False,
        "shuffle": True,
        "init_cls": 3,
        "increment": 3,
        "model_name": "asib_cl",
        "convnet_type": "resnet32",
        "device": ["0"],
        "seed": [1993],
        "ib_beta": 0.1,
        "lambda_D": 1.0,
        "lambda_IB": 1.0
    }

@pytest.fixture(scope="session")
def dummy_network():
    """더미 네트워크 클래스"""
    class DummyNetwork:
        def __init__(self, feature_dim=512):
            self.feature_dim = feature_dim
            
        def __call__(self, x):
            # 딕셔너리 형태로 반환 (PyCIL 형식)
            return {"logits": torch.randn(x.shape[0], 10)}
            
        def extract_vector(self, x):
            return torch.randn(x.shape[0], self.feature_dim)
            
        def update_fc(self, num_classes):
            pass
            
        def to(self, device):
            return self
    
    return DummyNetwork

@pytest.fixture(scope="session")
def temp_config_file(tmp_path_factory):
    """임시 설정 파일 생성"""
    tmp_path = tmp_path_factory.mktemp("config")
    config_file = tmp_path / "test_config.json"
    
    config = {
        "prefix": "test",
        "dataset": "cifar100",
        "memory_size": 50,
        "memory_per_class": 5,
        "fixed_memory": False,
        "shuffle": True,
        "init_cls": 3,
        "increment": 3,
        "model_name": "asib_cl",
        "convnet_type": "resnet32",
        "device": ["0"],
        "seed": [1993],
        "ib_beta": 0.1,
        "lambda_D": 1.0,
        "lambda_IB": 1.0
    }
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f)
    
    return str(config_file)

@pytest.fixture(scope="session")
def registry_configs():
    """Registry 관련 설정 fixture"""
    return {
        "registry_key_path": "configs/registry_key.yaml",
        "registry_map_path": "configs/registry_map.yaml",
        "experiment_configs_dir": "configs/experiment",
        "finetune_configs_dir": "configs/finetune",
        "models_dir": "models"
    }

@pytest.fixture(scope="session")
def registry_validation():
    """Registry 검증을 위한 fixture"""
    def validate_registry():
        """Registry 일관성 검증"""
        import yaml
        from pathlib import Path
        
        # Registry 파일들 로드
        key_path = Path("configs/registry_key.yaml")
        map_path = Path("configs/registry_map.yaml")
        
        if not key_path.exists() or not map_path.exists():
            return False, "Registry files not found"
        
        with open(key_path, 'r', encoding='utf-8') as f:
            key_config = yaml.safe_load(f)
        with open(map_path, 'r', encoding='utf-8') as f:
            map_config = yaml.safe_load(f)
        
        # 일관성 검사
        key_teachers = set(key_config.get('teacher_keys', []))
        map_teachers = set(map_config.get('teachers', {}).keys())
        
        key_students = set(key_config.get('student_keys', []))
        map_students = set(map_config.get('students', {}).keys())
        
        if key_teachers != map_teachers:
            return False, f"Teacher mismatch: key={key_teachers}, map={map_teachers}"
        
        if key_students != map_students:
            return False, f"Student mismatch: key={key_students}, map={map_students}"
        
        # _student 접미사 검사
        for student_key in key_students:
            if student_key.endswith('_student'):
                return False, f"Student key '{student_key}' should not end with '_student'"
        
        return True, "Registry validation passed"
    
    return validate_registry

@pytest.fixture(scope="session")
def main_config():
    """main.py 테스트용 설정"""
    return {
        "experiment": {
            "dataset": {
                "dataset": {
                    "name": "cifar100",
                    "root": "./data",
                    "small_input": True,
                    "data_aug": 1
                },
                "batch_size": 4,
                "num_workers": 0
            },
            "method": {
                "method": {
                    "name": "asib",
                    "ce_alpha": 0.3,
                    "kd_alpha": 0.0,
                    "kd_ens_alpha": 0.7
                }
            },
            "schedule": {
                "type": "cosine",
                "lr_warmup_epochs": 5,
                "min_lr": 1e-05
            },
            "device": "cuda",
            "seed": 42,
            "small_input": True,
            "batch_size": 4,
            "use_partial_freeze": False,
            "use_amp": True,
            "amp_dtype": "float16",
            "kd_ens_alpha": 0.5,
            "hybrid_beta": 0.05,
            "mixup_alpha": 0.0,
            "cutmix_alpha_distill": 0.0,
            "use_disagree_weight": False,
            "disagree_mode": "both_wrong",
            "disagree_lambda_high": 1.0,
            "disagree_lambda_low": 1.0,
            "feat_kd_alpha": 0.0,
            "feat_kd_key": "feat_2d",
            "feat_kd_norm": "none",
            "rkd_loss_weight": 0.0,
            "rkd_gamma": 2.0,
            "use_ib": False,
            "ib_beta": 0.0,
            "ib_beta_warmup_epochs": 0,
            "ib_mbm_out_dim": 2048,
            "ib_mbm_n_head": 8,
            "ib_mbm_dropout": 0.0,
            "synergy_head_dropout": 0.0,
            "ib_mbm_learnable_q": False,
            "ib_mbm_reg_lambda": 0.0,
            "use_cccp": False,
            "tau": 4.0,
            "reg_lambda": 0.0,
            "grad_clip_norm": 1.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "disable_flops": True,
            "student_epochs_per_stage": 15,
            "teacher1": {
                "model": {
                    "teacher": {
                        "name": "convnext_s",
                        "pretrained": True
                    }
                }
            },
            "teacher2": {
                "model": {
                    "teacher": {
                        "name": "resnet152",
                        "pretrained": True
                    }
                }
            },
            "model": {
                "student": {
                    "model": {
                        "student": {
                            "name": "resnet101_scratch",
                            "pretrained": False
                        }
                    }
                }
            }
        }
    }

@pytest.fixture(scope="session")
def training_config():
    """훈련 테스트용 설정"""
    return {
        "experiment": {
            "dataset": {
                "dataset": {
                    "name": "cifar100",
                    "root": "./data",
                    "small_input": True,
                    "data_aug": 1
                },
                "batch_size": 4,
                "num_workers": 0
            },
            "method": {
                "method": {
                    "name": "asib",
                    "ce_alpha": 0.3,
                    "kd_alpha": 0.0,
                    "kd_ens_alpha": 0.7
                }
            },
            "schedule": {
                "type": "cosine",
                "lr_warmup_epochs": 5,
                "min_lr": 1e-05
            },
            "device": "cuda",
            "seed": 42,
            "small_input": True,
            "batch_size": 4,
            "use_partial_freeze": False,
            "use_amp": True,
            "amp_dtype": "float16",
            "student_epochs_per_stage": 15,
            "teacher1": {
                "model": {
                    "teacher": {
                        "name": "convnext_s",
                        "pretrained": True
                    }
                }
            },
            "teacher2": {
                "model": {
                    "teacher": {
                        "name": "resnet152",
                        "pretrained": True
                    }
                }
            },
            "model": {
                "student": {
                    "model": {
                        "student": {
                            "name": "resnet101_scratch",
                            "pretrained": False
                        }
                    }
                }
            }
        }
    }
