#!/usr/bin/env python3
"""Common fixtures for all tests"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# pytest 설정
def pytest_configure(config):
    """pytest 설정"""
    # test_model_names.py는 pytest로 실행하지 않음 (main 함수만 있음)
    config.addinivalue_line(
        "markers", "manual: marks tests as manual (deselect with '-m \"not manual\"')"
    )

def pytest_collection_modifyitems(config, items):
    """테스트 수집 시 수정"""
    for item in items:
        if "test_model_names" in item.nodeid:
            item.add_marker(pytest.mark.manual)

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def config_name():
    """Sample config name for testing"""
    return "test_config"

@pytest.fixture
def method_class():
    """Sample method class for testing"""
    class DummyMethod:
        def __init__(self):
            self.name = "dummy_method"
    return DummyMethod

@pytest.fixture
def method_name():
    """Sample method name for testing"""
    return "dummy_method"

@pytest.fixture
def config():
    """Sample config for testing"""
    return OmegaConf.create({
        "method": "asib",
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 10
    })

@pytest.fixture
def model_name():
    """Sample model name for testing"""
    return "resnet50"

@pytest.fixture
def dummy_teachers():
    """Create dummy teacher models for testing"""
    class DummyTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_dim = 2048
            self.classifier = nn.Linear(self.feature_dim, 100)
            
        def forward(self, x):
            # Return features and logits
            feat = torch.randn(x.shape[0], self.feature_dim)
            logit = self.classifier(feat)
            return {"feat_2d": feat, "logit": logit}
    
    t1 = DummyTeacher()
    t2 = DummyTeacher()
    return t1, t2
