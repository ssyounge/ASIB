#!/usr/bin/env python3
"""Test all utility modules"""

import torch
import pytest
import numpy as np
from pathlib import Path

# Import utils modules
from utils.common import (
    smart_tqdm, mixup_data, cutmix_data, mixup_criterion,
    get_amp_components, count_trainable_parameters
)
from utils.common.config import load_config, save_config
from utils.common.params import count_trainable_parameters as count_params
# from utils.data import get_dataloaders  # Function not available
from utils.logging.setup import setup_logging, get_logger
from utils.training.metrics import compute_accuracy, StageMeter, ExperimentMeter
from utils.training.freeze import freeze_all, unfreeze_by_regex, apply_bn_ln_policy


class TestCommonUtils:
    """Test common utility functions"""
    
    def test_smart_tqdm(self):
        """Test smart_tqdm function"""
        from utils.common import smart_tqdm
        
        # Test with list
        items = list(range(10))
        result = list(smart_tqdm(items, desc="Test"))
        assert result == items
    
    def test_mixup_data(self):
        """Test mixup_data function"""
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 100, (4,))
        alpha = 0.2
        
        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha)
        
        assert mixed_x.shape == x.shape
        assert y_a.shape == y.shape
        assert y_b.shape == y.shape
        assert 0 <= lam <= 1
    
    def test_cutmix_data(self):
        """Test cutmix_data function"""
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 100, (4,))
        alpha = 0.2
        
        mixed_x, y_a, y_b, lam = cutmix_data(x, y, alpha)
        
        assert mixed_x.shape == x.shape
        assert y_a.shape == y.shape
        assert y_b.shape == y.shape
        assert 0 <= lam <= 1
    
    def test_mixup_criterion(self):
        """Test mixup_criterion function"""
        criterion = torch.nn.CrossEntropyLoss()
        pred = torch.randn(4, 100)
        y_a = torch.randint(0, 100, (4,))
        y_b = torch.randint(0, 100, (4,))
        lam = 0.5
        
        loss = mixup_criterion(criterion, pred, y_a, y_b, lam)
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_get_amp_components(self):
        """Test get_amp_components function"""
        # Test with dict config
        cfg = {"use_amp": True}
        scaler, autocast = get_amp_components(cfg)
        assert scaler is not None or autocast is not None
        
        # Test with dict config without use_amp
        cfg = {"use_amp": False}
        scaler, autocast = get_amp_components(cfg)
    
    def test_count_trainable_parameters(self):
        """Test count_trainable_parameters function"""
        model = torch.nn.Linear(10, 5)
        count = count_trainable_parameters(model)
        assert count == 55  # 10*5 + 5 (weights + bias)


class TestConfigUtils:
    """Test configuration utilities"""
    
    def test_load_config(self, tmp_path):
        """Test load_config function"""
        # Create a temporary config file
        config_content = """
{
    "test": {
        "value": 42,
        "nested": {
            "key": "test"
        }
    }
}
"""
        config_file = tmp_path / "test_config.json"
        config_file.write_text(config_content)
        
        # Test loading
        config = load_config(str(config_file))
        assert config["test"]["value"] == 42
        assert config["test"]["nested"]["key"] == "test"
    
    def test_save_config(self, tmp_path):
        """Test save_config function"""
        config = {"test": {"value": 42}}
        config_file = tmp_path / "test_save_config.yaml"
        
        # Test saving
        save_config(config, str(config_file))
        assert config_file.exists()
        
        # Test loading back
        loaded_config = load_config(str(config_file))
        assert loaded_config["test"]["value"] == 42


class TestDataUtils:
    """Test data utilities"""
    
    def test_get_dataloaders(self):
        """Test get_dataloaders function"""
        # Function not available, skip test
        pytest.skip("get_dataloaders function not available")


class TestLoggingUtils:
    """Test logging utilities"""
    
    def test_setup_logging(self, tmp_path):
        """Test setup_logging function"""
        log_dir = tmp_path / "logs"
        cfg = {
            "results_dir": str(log_dir),
            "log_filename": "test.log"
        }
        logger = setup_logging(cfg)
        assert logger is not None
        assert log_dir.exists()
    
    def test_get_logger(self):
        """Test get_logger function"""
        logger = get_logger("test_logger")
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'error')


class TestTrainingMetrics:
    """Test training metrics"""
    
    def test_compute_accuracy(self):
        """Test compute_accuracy function"""
        pred = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        target = torch.tensor([1, 0, 1])
        
        accuracy = compute_accuracy(pred, target)
        assert 0 <= accuracy <= 100  # accuracy is returned as percentage
        assert torch.isfinite(torch.tensor(accuracy))
    
    def test_stage_meter(self):
        """Test StageMeter class"""
        # Create dummy logger and config
        import logging
        logger = logging.getLogger("test")
        cfg = {}
        
        # Create dummy student model
        class DummyStudent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(10, 5)
            
            def forward(self, x):
                return {"logit": self.classifier(x)}
        
        student = DummyStudent()
        
        # Create a mock logger with update_metric method
        class MockLogger:
            def __init__(self):
                self.metrics = {}
            
            def update_metric(self, key, value):
                self.metrics[key] = value
            
            def info(self, msg, *args):
                pass
        
        mock_logger = MockLogger()
        
        meter = StageMeter(1, mock_logger, cfg, student)
        
        # Test update
        meter.step(4)
        
        # Test finish
        result = meter.finish(0.8)
        assert "wall_min" in result
        assert "acc" in result
    
    def test_experiment_meter(self):
        """Test ExperimentMeter class"""
        # Create dummy logger and config
        import logging
        logger = logging.getLogger("test")
        cfg = {}
        
        # Create dummy student model
        class DummyStudent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(10, 5)
            
            def forward(self, x):
                return {"logit": self.classifier(x)}
        
        student = DummyStudent()
        
        meter = ExperimentMeter(logger, cfg, student)
        
        # Test add stage
        meter.add_stage_metrics(10.0, 0.1, 1000.0, 0.8)
        
        # Test finalize - this should return None but log the results
        result = meter.finish_experiment()
        # finish_experiment returns None but logs the results
        assert result is None


class TestFreezeUtils:
    """Test freeze utilities"""
    
    def test_freeze_all(self):
        """Test freeze_all function"""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        # Check initial state
        initial_params = [p.requires_grad for p in model.parameters()]
        assert any(initial_params)
        
        # Freeze all
        freeze_all(model)
        
        # Check frozen state
        frozen_params = [p.requires_grad for p in model.parameters()]
        assert not any(frozen_params)
    
    def test_unfreeze_by_regex(self):
        """Test unfreeze_by_regex function"""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        # Freeze all first
        freeze_all(model)
        
        # Unfreeze by regex - use the actual parameter name pattern
        unfreeze_by_regex(model, r"2\.weight")
        
        # Check specific layer is unfrozen
        params = list(model.parameters())
        assert not params[0].requires_grad  # 0.weight (frozen)
        assert params[2].requires_grad      # 2.weight (unfrozen, matches "2.weight" pattern)
    
    def test_apply_bn_ln_policy(self):
        """Test apply_bn_ln_policy function"""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.BatchNorm1d(5),
            torch.nn.Linear(5, 1)
        )
        
        # Apply policy
        apply_bn_ln_policy(model, train_bn=False, train_ln=True)
        
        # Check BN parameters are frozen
        bn_params = list(model[1].parameters())
        assert not bn_params[0].requires_grad  # weight
        assert not bn_params[1].requires_grad  # bias


class TestLossFunctions:
    """Test loss functions"""
    
    def test_ib_loss(self):
        """Test IB loss function"""
        from modules.losses import ib_loss
        
        # Create dummy inputs
        student_feat = torch.randn(4, 128)
        teacher_feat = torch.randn(4, 128)
        beta = 0.1
        
        loss = ib_loss(student_feat, teacher_feat, beta)
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_kl_loss(self):
        """Test KL loss function"""
        from modules.losses import kl_loss
        
        # Create dummy inputs
        student_logits = torch.randn(4, 100)
        teacher_logits = torch.randn(4, 100)
        temperature = 4.0
        
        loss = kl_loss(student_logits, teacher_logits, temperature)
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_mse_loss(self):
        """Test MSE loss function"""
        from modules.losses import mse_loss
        
        # Create dummy inputs
        student_feat = torch.randn(4, 128)
        teacher_feat = torch.randn(4, 128)
        
        loss = mse_loss(student_feat, teacher_feat)
        assert torch.isfinite(loss)
        assert loss > 0 