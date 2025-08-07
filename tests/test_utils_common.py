#!/usr/bin/env python3
"""Test utils.common functions used in main.py"""

import torch
import pytest
import numpy as np
import random
from unittest.mock import patch, MagicMock

from utils.common import (
    set_random_seed,
    check_label_range,
    get_model_num_classes,
    count_trainable_parameters,
    get_amp_components,
    mixup_data,
    cutmix_data,
    mixup_criterion
)


class TestSetRandomSeed:
    """Test set_random_seed function"""
    
    def test_set_random_seed_basic(self):
        """Test basic seed setting"""
        set_random_seed(42)
        
        # Check that seeds are set
        assert torch.initial_seed() != 0
        
    def test_set_random_seed_deterministic(self):
        """Test deterministic mode"""
        set_random_seed(42, deterministic=True)
        
        # Check that deterministic flags are set
        if torch.cuda.is_available():
            assert torch.backends.cudnn.deterministic is True
            assert torch.backends.cudnn.benchmark is False
    
    def test_set_random_seed_reproducibility(self):
        """Test that same seed produces same results"""
        set_random_seed(42)
        rand1 = torch.randn(10)
        
        set_random_seed(42)
        rand2 = torch.randn(10)
        
        assert torch.allclose(rand1, rand2)


class TestCheckLabelRange:
    """Test check_label_range function"""
    
    def test_check_label_range_valid(self):
        """Test with valid labels"""
        class MockDataset:
            def __init__(self):
                self.targets = [0, 1, 2, 3, 4]
        
        dataset = MockDataset()
        check_label_range(dataset, 5)  # Should not raise
        
    def test_check_label_range_invalid_min(self):
        """Test with invalid minimum label"""
        class MockDataset:
            def __init__(self):
                self.targets = [-1, 0, 1, 2, 3]
        
        dataset = MockDataset()
        with pytest.raises(ValueError, match="Dataset labels must be within"):
            check_label_range(dataset, 5)
    
    def test_check_label_range_invalid_max(self):
        """Test with invalid maximum label"""
        class MockDataset:
            def __init__(self):
                self.targets = [0, 1, 2, 3, 5]
        
        dataset = MockDataset()
        with pytest.raises(ValueError, match="Dataset labels must be within"):
            check_label_range(dataset, 5)
    
    def test_check_label_range_no_targets(self):
        """Test with dataset that has no targets attribute"""
        class MockDataset:
            pass
        
        dataset = MockDataset()
        check_label_range(dataset, 5)  # Should not raise
    
    def test_check_label_range_labels_attribute(self):
        """Test with dataset that uses 'labels' attribute"""
        class MockDataset:
            def __init__(self):
                self.labels = [0, 1, 2, 3, 4]
        
        dataset = MockDataset()
        check_label_range(dataset, 5)  # Should not raise


class TestGetModelNumClasses:
    """Test get_model_num_classes function"""
    
    def test_get_model_num_classes_linear(self):
        """Test with linear layer"""
        model = torch.nn.Linear(10, 5)
        num_classes = get_model_num_classes(model)
        assert num_classes == 5
    
    def test_get_model_num_classes_sequential(self):
        """Test with sequential model"""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5)
        )
        num_classes = get_model_num_classes(model)
        assert num_classes == 5
    
    def test_get_model_num_classes_no_fc(self):
        """Test with model that has no final layer"""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU()
        )
        with pytest.raises(AttributeError):
            get_model_num_classes(model)


class TestCountTrainableParameters:
    """Test count_trainable_parameters function"""
    
    def test_count_trainable_parameters_basic(self):
        """Test basic parameter counting"""
        model = torch.nn.Linear(10, 5)
        count = count_trainable_parameters(model)
        assert count == 55  # 10*5 + 5 (weights + bias)
    
    def test_count_trainable_parameters_with_frozen(self):
        """Test with frozen parameters"""
        model = torch.nn.Linear(10, 5)
        model.weight.requires_grad = False
        count = count_trainable_parameters(model)
        assert count == 5  # Only bias is trainable
    
    def test_count_trainable_parameters_complex(self):
        """Test with complex model"""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5)
        )
        count = count_trainable_parameters(model)
        expected = 10*20 + 20 + 20*5 + 5  # All weights and biases
        assert count == expected


class TestGetAmpComponents:
    """Test get_amp_components function"""
    
    def test_get_amp_components_disabled(self):
        """Test with AMP disabled"""
        cfg = {"use_amp": False}
        autocast_ctx, scaler = get_amp_components(cfg)
        
        # Should return no-op context managers
        assert autocast_ctx is not None
        assert scaler is not None
    
    def test_get_amp_components_enabled(self):
        """Test with AMP enabled"""
        cfg = {"use_amp": True, "amp_dtype": "float16"}
        autocast_ctx, scaler = get_amp_components(cfg)
        
        assert autocast_ctx is not None
        assert scaler is not None


class TestMixupData:
    """Test mixup_data function"""
    
    def test_mixup_data_basic(self):
        """Test basic mixup functionality"""
        x = torch.randn(4, 3, 32, 32)
        y = torch.tensor([0, 1, 2, 3])
        
        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)
        
        assert mixed_x.shape == x.shape
        assert y_a.shape == y.shape
        assert y_b.shape == y.shape
        assert 0.0 <= lam <= 1.0
    
    def test_mixup_data_alpha_zero(self):
        """Test with alpha=0 (no mixing)"""
        x = torch.randn(4, 3, 32, 32)
        y = torch.tensor([0, 1, 2, 3])
        
        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.0)
        
        # Should be close to original data
        assert torch.allclose(mixed_x, x, atol=1e-6)
        assert torch.allclose(y_a, y)
        assert torch.allclose(y_b, y)
        assert lam == 1.0


class TestCutmixData:
    """Test cutmix_data function"""
    
    def test_cutmix_data_basic(self):
        """Test basic cutmix functionality"""
        x = torch.randn(4, 3, 32, 32)
        y = torch.tensor([0, 1, 2, 3])
        
        mixed_x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)
        
        assert mixed_x.shape == x.shape
        assert y_a.shape == y.shape
        assert y_b.shape == y.shape
        assert 0.0 <= lam <= 1.0
    
    def test_cutmix_data_alpha_zero(self):
        """Test with alpha=0 (no mixing)"""
        x = torch.randn(4, 3, 32, 32)
        y = torch.tensor([0, 1, 2, 3])
        
        mixed_x, y_a, y_b, lam = cutmix_data(x, y, alpha=0.0)
        
        # Should be close to original data
        assert torch.allclose(mixed_x, x, atol=1e-6)
        assert torch.allclose(y_a, y)
        assert torch.allclose(y_b, y)
        assert lam == 1.0


class TestMixupCriterion:
    """Test mixup_criterion function"""
    
    def test_mixup_criterion_basic(self):
        """Test basic mixup criterion"""
        criterion = torch.nn.CrossEntropyLoss()
        pred = torch.randn(4, 10)
        y_a = torch.tensor([0, 1, 2, 3])
        y_b = torch.tensor([1, 2, 3, 0])
        lam = 0.5
        
        loss = mixup_criterion(criterion, pred, y_a, y_b, lam)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
    
    def test_mixup_criterion_extreme_values(self):
        """Test with extreme lambda values"""
        criterion = torch.nn.CrossEntropyLoss()
        pred = torch.randn(4, 10)
        y_a = torch.tensor([0, 1, 2, 3])
        y_b = torch.tensor([1, 2, 3, 0])
        
        # Test lam = 0
        loss_a = mixup_criterion(criterion, pred, y_a, y_b, 0.0)
        
        # Test lam = 1
        loss_b = mixup_criterion(criterion, pred, y_a, y_b, 1.0)
        
        assert isinstance(loss_a, torch.Tensor)
        assert isinstance(loss_b, torch.Tensor)


class TestIntegration:
    """Test integration scenarios"""
    
    def test_main_py_utils_flow(self):
        """Test the flow of utils functions as used in main.py"""
        # Simulate main.py setup
        set_random_seed(42, deterministic=True)
        
        # Create mock dataset
        class MockDataset:
            def __init__(self):
                self.targets = list(range(100))
        
        dataset = MockDataset()
        check_label_range(dataset, 100)
        
        # Create model and count parameters
        model = torch.nn.Linear(10, 100)
        param_count = count_trainable_parameters(model)
        assert param_count > 0
        
        # Test AMP components
        cfg = {"use_amp": False}
        autocast_ctx, scaler = get_amp_components(cfg)
        assert autocast_ctx is not None
        assert scaler is not None
