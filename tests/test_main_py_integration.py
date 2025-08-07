import pytest
import torch
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.abspath('.'))

class TestMainPyIntegration:
    """Test main.py integration to catch potential issues"""
    
    def test_main_py_imports(self):
        """Test that all imports in main.py work correctly"""
        # Test core imports
        from core.utils import (
            setup_partial_freeze_schedule_with_cfg,
            setup_safety_switches_with_cfg,
            auto_set_mbm_query_dim_with_model,
            cast_numeric_configs,
        )
        
        # Test data imports
        from data.cifar100 import get_cifar100_loaders
        from data.imagenet32 import get_imagenet32_loaders
        
        # Test model imports
        from models.common.base_wrapper import MODEL_REGISTRY
        from models.common import registry as _reg
        
        # All imports should work
        assert callable(setup_partial_freeze_schedule_with_cfg)
        assert callable(setup_safety_switches_with_cfg)
        assert callable(auto_set_mbm_query_dim_with_model)
        assert callable(cast_numeric_configs)
    
    def test_main_py_function_calls(self):
        """Test that main.py function calls work correctly"""
        from core.utils import setup_partial_freeze_schedule_with_cfg, setup_safety_switches_with_cfg
        
        # Test the function calls that were failing
        cfg = {"num_stages": 4, "some_config": "value"}
        num_stages = 4
        
        # These should work without errors
        setup_partial_freeze_schedule_with_cfg(cfg=cfg, num_stages=num_stages)
        setup_safety_switches_with_cfg(cfg=cfg, num_stages=num_stages)
    
    def test_config_processing(self):
        """Test config processing functions"""
        from core.utils import cast_numeric_configs
        
        # Test config casting
        cfg = {
            "student_lr": "0.1",
            "teacher_lr": "0.01",
            "student_weight_decay": "0.0001",
            "teacher_weight_decay": "0.0001",
            "momentum": "0.9",
            "nesterov": "true",
            "num_stages": "4",
            "student_epochs_per_stage": "15",
            "teacher_adapt_epochs": "0",
        }
        
        # This should convert string values to appropriate types
        cast_numeric_configs(cfg)
        
        # Check that values were converted
        assert isinstance(cfg["student_lr"], float)
        assert isinstance(cfg["teacher_lr"], float)
        assert isinstance(cfg["student_weight_decay"], float)
        assert isinstance(cfg["teacher_weight_decay"], float)
        assert isinstance(cfg["momentum"], float)
        assert isinstance(cfg["nesterov"], bool)
        assert isinstance(cfg["num_stages"], int)
        assert isinstance(cfg["student_epochs_per_stage"], int)
        assert isinstance(cfg["teacher_adapt_epochs"], int)
    
    def test_model_creation_functions(self):
        """Test model creation functions"""
        from core import create_teacher_by_name, create_student_by_name
        
        # Test teacher creation
        teacher = create_teacher_by_name(
            teacher_name="convnext_s",
            num_classes=100,
            pretrained=True,
            small_input=True,
            cfg={}
        )
        assert teacher is not None
        
        # Test student creation
        student = create_student_by_name(
            student_name="resnet50_scratch",
            num_classes=100,
            pretrained=False,
            small_input=True,
            cfg={}
        )
        assert student is not None
    
    def test_dataset_loading_functions(self):
        """Test dataset loading functions"""
        from data.cifar100 import get_cifar100_loaders
        from data.imagenet32 import get_imagenet32_loaders
        
        # Mock the actual data loading to avoid file dependencies
        with patch('data.cifar100.CIFAR100NPZ') as mock_cifar, \
             patch('data.imagenet32.ImageNet32') as mock_imagenet:
            
            # Create mock datasets
            mock_cifar_dataset = MagicMock()
            mock_cifar_dataset.classes = list(range(100))
            mock_cifar_dataset.num_classes = 100
            mock_cifar_dataset.__len__ = lambda self: 1000
            mock_cifar_dataset.__getitem__ = lambda self, idx: (torch.randn(3, 32, 32), torch.randint(0, 100, (1,)).item())
            mock_cifar.return_value = mock_cifar_dataset
            
            mock_imagenet_dataset = MagicMock()
            mock_imagenet_dataset.classes = list(range(1000))
            mock_imagenet_dataset.num_classes = 1000
            mock_imagenet_dataset.__len__ = lambda self: 1000
            mock_imagenet_dataset.__getitem__ = lambda self, idx: (torch.randn(3, 32, 32), torch.randint(0, 1000, (1,)).item())
            mock_imagenet.return_value = mock_imagenet_dataset
            
            # Test CIFAR-100 loading
            train_loader, test_loader = get_cifar100_loaders(
                root="./data",
                batch_size=16,
                num_workers=0,
                augment=False
            )
            assert train_loader is not None
            assert test_loader is not None
            
            # Test ImageNet-32 loading
            train_loader, test_loader = get_imagenet32_loaders(
                root="./data",
                batch_size=16,
                num_workers=0,
                augment=False
            )
            assert train_loader is not None
            assert test_loader is not None
    
    def test_overlap_dataset_loading(self):
        """Test overlap dataset loading"""
        from data.cifar100_overlap import get_overlap_loaders
        
        # Mock the base CIFAR-100 loader
        with patch('data.cifar100_overlap.get_cifar100_loaders') as mock_get_loaders:
            # Create mock loaders
            mock_train_dataset = MagicMock()
            mock_train_dataset.classes = list(range(100))
            mock_train_dataset.num_classes = 100
            mock_train_dataset.class_indices = list(range(50))  # For overlap
            mock_train_dataset.__len__ = lambda self: 1000
            mock_train_dataset.__getitem__ = lambda self, idx: (torch.randn(3, 32, 32), torch.randint(0, 100, (1,)).item())
            
            mock_test_dataset = MagicMock()
            mock_test_dataset.classes = list(range(100))
            mock_test_dataset.num_classes = 100
            mock_test_dataset.class_indices = list(range(50))  # For overlap
            mock_test_dataset.__len__ = lambda self: 1000
            mock_test_dataset.__getitem__ = lambda self, idx: (torch.randn(3, 32, 32), torch.randint(0, 100, (1,)).item())
            
            mock_train_loader = MagicMock()
            mock_train_loader.dataset = mock_train_dataset
            mock_test_loader = MagicMock()
            mock_test_loader.dataset = mock_test_dataset
            
            mock_get_loaders.return_value = (mock_train_loader, mock_test_loader)
            
            # Mock CIFAR100OverlapDataset to avoid DataLoader issues
            with patch('data.cifar100_overlap.CIFAR100OverlapDataset') as mock_overlap_dataset:
                mock_overlap_dataset.return_value = MagicMock(
                    __len__=lambda self: 500,
                    __getitem__=lambda self, idx: (torch.randn(3, 32, 32), torch.randint(0, 100, (1,)).item()),
                    classes=list(range(100)),
                    num_classes=100
                )
                
                # Test overlap loader creation
                (A_tr, A_te), (B_tr, B_te), pairs = get_overlap_loaders(
                    pct_overlap=50,
                    batch_size=16,
                    num_workers=0,
                    augment=False
                )
                
                assert A_tr is not None
                assert A_te is not None
                assert B_tr is not None
                assert B_te is not None
                assert pairs is not None 