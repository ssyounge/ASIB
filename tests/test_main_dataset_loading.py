import pytest
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.abspath('.'))

class TestMainDatasetLoading:
    """Test dataset loading in main.py to identify why classes/num_classes are missing"""
    
    def test_cifar100_dataset_attributes(self):
        """Test if CIFAR-100 dataset has required attributes"""
        from data.cifar100 import get_cifar100_loaders
        
        # Mock the actual data loading to avoid file dependencies
        with patch('data.cifar100.CIFAR100NPZ') as mock_cifar:
            # Create a mock dataset with proper attributes
            mock_dataset = MagicMock()
            mock_dataset.classes = list(range(100))
            mock_dataset.num_classes = 100
            mock_dataset.__len__ = lambda self: 1000
            mock_dataset.__getitem__ = lambda self, idx: (torch.randn(3, 32, 32), torch.randint(0, 100, (1,)).item())
            
            mock_cifar.return_value = mock_dataset
            
            # Test the loader creation
            train_loader, test_loader = get_cifar100_loaders(
                root="./data",
                batch_size=16,
                num_workers=0,
                augment=False
            )
            
            # Check if the dataset has required attributes
            dataset = train_loader.dataset
            assert hasattr(dataset, 'classes') or hasattr(dataset, 'num_classes'), \
                f"Dataset {type(dataset)} missing classes/num_classes attributes"
    
    def test_imagenet32_dataset_attributes(self):
        """Test if ImageNet-32 dataset has required attributes"""
        from data.imagenet32 import get_imagenet32_loaders
        
        # Mock the actual data loading
        with patch('data.imagenet32.ImageNet32') as mock_imagenet:
            # Create a mock dataset with proper attributes
            mock_dataset = MagicMock()
            mock_dataset.classes = list(range(1000))
            mock_dataset.num_classes = 1000
            mock_dataset.__len__ = lambda self: 1000
            mock_dataset.__getitem__ = lambda self, idx: (torch.randn(3, 32, 32), torch.randint(0, 1000, (1,)).item())
            
            mock_imagenet.return_value = mock_dataset
            
            # Test the loader creation
            train_loader, test_loader = get_imagenet32_loaders(
                root="./data",
                batch_size=16,
                num_workers=0,
                augment=False
            )
            
            # Check if the dataset has required attributes
            dataset = train_loader.dataset
            assert hasattr(dataset, 'classes') or hasattr(dataset, 'num_classes'), \
                f"Dataset {type(dataset)} missing classes/num_classes attributes"
    
    def test_overlap_dataset_attributes(self):
        """Test if overlap dataset has required attributes"""
        from data.cifar100_overlap import get_overlap_loaders
        
        # Mock the base CIFAR-100 loader
        with patch('data.cifar100_overlap.get_cifar100_loaders') as mock_get_loaders:
            # Create mock loaders with proper datasets
            mock_train_dataset = MagicMock()
            mock_train_dataset.classes = list(range(100))
            mock_train_dataset.num_classes = 100
            mock_train_dataset.__len__ = lambda self: 1000
            mock_train_dataset.__getitem__ = lambda self, idx: (torch.randn(3, 32, 32), torch.randint(0, 100, (1,)).item())
            
            mock_test_dataset = MagicMock()
            mock_test_dataset.classes = list(range(100))
            mock_test_dataset.num_classes = 100
            mock_test_dataset.__len__ = lambda self: 1000
            mock_test_dataset.__getitem__ = lambda self, idx: (torch.randn(3, 32, 32), torch.randint(0, 100, (1,)).item())
            
            mock_train_loader = MagicMock()
            mock_train_loader.dataset = mock_train_dataset
            mock_test_loader = MagicMock()
            mock_test_loader.dataset = mock_test_dataset
            
            mock_get_loaders.return_value = (mock_train_loader, mock_test_loader)
            
            # Mock CIFAR100OverlapDataset
            with patch('data.cifar100_overlap.CIFAR100OverlapDataset') as mock_overlap_dataset:
                # Create mock overlap dataset
                mock_overlap_instance = MagicMock()
                mock_overlap_instance.classes = list(range(100))
                mock_overlap_instance.num_classes = 100
                mock_overlap_instance.__len__ = lambda self: 1000
                mock_overlap_instance.__getitem__ = lambda self, idx: (torch.randn(3, 32, 32), torch.randint(0, 100, (1,)).item())
                mock_overlap_dataset.return_value = mock_overlap_instance
                
                # Test overlap loader creation
                (A_tr, A_te), (B_tr, B_te), pairs = get_overlap_loaders(
                    pct_overlap=50,
                    batch_size=16,
                    num_workers=0,
                    augment=False
                )
                
                # Check if datasets have required attributes
                for loader in [A_tr, A_te, B_tr, B_te]:
                    dataset = loader.dataset
                    assert hasattr(dataset, 'classes') or hasattr(dataset, 'num_classes'), \
                        f"Dataset {type(dataset)} missing classes/num_classes attributes"
    
    def test_main_py_dataset_check_logic_with_real_datasets(self):
        """Test the dataset attribute checking logic from main.py"""
        
        def check_dataset_attributes(dataset):
            """Simulate main.py's dataset attribute checking logic"""
            # Check if dataset has classes attribute
            if hasattr(dataset, 'classes') and dataset.classes is not None:
                return len(dataset.classes)
            
            # Check if dataset has num_classes attribute
            if hasattr(dataset, 'num_classes') and dataset.num_classes is not None:
                return dataset.num_classes
            
            # Try to infer from dataset length (fallback)
            if hasattr(dataset, '__len__'):
                return len(dataset)
            
            return None
        
        # Test with different dataset types
        test_cases = [
            # (dataset_type, expected_result, description)
            ("dataset_with_classes", 100, "Dataset with classes attribute"),
            ("dataset_with_num_classes", 1000, "Dataset with num_classes attribute"),
            ("dataset_with_length", 500, "Dataset with only length"),
            ("dataset_with_nothing", None, "Dataset with no attributes"),
        ]
        
        for dataset_type, expected, desc in test_cases:
            if dataset_type == "dataset_with_nothing":
                # Create a class without __len__ method
                class MockDatasetNoLen:
                    def __init__(self):
                        self.classes = None
                        self.num_classes = None
                
                mock_dataset = MockDatasetNoLen()
            else:
                # Create a custom mock class instead of MagicMock
                class MockDataset:
                    def __init__(self):
                        self.classes = None
                        self.num_classes = None
                        self._length = None
                    
                    def __len__(self):
                        return self._length if self._length is not None else 0
                
                mock_dataset = MockDataset()
                
                if dataset_type == "dataset_with_classes":
                    mock_dataset.classes = list(range(100))
                elif dataset_type == "dataset_with_num_classes":
                    mock_dataset.num_classes = 1000
                elif dataset_type == "dataset_with_length":
                    mock_dataset._length = 500
            
            result = check_dataset_attributes(mock_dataset)
            assert result == expected, f"Failed for {desc}: expected {expected}, got {result}"
    
    def test_concat_dataset_behavior(self):
        """Test behavior with ConcatDataset"""
        from torch.utils.data import ConcatDataset
        
        def check_dataset_attributes(dataset):
            """Simulate main.py's dataset attribute checking logic"""
            # Check if dataset has classes attribute
            if hasattr(dataset, 'classes') and dataset.classes is not None:
                return len(dataset.classes)
            
            # Check if dataset has num_classes attribute
            if hasattr(dataset, 'num_classes') and dataset.num_classes is not None:
                return dataset.num_classes
            
            # Try to infer from dataset length (fallback)
            if hasattr(dataset, '__len__'):
                return len(dataset)
            
            return None
        
        # Create mock sub-datasets
        dataset1 = MagicMock()
        dataset1.classes = list(range(50))
        dataset1.__len__ = lambda self: 100
        
        dataset2 = MagicMock()
        dataset2.classes = list(range(50, 100))
        dataset2.__len__ = lambda self: 100
        
        # Create ConcatDataset
        concat_dataset = ConcatDataset([dataset1, dataset2])
        
        # Test attribute checking
        result = check_dataset_attributes(concat_dataset)
        # ConcatDataset doesn't have classes/num_classes, so should fall back to length
        assert result == 200, f"ConcatDataset should return length, got {result}" 