import pytest
import torch
from torch.utils.data import DataLoader
import numpy as np

class TestDatasetFix:
    """Test that dataset fixes work correctly"""
    
    def test_cifar100_has_required_attributes(self):
        """Test that CIFAR100NPZ has classes and num_classes attributes"""
        from data.cifar100 import CIFAR100NPZ
        
        # Create a mock dataset (we'll mock the file loading)
        with pytest.MonkeyPatch().context() as m:
            # Mock np.load to return dummy data
            def mock_np_load(file_path):
                class MockData:
                    def __enter__(self):
                        return self
                    def __exit__(self, *args):
                        pass
                    def __getitem__(self, key):
                        if key == 'data':
                            return np.random.randint(0, 256, (100, 32, 32, 3), dtype=np.uint8)
                        elif key == 'labels':
                            return np.random.randint(0, 100, (100,), dtype=np.int64)
                return MockData()
            
            m.setattr(np, 'load', mock_np_load)
            
            # Create dataset
            dataset = CIFAR100NPZ(root="./data", train=True, transform=None)
            
            # Check required attributes
            assert hasattr(dataset, 'classes')
            assert hasattr(dataset, 'num_classes')
            assert dataset.classes == list(range(100))
            assert dataset.num_classes == 100
    
    def test_imagenet32_has_required_attributes(self):
        """Test that ImageNet32 has classes and num_classes attributes"""
        from data.imagenet32 import ImageNet32
        
        # Create a mock dataset (we'll mock the file loading)
        with pytest.MonkeyPatch().context() as m:
            # Mock np.load to return dummy data
            def mock_np_load(file_path):
                class MockData:
                    def __enter__(self):
                        return self
                    def __exit__(self, *args):
                        pass
                    def __getitem__(self, key):
                        if key == 'data':
                            return np.random.randint(0, 256, (100, 3, 32, 32), dtype=np.uint8)
                        elif key == 'labels':
                            return np.random.randint(0, 1000, (100,), dtype=np.int64)
                return MockData()
            
            m.setattr(np, 'load', mock_np_load)
            
            # Create dataset
            dataset = ImageNet32(root="./data", split="train", transform=None)
            
            # Check required attributes
            assert hasattr(dataset, 'classes')
            assert hasattr(dataset, 'num_classes')
            assert len(dataset.classes) == 1000
            assert dataset.num_classes == 1000
    
    def test_main_py_dataset_check_passes(self):
        """Test that the main.py dataset check logic passes with our fixes"""
        
        def simulate_main_py_check(dataset):
            """Simulate the exact logic from main.py"""
            if isinstance(dataset, torch.utils.data.ConcatDataset):
                return 100  # CIFAR-100 fixed
            else:
                n_classes = getattr(dataset, "classes", None)
                if n_classes is None:
                    n_classes = getattr(dataset, "num_classes", None)
                if n_classes is None:
                    raise AttributeError("Dataset must expose `classes` or `num_classes`")
                return len(n_classes) if not isinstance(n_classes, int) else n_classes
        
        # Test with CIFAR100NPZ
        from data.cifar100 import CIFAR100NPZ
        with pytest.MonkeyPatch().context() as m:
            def mock_np_load(file_path):
                class MockData:
                    def __enter__(self):
                        return self
                    def __exit__(self, *args):
                        pass
                    def __getitem__(self, key):
                        if key == 'data':
                            return np.random.randint(0, 256, (100, 32, 32, 3), dtype=np.uint8)
                        elif key == 'labels':
                            return np.random.randint(0, 100, (100,), dtype=np.int64)
                return MockData()
            
            m.setattr(np, 'load', mock_np_load)
            
            dataset = CIFAR100NPZ(root="./data", train=True, transform=None)
            result = simulate_main_py_check(dataset)
            assert result == 100
        
        # Test with ImageNet32
        from data.imagenet32 import ImageNet32
        with pytest.MonkeyPatch().context() as m:
            def mock_np_load(file_path):
                class MockData:
                    def __enter__(self):
                        return self
                    def __exit__(self, *args):
                        pass
                    def __getitem__(self, key):
                        if key == 'data':
                            return np.random.randint(0, 256, (100, 3, 32, 32), dtype=np.uint8)
                        elif key == 'labels':
                            return np.random.randint(0, 1000, (100,), dtype=np.int64)
                return MockData()
            
            m.setattr(np, 'load', mock_np_load)
            
            dataset = ImageNet32(root="./data", split="train", transform=None)
            result = simulate_main_py_check(dataset)
            assert result == 1000 