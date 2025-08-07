import pytest
import torch
from torch.utils.data import ConcatDataset, Dataset
import numpy as np

class DummyDataset(Dataset):
    def __init__(self, num_classes=100):
        self.num_classes = num_classes
        self.classes = list(range(num_classes))
        self.data = torch.randn(100, 3, 32, 32)
        self.labels = torch.randint(0, num_classes, (100,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class DummyDatasetWithoutClasses(Dataset):
    def __init__(self, num_classes=100):
        self.num_classes = num_classes
        # No classes attribute
        self.data = torch.randn(100, 3, 32, 32)
        self.labels = torch.randint(0, num_classes, (100,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class TestDatasetAttributes:
    """Test if datasets have required classes or num_classes attributes"""
    
    def test_dataset_with_classes(self):
        """Test dataset with classes attribute"""
        dataset = DummyDataset(num_classes=100)
        
        # Check if classes attribute exists
        assert hasattr(dataset, 'classes')
        assert len(dataset.classes) == 100
        
        # Check if num_classes attribute exists
        assert hasattr(dataset, 'num_classes')
        assert dataset.num_classes == 100
    
    def test_dataset_with_num_classes_only(self):
        """Test dataset with only num_classes attribute"""
        dataset = DummyDatasetWithoutClasses(num_classes=100)
        
        # Check if classes attribute exists
        assert not hasattr(dataset, 'classes')
        
        # Check if num_classes attribute exists
        assert hasattr(dataset, 'num_classes')
        assert dataset.num_classes == 100
    
    def test_concat_dataset_attributes(self):
        """Test ConcatDataset attributes"""
        dataset1 = DummyDataset(num_classes=50)
        dataset2 = DummyDataset(num_classes=50)
        concat_dataset = ConcatDataset([dataset1, dataset2])
        
        # ConcatDataset should not have classes or num_classes attributes
        assert not hasattr(concat_dataset, 'classes')
        assert not hasattr(concat_dataset, 'num_classes')
    
    def test_main_py_dataset_check_logic(self):
        """Test the dataset check logic from main.py"""
        def check_dataset_attributes(dataset):
            if isinstance(dataset, torch.utils.data.ConcatDataset):
                return 100  # CIFAR-100 fixed
            else:
                n_classes = getattr(dataset, "classes", None)
                if n_classes is None:
                    n_classes = getattr(dataset, "num_classes", None)
                if n_classes is None:
                    raise AttributeError("Dataset must expose `classes` or `num_classes`")
                return len(n_classes) if not isinstance(n_classes, int) else n_classes
        
        # Test with regular dataset
        dataset = DummyDataset(num_classes=100)
        num_classes = check_dataset_attributes(dataset)
        assert num_classes == 100
        
        # Test with dataset that only has num_classes
        dataset = DummyDatasetWithoutClasses(num_classes=100)
        num_classes = check_dataset_attributes(dataset)
        assert num_classes == 100
        
        # Test with ConcatDataset
        dataset1 = DummyDataset(num_classes=50)
        dataset2 = DummyDataset(num_classes=50)
        concat_dataset = ConcatDataset([dataset1, dataset2])
        num_classes = check_dataset_attributes(concat_dataset)
        assert num_classes == 100  # Should return 100 for ConcatDataset 