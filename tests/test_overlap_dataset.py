import pytest
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from data.cifar100_overlap import CIFAR100OverlapDataset, get_overlap_loaders

class MockCIFAR100Dataset(Dataset):
    """Mock CIFAR-100 dataset for testing"""
    def __init__(self, num_samples=1000, num_classes=100):
        self.num_classes = num_classes
        self.classes = list(range(num_classes))
        self.data = torch.randn(num_samples, 3, 32, 32)
        self.labels = torch.randint(0, num_classes, (num_samples,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class TestOverlapDataset:
    """Test overlap dataset functionality"""
    
    def test_cifar100_overlap_dataset_creation(self):
        """Test CIFAR100OverlapDataset creation"""
        # Create mock dataset
        mock_dataset = MockCIFAR100Dataset(num_samples=1000, num_classes=100)
        
        # Create overlap dataset with specific classes
        class_indices = [0, 1, 2, 3, 4]  # First 5 classes
        overlap_dataset = CIFAR100OverlapDataset(mock_dataset, class_indices)
        
        # Check attributes
        assert hasattr(overlap_dataset, 'classes')
        assert hasattr(overlap_dataset, 'num_classes')
        assert overlap_dataset.classes == class_indices
        assert overlap_dataset.num_classes == 5
        
        # Check length
        assert len(overlap_dataset) > 0
        
        # Check data
        data, label = overlap_dataset[0]
        assert isinstance(data, torch.Tensor)
        assert data.shape == (3, 32, 32)
        assert label in range(5)  # Should be mapped to 0-4
    
    def test_overlap_dataset_with_classes_attribute(self):
        """Test that overlap dataset has required attributes for main.py"""
        mock_dataset = MockCIFAR100Dataset(num_samples=1000, num_classes=100)
        class_indices = [0, 1, 2, 3, 4]
        overlap_dataset = CIFAR100OverlapDataset(mock_dataset, class_indices)
        
        # Test the logic from main.py
        n_classes = getattr(overlap_dataset, "classes", None)
        if n_classes is None:
            n_classes = getattr(overlap_dataset, "num_classes", None)
        if n_classes is None:
            raise AttributeError("Dataset must expose `classes` or `num_classes`")
        
        num_classes = len(n_classes) if not isinstance(n_classes, int) else n_classes
        assert num_classes == 5
    
    def test_overlap_dataset_integration(self):
        """Test integration with DataLoader"""
        mock_dataset = MockCIFAR100Dataset(num_samples=1000, num_classes=100)
        class_indices = [0, 1, 2, 3, 4]
        overlap_dataset = CIFAR100OverlapDataset(mock_dataset, class_indices)
        
        # Create DataLoader
        loader = DataLoader(
            overlap_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0  # Use 0 for testing
        )
        
        # Test iteration
        batch = next(iter(loader))
        data, labels = batch
        assert data.shape[0] <= 32  # batch_size
        assert data.shape[1:] == (3, 32, 32)
        assert labels.shape[0] <= 32
        assert all(label in range(5) for label in labels) 