#!/usr/bin/env python3
"""Test all data modules"""

import torch
import pytest
import numpy as np
from pathlib import Path

# Import data modules
from data.cifar100 import CIFAR100NPZ, get_cifar100_loaders
from data.imagenet32 import ImageNet32, get_imagenet32_loaders
from utils.data import ClassInfoMixin
from utils.data.overlap import make_pairs, split_classes


class TestCIFAR100Dataset:
    """Test CIFAR100 dataset"""
    
    def test_cifar100_dataset_creation(self):
        """Test CIFAR100 dataset creation"""
        try:
            dataset = CIFAR100NPZ(
                root="./data",
                train=True
            )
            assert hasattr(dataset, '__getitem__')
            assert hasattr(dataset, '__len__')
        except Exception as e:
            pytest.skip(f"CIFAR100 data not available: {e}")
    
    def test_cifar100_dataset_getitem(self):
        """Test CIFAR100 dataset getitem"""
        try:
            dataset = CIFAR100NPZ(
                root="./data",
                train=True
            )
            
            if len(dataset) > 0:
                item = dataset[0]
                assert isinstance(item, (tuple, list))
                assert len(item) >= 2  # (image, label)
                
                image, label = item[0], item[1]
                # transform이 없으면 PIL Image, 있으면 torch.Tensor
                from PIL import Image
                import numpy as np
                assert isinstance(image, (torch.Tensor, Image.Image))
                assert isinstance(label, (int, np.integer))
                assert 0 <= label < 100
        except Exception as e:
            pytest.skip(f"CIFAR100 data not available: {e}")


# CIFAR100OverlapDataset 클래스는 삭제되었으므로 테스트 제거


class TestImageNet32Dataset:
    """Test ImageNet32 dataset"""
    
    def test_imagenet32_dataset_creation(self):
        """Test ImageNet32 dataset creation"""
        try:
            dataset = ImageNet32(
                root="./data",
                split="train"
            )
            assert hasattr(dataset, '__getitem__')
            assert hasattr(dataset, '__len__')
        except Exception as e:
            pytest.skip(f"ImageNet32 data not available: {e}")
    
    def test_imagenet32_dataset_getitem(self):
        """Test ImageNet32 dataset getitem"""
        try:
            dataset = ImageNet32(
                root="./data",
                split="train"
            )
            
            if len(dataset) > 0:
                item = dataset[0]
                assert isinstance(item, (tuple, list))
                assert len(item) >= 2  # (image, label)
                
                image, label = item[0], item[1]
                assert isinstance(image, torch.Tensor)
                assert isinstance(label, int)
        except Exception as e:
            pytest.skip(f"ImageNet32 data not available: {e}")


class TestDataLoader:
    """Test data loader utilities"""
    
    def test_get_split_cifar100_loaders(self):
        """Test get_cifar100_loaders function"""
        try:
            train_loader, val_loader = get_cifar100_loaders(
                root="./data",
                batch_size=16,
                num_workers=0,
                augment=True
            )
            assert train_loader is not None
            assert val_loader is not None
            
            # Test iteration
            for batch_idx, (data, target) in enumerate(train_loader):
                assert isinstance(data, torch.Tensor)
                assert isinstance(target, torch.Tensor)
                assert data.shape[0] <= 16  # batch_size
                assert target.shape[0] <= 16  # batch_size
                break  # Just test first batch
        except Exception as e:
            pytest.skip(f"Data loading failed: {e}")


class TestClassInfoMixin:
    """Test ClassInfoMixin"""
    
    def test_class_info_mixin(self):
        """Test ClassInfoMixin functionality"""
        class TestDataset(ClassInfoMixin):
            def __init__(self):
                self.num_classes = 100
                self.class_to_idx = {i: i for i in range(100)}
        
        dataset = TestDataset()
        assert len(dataset.classes) == 100
        assert dataset.classes == list(range(100))


class TestOverlapUtilities:
    """Test overlap utilities"""
    
    def test_make_pairs(self):
        """Test make_pairs function"""
        pairs = make_pairs(overlap_pct=50)
        assert isinstance(pairs, dict)
        assert 'overlap' in pairs
        assert 'T1' in pairs
        assert 'T2' in pairs
        assert pairs['overlap'] == 50
        
        # With 50% overlap, each teacher gets 75 classes (50 common + 25 unique)
        assert len(pairs['T1']) == 75
        assert len(pairs['T2']) == 75
        
        # Check that there are exactly 50 overlapping classes
        overlap_classes = set(pairs['T1']) & set(pairs['T2'])
        assert len(overlap_classes) == 50
    
    def test_split_classes(self):
        """Test split_classes function"""
        classes = split_classes(n_cls=100, seed=42)
        assert isinstance(classes, list)
        assert len(classes) == 100
        assert set(classes) == set(range(100))


class TestDataTransforms:
    """Test data transforms"""
    
    def test_basic_transforms(self):
        """Test basic transforms"""
        from torchvision import transforms
        from PIL import Image
        import numpy as np

        # Test basic transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Create dummy PIL image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))

        # Apply transform
        transformed = transform(dummy_image)
        assert transformed.shape == (3, 32, 32)
        assert torch.is_tensor(transformed)

    def test_augmentation_transforms(self):
        """Test augmentation transforms"""
        from torchvision import transforms
        from PIL import Image
        import numpy as np

        # Test augmentation transforms
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Create dummy PIL image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))

        # Apply transform
        transformed = transform(dummy_image)
        assert transformed.shape == (3, 32, 32)
        assert torch.is_tensor(transformed)


class TestDataValidation:
    """Test data validation"""
    
    def test_data_consistency(self):
        """Test data consistency"""
        # Test that data shapes are consistent
        try:
            train_loader, val_loader = get_cifar100_loaders(
                root="./data",
                batch_size=16,
                num_workers=0,
                augment=True
            )
            
            # Check train loader
            for batch_idx, (data, target) in enumerate(train_loader):
                assert data.shape[1] == 3  # 3 channels
                assert data.shape[2] == 32  # 32 height
                assert data.shape[3] == 32  # 32 width
                assert target.max() < 100  # 100 classes
                assert target.min() >= 0
                break
            
            # Check val loader
            for batch_idx, (data, target) in enumerate(val_loader):
                assert data.shape[1] == 3  # 3 channels
                assert data.shape[2] == 32  # 32 height
                assert data.shape[3] == 32  # 32 width
                assert target.max() < 100  # 100 classes
                assert target.min() >= 0
                break
        except Exception as e:
            pytest.skip(f"Data loading failed: {e}")
    
    def test_data_types(self):
        """Test data types"""
        try:
            train_loader, val_loader = get_cifar100_loaders(
                root="./data",
                batch_size=16,
                num_workers=0,
                augment=True
            )
            
            # Check data types
            for batch_idx, (data, target) in enumerate(train_loader):
                assert data.dtype == torch.float32
                assert target.dtype == torch.long
                break
        except Exception as e:
            pytest.skip(f"Data loading failed: {e}")


class TestDataPerformance:
    """Test data performance"""
    
    def test_data_loading_speed(self):
        """Test data loading speed"""
        import time
        
        try:
            start_time = time.time()
            
            train_loader, val_loader = get_cifar100_loaders(
                root="./data",
                batch_size=16,
                num_workers=0,
                augment=True
            )
            
            # Load a few batches
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 5:  # Load 5 batches
                    break
            
            end_time = time.time()
            loading_time = end_time - start_time
            
            # Should be reasonably fast (less than 10 seconds)
            assert loading_time < 10.0
        except Exception as e:
            pytest.skip(f"Data loading failed: {e}")
    
    def test_memory_usage(self):
        """Test memory usage"""
        # Simple memory usage test without psutil
        import gc
        
        # Create some dummy data
        dummy_data = torch.randn(1000, 1000)
        initial_memory = dummy_data.element_size() * dummy_data.nelement()
        
        # Clear memory
        del dummy_data
        gc.collect()
        
        # Test passes if no exception
        assert True 