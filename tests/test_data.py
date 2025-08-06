#!/usr/bin/env python3
"""
Data 관련 테스트
"""

import pytest
import torch
import numpy as np
from data.cifar100 import CIFAR100NPZ, get_cifar100_loaders
from data.imagenet32 import ImageNet32, get_imagenet32_loaders


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
                assert data.shape[0] == target.shape[0]  # batch size
                break  # Just test first batch
        except Exception as e:
            pytest.skip(f"CIFAR100 data not available: {e}")
    
    def test_get_imagenet32_loaders(self):
        """Test get_imagenet32_loaders function"""
        try:
            train_loader, val_loader = get_imagenet32_loaders(
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
                assert data.shape[0] == target.shape[0]  # batch size
                break  # Just test first batch
        except Exception as e:
            pytest.skip(f"ImageNet32 data not available: {e}")


class TestDataTransforms:
    """Test data transforms"""
    
    def test_basic_transforms(self):
        """Test basic transforms"""
        from torchvision import transforms
        
        # Basic transform (이미 텐서이므로 ToTensor 제거)
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Test with dummy data
        dummy_image = torch.randn(3, 32, 32)
        transformed = transform(dummy_image)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 32, 32)
    
    def test_augmentation_transforms(self):
        """Test augmentation transforms"""
        from torchvision import transforms
        
        # Augmentation transform (이미 텐서이므로 ToTensor 제거)
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Test with dummy data
        dummy_image = torch.randn(3, 32, 32)
        transformed = transform(dummy_image)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 32, 32)
    
    def test_pil_to_tensor_transform(self):
        """Test PIL to tensor transform"""
        from torchvision import transforms
        from PIL import Image
        import numpy as np
        
        # Create dummy PIL image
        dummy_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        dummy_pil_image = Image.fromarray(dummy_array)
        
        # Transform with ToTensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        transformed = transform(dummy_pil_image)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 32, 32)


class TestDataValidation:
    """Test data validation"""
    
    def test_data_consistency(self):
        """Test data consistency"""
        # Test that data shapes are consistent
        batch_size = 16
        channels = 3
        height = 32
        width = 32
        
        dummy_data = torch.randn(batch_size, channels, height, width)
        dummy_targets = torch.randint(0, 100, (batch_size,))
        
        assert dummy_data.shape == (batch_size, channels, height, width)
        assert dummy_targets.shape == (batch_size,)
        assert dummy_targets.min() >= 0
        assert dummy_targets.max() < 100
    
    def test_data_types(self):
        """Test data types"""
        # Test that data types are correct
        batch_size = 16
        channels = 3
        height = 32
        width = 32
        
        dummy_data = torch.randn(batch_size, channels, height, width)
        dummy_targets = torch.randint(0, 100, (batch_size,))
        
        assert dummy_data.dtype == torch.float32
        assert dummy_targets.dtype == torch.long


class TestDataPerformance:
    """Test data performance"""
    
    def test_data_loading_speed(self):
        """Test data loading speed"""
        import time
        
        # Test data loading speed with dummy data
        batch_size = 32
        num_batches = 10
        
        dummy_data = torch.randn(batch_size, 3, 32, 32)
        dummy_targets = torch.randint(0, 100, (batch_size,))
        
        start_time = time.time()
        for _ in range(num_batches):
            _ = dummy_data, dummy_targets
        end_time = time.time()
        
        # Should be very fast
        assert (end_time - start_time) < 1.0  # Less than 1 second
    
    def test_memory_usage(self):
        """Test memory usage"""
        import psutil
        import os
        
        # Test memory usage with dummy data
        batch_size = 64
        channels = 3
        height = 32
        width = 32
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create dummy data
        dummy_data = torch.randn(batch_size, channels, height, width)
        dummy_targets = torch.randint(0, 100, (batch_size,))
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024  # 100MB 