#!/usr/bin/env python
# data/cifar100_overlap.py
"""CIFAR-100 with class overlap for two teachers."""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from data.cifar100 import get_cifar100_loaders
import numpy as np
from utils.data.overlap import make_pairs

class CIFAR100OverlapDataset(Dataset):
    """CIFAR-100 dataset with class overlap support"""
    
    def __init__(self, dataset, class_indices):
        self.dataset = dataset
        self.class_indices = class_indices
        self.classes = class_indices  # Add classes attribute
        self.num_classes = len(class_indices)  # Add num_classes attribute
        
        # Filter dataset to only include specified classes
        self.indices = []
        for idx, (_, label) in enumerate(dataset):
            if label in class_indices:
                self.indices.append(idx)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        data, label = self.dataset[data_idx]
        # Map label to new class index
        new_label = self.class_indices.index(label)
        return data, new_label

def get_overlap_loaders(pct_overlap, batch_size, num_workers=2, augment=True, seed=42):
    """Get CIFAR-100 loaders with class overlap"""
    
    # Get base CIFAR-100 loaders
    train_loader, test_loader = get_cifar100_loaders(
        root="./data",
        batch_size=batch_size,
        num_workers=num_workers,
        augment=augment
    )
    
    # Generate class pairs with overlap
    pairs = make_pairs(pct_overlap, n_cls=100, seed=seed)
    t1_classes = pairs["T1"]
    t2_classes = pairs["T2"]
    
    # Create datasets for each teacher
    t1_train_dataset = CIFAR100OverlapDataset(train_loader.dataset, t1_classes)
    t1_test_dataset = CIFAR100OverlapDataset(test_loader.dataset, t1_classes)
    
    t2_train_dataset = CIFAR100OverlapDataset(train_loader.dataset, t2_classes)
    t2_test_dataset = CIFAR100OverlapDataset(test_loader.dataset, t2_classes)
    
    # Create data loaders
    t1_train_loader = DataLoader(
        t1_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    t1_test_loader = DataLoader(
        t1_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    t2_train_loader = DataLoader(
        t2_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    t2_test_loader = DataLoader(
        t2_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return (t1_train_loader, t1_test_loader), (t2_train_loader, t2_test_loader), pairs 