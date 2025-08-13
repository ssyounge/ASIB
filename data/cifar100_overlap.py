#!/usr/bin/env python
# data/cifar100_overlap.py
"""CIFAR-100 with class overlap for two teachers."""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from data.cifar100 import get_cifar100_loaders
import numpy as np
from utils.data.overlap import make_pairs

class CIFAR100OverlapDataset(Dataset):
    """CIFAR-100 dataset with class overlap support (optimized)."""
    def __init__(self, dataset, class_indices):
        self.dataset = dataset
        self.class_indices = list(class_indices)
        self.classes = self.class_indices
        self.num_classes = len(self.class_indices)
        # fast label map and index filtering
        label_map = {orig: i for i, orig in enumerate(self.class_indices)}
        targets = getattr(dataset, "targets", None)
        if targets is None:
            # fallback: extract labels without applying transforms
            targets = [dataset[i][1] for i in range(len(dataset))]
        self._label_map = label_map
        self.indices = [i for i, y in enumerate(targets) if y in label_map]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        x, y = self.dataset[i]
        return x, self._label_map[y]

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