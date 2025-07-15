# data/overlap.py
"""Utility to create two overlapping CIFAR-100 loaders."""

from __future__ import annotations

import torch
import torchvision
import torchvision.transforms as T
from typing import Tuple


def get_overlap_loaders(
    rho: float,
    *,
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Return two train loaders sharing a fraction of data.

    Parameters
    ----------
    rho : float
        Overlap ratio between the two splits (0.0 to 1.0).
    root : str, optional
        Dataset root folder, by default ``"./data"``.
    batch_size : int, optional
        Mini-batch size, by default ``128``.
    num_workers : int, optional
        Data loader worker count, by default ``0``.
    seed : int, optional
        Random seed used for the split, by default ``42``.

    Returns
    -------
    tuple of DataLoader
        ``(train_loader1, train_loader2, test_loader)``.
    """
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    base_train = torchvision.datasets.CIFAR100(
        root=root, train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=root, train=False, download=True, transform=transform_test
    )

    n = len(base_train)
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    overlap_n = int(n * rho)
    unique_n = (n - overlap_n) // 2

    shared_idx = indices[:overlap_n]
    unique1 = indices[overlap_n : overlap_n + unique_n]
    unique2 = indices[overlap_n + unique_n : overlap_n + 2 * unique_n]
    leftover = indices[overlap_n + 2 * unique_n :]
    shared_idx.extend(leftover)

    idx1 = unique1 + shared_idx
    idx2 = unique2 + shared_idx

    ds1 = torch.utils.data.Subset(base_train, idx1)
    ds2 = torch.utils.data.Subset(base_train, idx2)

    dl_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_loader1 = torch.utils.data.DataLoader(ds1, **dl_kwargs)
    train_loader2 = torch.utils.data.DataLoader(ds2, **dl_kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader1, train_loader2, test_loader
