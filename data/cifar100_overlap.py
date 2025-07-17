# data/cifar100_overlap.py
"""Utilities for CIFAR-100 class-overlap splits."""

import warnings
from typing import Mapping, Any, Optional, Sequence, Tuple

import torch
import torchvision
import torchvision.transforms as T
from utils.transform_utils import SafeToTensor

__all__ = ["get_overlap_loaders"]

_MEAN = (0.5071, 0.4865, 0.4409)
_STD = (0.2673, 0.2564, 0.2762)


def _subset(dataset: torchvision.datasets.CIFAR100, class_ids: Sequence[int]):
    idx = [i for i, t in enumerate(dataset.targets) if t in class_ids]
    return torch.utils.data.Subset(dataset, idx)


def get_overlap_loaders(
    rho: float,
    *,
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
    augment: bool = True,
    randaug_N: int = 0,
    randaug_M: int = 0,
    cfg: Optional[Mapping[str, Any]] = None,
    randaug_default_N: int = 2,
    randaug_default_M: int = 9,
    persistent_train: bool = False,
    persistent_test: Optional[bool] = None,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """Return dataloaders for two CIFAR-100 splits with partial overlap.

    ``rho`` denotes the fraction of the first 50 classes that also appear in the
    second split. ``rho=0`` results in disjoint splits while ``rho=1`` makes the
    two splits identical.
    """
    assert 0.0 <= rho <= 1.0, "rho must be in [0, 1]"

    if augment:
        ops = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
        if cfg is not None:
            randaug_default_N = cfg.get("randaug_default_N", randaug_default_N)
            randaug_default_M = cfg.get("randaug_default_M", randaug_default_M)
        if randaug_N > 0 and randaug_M > 0:
            ops.append(T.RandAugment(num_ops=randaug_N, magnitude=randaug_M))
        else:
            ops.append(
                T.RandAugment(num_ops=randaug_default_N, magnitude=randaug_default_M)
            )
        ops = ops + [SafeToTensor(), T.Normalize(_MEAN, _STD)]
        transform_train = T.Compose(ops)
    else:
        transform_train = T.Compose([SafeToTensor(), T.Normalize(_MEAN, _STD)])

    transform_test = T.Compose([SafeToTensor(), T.Normalize(_MEAN, _STD)])

    base_train = torchvision.datasets.CIFAR100(
        root=root, train=True, download=True, transform=transform_train
    )
    base_test = torchvision.datasets.CIFAR100(
        root=root, train=False, download=True, transform=transform_test
    )

    classes = list(range(100))
    first = classes[:50]
    second = classes[50:]
    n_overlap = int(round(50 * rho))
    split1 = first
    split2 = first[:n_overlap] + second[:50 - n_overlap]

    train_ds1 = _subset(base_train, split1)
    train_ds2 = _subset(base_train, split2)
    test_ds1 = _subset(base_test, split1)
    test_ds2 = _subset(base_test, split2)

    if persistent_test is None:
        persistent_test = persistent_train

    if persistent_train and num_workers == 0:
        warnings.warn("persistent_workers=True 이지만 num_workers=0 → 비활성화")
        persistent_train = False
    if persistent_test and num_workers == 0:
        warnings.warn("persistent_workers=True 이지만 num_workers=0 → 비활성화")
        persistent_test = False

    mp_train = (
        torch.multiprocessing.get_context("spawn")
        if persistent_train and num_workers > 0
        else None
    )
    mp_test = (
        torch.multiprocessing.get_context("spawn")
        if persistent_test and num_workers > 0
        else None
    )

    dl_kwargs_train = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_train and num_workers > 0,
    )
    dl_kwargs_test = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_test and num_workers > 0,
    )
    if mp_train is not None and "multiprocessing_context" in torch.utils.data.DataLoader.__init__.__code__.co_varnames:
        dl_kwargs_train["multiprocessing_context"] = mp_train
    if mp_test is not None and "multiprocessing_context" in torch.utils.data.DataLoader.__init__.__code__.co_varnames:
        dl_kwargs_test["multiprocessing_context"] = mp_test

    train_loader1 = torch.utils.data.DataLoader(train_ds1, shuffle=True, **dl_kwargs_train)
    train_loader2 = torch.utils.data.DataLoader(train_ds2, shuffle=True, **dl_kwargs_train)
    test_loader1 = torch.utils.data.DataLoader(test_ds1, shuffle=False, **dl_kwargs_test)
    test_loader2 = torch.utils.data.DataLoader(test_ds2, shuffle=False, **dl_kwargs_test)

    return train_loader1, train_loader2, test_loader1, test_loader2
