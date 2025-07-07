# data/cifar100.py

import warnings
import torch
import torchvision
import torchvision.transforms as T
from typing import Mapping, Any, Optional

def get_cifar100_loaders(
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
):
    """
    CIFAR-100 size = (32x32)
    Returns:
        train_loader, test_loader
    """
    if augment:
        aug_ops = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
        if cfg is not None:
            randaug_default_N = cfg.get("randaug_default_N", randaug_default_N)
            randaug_default_M = cfg.get("randaug_default_M", randaug_default_M)
        if randaug_N > 0 and randaug_M > 0:
            aug_ops.append(T.RandAugment(num_ops=randaug_N, magnitude=randaug_M))
        else:
            aug_ops.append(T.RandAugment(num_ops=randaug_default_N, magnitude=randaug_default_M))
        aug_ops.extend([
            T.ToTensor(),
            T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
        transform_train = T.Compose(aug_ops)
    else:
        transform_train = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5071,0.4865,0.4409),
                        (0.2673,0.2564,0.2762))
        ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071,0.4865,0.4409),
                    (0.2673,0.2564,0.2762))
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=root,
        train=True,
        download=True,
        transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=root,
        train=False,
        download=True,
        transform=transform_test
    )

    if persistent_test is None:
        persistent_test = persistent_train

    if persistent_train and num_workers == 0:
        warnings.warn("persistent_workers=True 이지만 num_workers=0 → 비활성화")
        persistent_train = False
    if persistent_test and num_workers == 0:
        warnings.warn("persistent_workers=True 이지만 num_workers=0 → 비활성화")
        persistent_test = False

    mp_ctx_train = (
        torch.multiprocessing.get_context("spawn")
        if persistent_train and num_workers > 0
        else None
    )
    mp_ctx_test = (
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

    if mp_ctx_train is not None:
        if "multiprocessing_context" in torch.utils.data.DataLoader.__init__.__code__.co_varnames:
            dl_kwargs_train["multiprocessing_context"] = mp_ctx_train
    if mp_ctx_test is not None:
        if "multiprocessing_context" in torch.utils.data.DataLoader.__init__.__code__.co_varnames:
            dl_kwargs_test["multiprocessing_context"] = mp_ctx_test

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        **dl_kwargs_train,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        **dl_kwargs_test,
    )
    return train_loader, test_loader
