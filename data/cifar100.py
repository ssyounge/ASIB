# data/cifar100.py

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
    persistent: bool = False,
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

    mp_ctx = (
        torch.multiprocessing.get_context("spawn")
        if persistent and num_workers > 0
        else None
    )

    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent and num_workers > 0,
    )

    if mp_ctx is not None:
        # ``multiprocessing_context`` is only available on newer PyTorch versions
        if "multiprocessing_context" in torch.utils.data.DataLoader.__init__.__code__.co_varnames:
            dl_kwargs["multiprocessing_context"] = mp_ctx

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        **dl_kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        **dl_kwargs,
    )
    return train_loader, test_loader
