# data/imagenet100.py

import os
from typing import Mapping, Any, Optional
import torch
import torchvision
import torchvision.transforms as T

def get_imagenet100_loaders(
    root: str = "./data/imagenet100",
    batch_size: int = 128,
    num_workers: int = 2,
    augment: bool = True,
    randaug_N: int = 0,
    randaug_M: int = 0,
    cfg: Optional[Mapping[str, Any]] = None,
    randaug_default_N: int = 2,
    randaug_default_M: int = 9,
):
    """
    ImageNet100 size = (224×224)
    Returns:
        train_loader, test_loader
    """
    if augment:
        aug_ops = [T.RandomResizedCrop(224), T.RandomHorizontalFlip()]
        if cfg is not None:
            randaug_default_N = cfg.get("randaug_default_N", randaug_default_N)
            randaug_default_M = cfg.get("randaug_default_M", randaug_default_M)
        if randaug_N > 0 and randaug_M > 0:
            aug_ops.append(T.RandAugment(num_ops=randaug_N, magnitude=randaug_M))
        else:
            aug_ops.append(T.RandAugment(num_ops=randaug_default_N, magnitude=randaug_default_M))
        aug_ops.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transform_train = T.Compose(aug_ops)
    else:
        transform_train = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
    transform_test = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(root, "train")
    test_dir  = os.path.join(root, "val")  # originally val, rename to test

    train_dataset = torchvision.datasets.ImageFolder(
        train_dir,
        transform=transform_train
    )
    test_dataset = torchvision.datasets.ImageFolder(
        test_dir,
        transform=transform_test
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, test_loader
