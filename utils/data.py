"""Utility helpers for continual-learning datasets."""

from typing import List, Tuple

import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms as T


def get_split_cifar100_loaders(
    num_tasks: int, batch_size: int, augment: bool = True
) -> List[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]]:
    """Split CIFAR100 into `num_tasks` tasks of 10 classes each."""

    transform_train = [T.ToTensor()]
    if augment:
        transform_train = [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            *transform_train,
        ]
    transform_train = T.Compose(transform_train)
    transform_test = T.Compose([T.ToTensor()])

    full_train = CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    full_test = CIFAR100(root="./data", train=False, download=True, transform=transform_test)

    task_loaders = []
    classes_per_task = len(full_train.classes) // num_tasks
    for t in range(num_tasks):
        cls_start = t * classes_per_task
        cls_end = cls_start + classes_per_task
        cls_range = list(range(cls_start, cls_end))

        train_indices = [i for i, (_, y) in enumerate(full_train) if y in cls_range]
        test_indices = [i for i, (_, y) in enumerate(full_test) if y in cls_range]

        train_subset = torch.utils.data.Subset(full_train, train_indices)
        test_subset = torch.utils.data.Subset(full_test, test_indices)

        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_subset, batch_size=batch_size, shuffle=False
        )
        task_loaders.append((train_loader, test_loader))

    return task_loaders

