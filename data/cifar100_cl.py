"""CIFAR-100 continual-learning data loader utilities."""

import torch
import torchvision
import torchvision.transforms as T


def _split_dataset(dataset, class_ids):
    idx = [i for i, t in enumerate(dataset.targets) if t in class_ids]
    return torch.utils.data.Subset(dataset, idx)


def get_cifar100_cl_loaders(
    root: str = "./data",
    task_id: int = 0,
    n_tasks: int = 10,
    batch_size: int = 128,
    num_workers: int = 2,
    randaug_N: int = 0,
    randaug_M: int = 0,
    persistent_train: bool = False,
):
    """Return train/test loaders for a single CIFAR-100 incremental task."""
    assert 0 <= task_id < n_tasks, "invalid task_id"
    classes_per = 100 // n_tasks
    start = task_id * classes_per
    end = start + classes_per
    class_ids = list(range(start, end))

    ops = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    if randaug_N > 0 and randaug_M > 0:
        ops.append(T.RandAugment(num_ops=randaug_N, magnitude=randaug_M))
    ops.extend([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    transform_train = T.Compose(ops)

    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    base_train = torchvision.datasets.CIFAR100(
        root=root, train=True, download=True, transform=transform_train
    )
    base_test = torchvision.datasets.CIFAR100(
        root=root, train=False, download=True, transform=transform_test
    )

    train_ds = _split_dataset(base_train, class_ids)
    test_ds = _split_dataset(base_test, class_ids)

    mp_ctx = (
        torch.multiprocessing.get_context("spawn")
        if persistent_train and num_workers > 0
        else None
    )

    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        persistent_workers=persistent_train and num_workers > 0,
    )
    if mp_ctx is not None and "multiprocessing_context" in torch.utils.data.DataLoader.__init__.__code__.co_varnames:
        dl_kwargs["multiprocessing_context"] = mp_ctx

    train_loader = torch.utils.data.DataLoader(train_ds, **dl_kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader

