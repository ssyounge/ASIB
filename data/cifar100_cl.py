"""CIFAR-100 continual-learning data loader utilities."""
# data/cifar100_cl.py

import torch
import torchvision
import torchvision.transforms as T
from utils.transform_utils import SafeToTensor
from functools import lru_cache

# ------------------------------------------------------------------
# 전역 class-order  (seed 고정 시 재현성 보장)
# ------------------------------------------------------------------
_CLASS_ORDER = list(range(100))          # 0-99 기본

# 필요하면 main 에서 set_class_order([...]) 호출
def set_class_order(order):
    assert len(order) == 100 and len(set(order)) == 100
    global _CLASS_ORDER
    _CLASS_ORDER = list(order)
    # ─ 새 class-order 적용 시 기존 캐시 무효화 ─
    _task_classes.cache_clear()

# task_id → class id list 캐싱  (IO-free)
@lru_cache(maxsize=32)
def _task_classes(task_id: int, n_tasks: int):
    assert 100 % n_tasks == 0, "n_tasks must divide 100"
    per = 100 // n_tasks
    st = task_id * per
    return _CLASS_ORDER[st : st + per]


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
    *,
    return_task_split: bool = False,
    buffer_size: int = 20,
):
    """Return train/test loaders for a single CIFAR-100 incremental task."""
    assert 0 <= task_id < n_tasks, "invalid task_id"
    class_ids = _task_classes(task_id, n_tasks)

    ops = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    if randaug_N > 0 and randaug_M > 0:
        ops.append(T.RandAugment(num_ops=randaug_N, magnitude=randaug_M))
    ops.extend([
        SafeToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    transform_train = T.Compose(ops)

    transform_test = T.Compose([
        SafeToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    base_train = torchvision.datasets.CIFAR100(
        root=root, train=True, download=True, transform=transform_train
    )
    base_test = torchvision.datasets.CIFAR100(
        root=root, train=False, download=True, transform=transform_test
    )

    if return_task_split:
        dataset = base_train
        cur_idx = [i for i, t in enumerate(dataset.targets) if t in class_ids]
        prev_cls = sum((_task_classes(t, n_tasks) for t in range(task_id)), [])
        rep_idx = []
        for c in prev_cls:
            idx_c = [i for i, t in enumerate(dataset.targets) if t == c]
            rep_idx.extend(idx_c[:buffer_size])
        task_split = {"cur_indices": cur_idx, "replay_indices": rep_idx}
        train_ds = dataset
    else:
        train_ds = _split_dataset(base_train, class_ids)
    # ① 현재 task-only
    cur_test  = _split_dataset(base_test, class_ids)
    # ② 지금까지 등장한 모든 class
    seen_cls  = sum((_task_classes(t, n_tasks) for t in range(task_id + 1)), [])
    seen_test = _split_dataset(base_test, seen_cls)

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

    # (cur-only, cumulative) 두 개 모두 반환
    test_loader_cur = torch.utils.data.DataLoader(
        cur_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader_seen = torch.utils.data.DataLoader(
        seen_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    if return_task_split:
        return train_loader, test_loader_cur, test_loader_seen, train_ds, task_split
    return train_loader, test_loader_cur, test_loader_seen


def get_balanced_loader(
    task_id: int,
    n_tasks: int,
    buffer_size: int = 20,
    batch_size: int = 128,
    num_workers: int = 2,
    root: str = "./data",
):
    """Return a class-balanced loader using replay buffer and current data."""
    ops = [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        SafeToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ]
    transform = T.Compose(ops)

    base_train = torchvision.datasets.CIFAR100(
        root=root, train=True, download=True, transform=transform
    )

    seen_cls = sum((_task_classes(t, n_tasks) for t in range(task_id + 1)), [])
    indices = []
    for c in seen_cls:
        idx_c = [i for i, t in enumerate(base_train.targets) if t == c]
        if c in _task_classes(task_id, n_tasks):
            indices.extend(idx_c)
        else:
            indices.extend(idx_c[:buffer_size])

    subset = torch.utils.data.Subset(base_train, indices)
    targets = [base_train.targets[i] for i in indices]
    from collections import Counter
    counts = Counter(targets)
    weights = [1.0 / counts[t] for t in targets]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True
    )
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader

