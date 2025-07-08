import pytest; pytest.importorskip("torch")
import torch
from data.cifar100_cl import set_class_order, get_cifar100_cl_loaders


def test_class_split_consistency():
    set_class_order(list(range(100)))
    train, cur, seen = get_cifar100_cl_loaders(n_tasks=10, task_id=3, batch_size=4)
    # 4-th task -> 클래스 30~39
    labels = [y for _, y in cur.dataset]
    assert min(labels) >= 30 and max(labels) <= 39
    assert len(seen.dataset) > len(cur.dataset)

