# tests/test_cifar100_overlap.py

import pytest; pytest.importorskip("torch")
import torch
import torchvision
import importlib


def test_overlap_loaders(monkeypatch):
    cifar_mod = importlib.import_module("data.cifar100")
    if not hasattr(cifar_mod, "get_overlap_loaders"):
        pytest.skip("get_overlap_loaders not available")
    get_overlap_loaders = cifar_mod.get_overlap_loaders

    class DummyCIFAR100(torch.utils.data.Dataset):
        def __init__(self, train=True, transform=None):
            self.data = torch.randn(100, 3, 32, 32)
            self.targets = list(range(100))
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img = self.data[idx]
            if self.transform:
                img = self.transform(img)
            return img, self.targets[idx]

    monkeypatch.setattr(torchvision.datasets, "CIFAR100", DummyCIFAR100)

    loader1, loader2, cls_t1, cls_t2 = get_overlap_loaders(rho=0.2)

    overlap = set(cls_t1) & set(cls_t2)
    assert len(overlap) == 20

    labels1 = [y for _, y in loader1.dataset]
    labels2 = [y for _, y in loader2.dataset]
    assert set(labels1).issubset(set(cls_t1))
    assert set(labels2).issubset(set(cls_t2))
