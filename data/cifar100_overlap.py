import random
import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR100

__all__ = ["get_overlap_loaders"]


def _split_classes(pct_overlap, seed=42):
    """Return two class lists with given overlap percentage."""
    rng = random.Random(seed)
    classes = list(range(100))
    rng.shuffle(classes)

    n_overlap = round(pct_overlap)
    shared = classes[:n_overlap]
    rem = classes[n_overlap:]

    half, extra = divmod(100 - n_overlap, 2)
    # extra(0 or 1)를 A 쪽에 먼저 배분해 100개 보존
    classes_A = shared + rem[: half + extra]
    classes_B = shared + rem[half + extra :]
    return classes_A, classes_B


def _make_loader(class_ids, train, batch_size, num_workers, augment):
    tr = [
        T.RandomCrop(32, 4),
        T.RandomHorizontalFlip(),
        T.RandAugment(),
    ] if augment and train else []
    tr += [
        T.ToTensor(),
        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ]
    ds = CIFAR100(root="./data", train=train, download=True, transform=T.Compose(tr))
    idx = [i for i, (_, y) in enumerate(ds) if y in class_ids]
    sub = torch.utils.data.Subset(ds, idx)
    return torch.utils.data.DataLoader(
        sub,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_overlap_loaders(pct_overlap=0, batch_size=128, num_workers=2, augment=True, seed=42):
    """Return loaders for two class subsets with given overlap."""
    cls_A, cls_B = _split_classes(pct_overlap, seed)
    A_tr = _make_loader(cls_A, True, batch_size, num_workers, augment)
    A_te = _make_loader(cls_A, False, batch_size, num_workers, False)
    B_tr = _make_loader(cls_B, True, batch_size, num_workers, augment)
    B_te = _make_loader(cls_B, False, batch_size, num_workers, False)
    return (A_tr, A_te), (B_tr, B_te), set(cls_A) & set(cls_B)
