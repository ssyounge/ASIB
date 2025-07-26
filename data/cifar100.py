# data/cifar100.py

import torch
import os
import torchvision
import torchvision.transforms as T

def get_cifar100_loaders(root="./data", batch_size=128, num_workers=2, augment=True):
    """
    CIFAR-100 size = (32x32)
    Returns:
        train_loader, test_loader
    """
    if augment:
        transform_train = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.RandAugment(),
            T.ToTensor(),
            T.Normalize((0.5071,0.4865,0.4409),
                        (0.2673,0.2564,0.2762))
        ])
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

    root = root or os.getenv("DATA_ROOT", "./data")
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

# —— Quick sanity check (optional) ————————————
#   python -m data.cifar100
if __name__ == "__main__":        # noqa: D401
    tr, te = get_cifar100_loaders(batch_size=256, augment=False)
    ys = [y for _, y in tr.dataset]
    print("[DBG] CIFAR-100 label range:", min(ys), max(ys))
    print("[DBG] train len =", len(tr.dataset), "test len =", len(te.dataset))
