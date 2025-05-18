# data/imagenet100.py

import os
import torch
import torchvision
import torchvision.transforms as T

def get_imagenet100_loaders(root, batch_size=128, num_workers=4):
    """
    Assuming root has subfolders for the 100 classes,
    e.g. root/train/<class_x>/..., root/val/<class_x>/...
    """
    transform_train = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    transform_val = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(root, "train")
    val_dir   = os.path.join(root, "val")

    train_dataset = torchvision.datasets.ImageFolder(
        train_dir,
        transform=transform_train
    )
    val_dataset = torchvision.datasets.ImageFolder(
        val_dir,
        transform=transform_val
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader
