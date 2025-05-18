# data/tinyimagenet.py

import os
import torch
import torchvision
import torchvision.transforms as T

def get_tinyimagenet_loaders(root, batch_size=128, num_workers=2):
    """
    Expecting a structure like:
      root/
        train/
          class_0/ images...
          class_1/ ...
          ...
        val/
          class_0/ ...
          ...
    """
    # e.g. basic transform
    transform_train = T.Compose([
        T.Resize((64,64)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4802, 0.4481, 0.3975),
                    (0.2302, 0.2265, 0.2262))
    ])
    transform_val = T.Compose([
        T.Resize((64,64)),
        T.ToTensor(),
        T.Normalize((0.4802, 0.4481, 0.3975),
                    (0.2302, 0.2265, 0.2262))
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
