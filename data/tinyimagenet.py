import os
import torch
import torchvision
import torchvision.transforms as T

def get_tinyimagenet_loaders(root="./data/tinyimagenet", batch_size=128, num_workers=2):
    """
    TinyImageNet size = (64×64)
    Returns:
        train_loader, test_loader
    """
    transform_train = T.Compose([
        T.Resize((64,64)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4802, 0.4481, 0.3975),
                    (0.2302, 0.2265, 0.2262))
    ])
    transform_test = T.Compose([
        T.Resize((64,64)),
        T.ToTensor(),
        T.Normalize((0.4802, 0.4481, 0.3975),
                    (0.2302, 0.2265, 0.2262))
    ])

    train_dir = os.path.join(root, "train")
    test_dir  = os.path.join(root, "val")  # rename "val" to "test" concept

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
