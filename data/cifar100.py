# data/cifar100.py

import torch
import torchvision
import torchvision.transforms as T

def get_cifar100_loaders(batch_size=128, root="./data"):
    """
    가장 기본적인 CIFAR-100 DataLoader 생성:
    - train / test 각각 반환
    - 특별한 변환 없이, ToTensor + 기본 Normalize 정도만.
    (복잡한 증강/transform은 다른 파일에서 처리하거나 main.py에서 구성)

    Returns:
        train_loader, test_loader
    """
    # 예시) 아주 간단한 transform (바로 raw tensor)
    # 여기서는 최소한의 Normalize만 적용
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
        num_workers=2,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    return train_loader, test_loader
