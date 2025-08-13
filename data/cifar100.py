# data/cifar100.py

import torch
import os
import logging
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class CustomToTensor:
    """Custom ToTensor transform to avoid numpy compatibility issues"""
    def __call__(self, pic):
        # PIL Image를 직접 torch tensor로 변환
        if hasattr(pic, 'convert'):
            # PIL Image인 경우
            pic = pic.convert('RGB')
            # PIL을 직접 torch tensor로 변환
            img = torch.tensor(np.array(pic), dtype=torch.uint8).permute((2, 0, 1)).contiguous()
            return img.float().div(255)
        else:
            # numpy array인 경우
            img = torch.from_numpy(pic.transpose((2, 0, 1))).float()
            return img / 255.0


class CIFAR100Dataset(Dataset):
    """CIFAR100 Dataset wrapper for consistency with other datasets."""
    
    def __init__(self, root="./data", train=True, download=True, transform=None):
        """
        Initialize CIFAR100 dataset.
        
        Parameters:
        -----------
        root : str
            Root directory for dataset
        train : bool
            If True, use training set, else use test set
        download : bool
            If True, download dataset if not present
        transform : callable, optional
            Optional transform to be applied on a sample
        """
        self.dataset = torchvision.datasets.CIFAR100(
            root=root,
            train=train,
            download=download,
            transform=transform
        )
        self.train = train
        
    def __getitem__(self, index):
        """Get item by index."""
        return self.dataset[index]
    
    def __len__(self):
        """Return dataset length."""
        return len(self.dataset)


class CIFAR100NPZ(Dataset):
    """CIFAR100 Dataset using NPZ files (ImageNet-32 형식과 동일)"""
    
    def __init__(self, root="./data", train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        
        # Add required attributes for main.py compatibility
        self.classes = list(range(100))  # CIFAR-100 has 100 classes
        self.num_classes = 100
        
        # NPZ 파일 로드 (ImageNet-32와 동일한 형식)
        if self.train:
            # 훈련 데이터: 10개 배치 파일 로드
            self.data = []
            self.targets = []
            
            for i in range(1, 11):  # train_data_batch_1.npz ~ train_data_batch_10.npz
                file_path = os.path.join(self.root, f'cifar100_train_npz/train_data_batch_{i}.npz')
                with np.load(file_path) as batch_data:
                    batch_images = batch_data['data']  # (N, 32, 32, 3)
                    batch_labels = batch_data['labels']
                    
                    self.data.append(batch_images)
                    self.targets.extend(batch_labels)
            
            # 모든 배치 데이터를 하나로 합치기
            self.data = np.concatenate(self.data, axis=0)  # (50000, 32, 32, 3)
            
        else:
            # 테스트 데이터: 단일 파일 로드
            file_path = os.path.join(self.root, 'cifar100_val_npz/val_data.npz')
            with np.load(file_path) as data:
                self.data = data['data']  # (10000, 32, 32, 3)
                self.targets = data['labels']
        
    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        
        # numpy array를 PIL Image로 변환
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target
    
    def __len__(self):
        return len(self.data)


def get_cifar100_loaders(root="./data", batch_size=128, num_workers=2, augment=True):
    """
    CIFAR-100 size = (32x32) - NPZ 파일 사용 (ImageNet-32와 동일한 형식)
    Returns:
        train_loader, test_loader
    """
    # 과대적합 방지를 위한 더 강한 증강
    if augment:
        transform_train = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=20),  # 더 큰 회전
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),  # 더 강한 색상 변화
            T.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(0.8, 1.2)),  # 더 큰 crop 변화
            T.RandomGrayscale(p=0.1),  # 그레이스케일 변환 추가
            CustomToTensor(),
            T.Normalize((0.5071,0.4865,0.4409),
                        (0.2673,0.2564,0.2762))
        ])
    else:
        transform_train = T.Compose([
            CustomToTensor(),
            T.Normalize((0.5071,0.4865,0.4409),
                        (0.2673,0.2564,0.2762))
        ])
    
    transform_test = T.Compose([
        CustomToTensor(),
        T.Normalize((0.5071,0.4865,0.4409),
                    (0.2673,0.2564,0.2762))
    ])

    root = root or os.getenv("DATA_ROOT", "./data")
    train_dataset = CIFAR100NPZ(root=root, train=True, transform=transform_train)
    test_dataset = CIFAR100NPZ(root=root, train=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # GPU 전송 속도 향상
        persistent_workers=True if num_workers > 0 else False,  # 워커 재사용으로 속도 향상
        prefetch_factor=4 if num_workers > 0 else None  # 미리 로드할 배치 수
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,  # GPU 전송 속도 향상
        persistent_workers=True if num_workers > 0 else False,  # 워커 재사용으로 속도 향상
        prefetch_factor=4 if num_workers > 0 else None  # 미리 로드할 배치 수
    )
    return train_loader, test_loader

# —— Quick sanity check (optional) ————————————
#   python -m data.cifar100
if __name__ == "__main__":  # noqa: D401
    tr, te = get_cifar100_loaders(batch_size=256, augment=False)
    ys = [y for _, y in tr.dataset]
    logging.basicConfig(level=logging.DEBUG)
    logging.debug("[DBG] CIFAR-100 label range: %s %s", min(ys), max(ys))
    logging.debug(
        "[DBG] train len = %s test len = %s", len(tr.dataset), len(te.dataset)
    )
