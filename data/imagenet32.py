# data/imagenet32.py

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from utils.data import ClassInfoMixin

class ImageNet32(ClassInfoMixin, Dataset):
    """
    Downsampled ImageNet-32x32 (Chrabaszcz et al.)
    – train_data_batch_{1..10}, valid_data, test_data  (Pickle)
    """
    def __init__(self, root, split="train", transform=None):
        root = root or os.getenv("DATA_ROOT", "./data")
        assert split in ("train", "val", "test")
        self.split, self.transform = split, transform

        fn_glob = {
            "train": [f"imagenet32_train_npz/train_data_batch_{i}.npz" for i in range(1, 11)],
            "val"  : ["imagenet32_val_npz/val_data.npz"],
            "test" : ["test_data"],
        }[split]

        self.data, self.labels = [], []
        for fn in fn_glob:
            fp = os.path.join(root, fn)
            if fn.endswith('.npz'):
                # NPZ 파일 처리
                with np.load(fp) as data:
                    self.data.append(data['data'])
                    self.labels += data['labels'].tolist()
            else:
                # 기존 pickle 파일 처리 (fallback)
                with open(fp, "rb") as f:
                    entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                self.labels += entry["labels"]
        # ------------------------------- #
        #  • data  : [N, 3, 32, 32] uint8
        #  • labels: 0-based int64
        #  • classes: List[int] (for len(...))
        # ------------------------------- #
        self.data   = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.labels = (np.array(self.labels, dtype=np.int64) - 1)
        # ClassInfoMixin에서 이미 classes 속성을 관리하므로 직접 설정하지 않음
        
        # Add num_classes for ClassInfoMixin compatibility
        self.num_classes = 1000  # ImageNet has 1000 classes

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx]).float().div_(255.0)
        if self.transform is not None:
            img = self.transform(img)          # e.g. RandomFlip etc.
        target = int(self.labels[idx])         # already 0-based
        return img, target


# Alias for compatibility
ImageNet32Dataset = ImageNet32


def get_imagenet32_loaders(root, batch_size=128, num_workers=2, augment=True):
    """
    ImageNet-32 size = (32x32) - 강화된 증강 적용
    Returns:
        train_loader, test_loader
    """
    # ImageNet-32용 강화된 증강 (CIFAR-100과 동일한 수준)
    if augment:
        tf_train = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),  # ImageNet은 더 작은 회전
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 적당한 색상 변화
            T.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # 작은 crop 변화
            T.RandomGrayscale(p=0.05),  # 낮은 확률의 그레이스케일
            # ImageNet-32용 정규화 (ImageNet 통계 사용)
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        tf_train = T.Compose([
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    tf_test = T.Compose([
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    tr_set = ImageNet32(root, "train", tf_train)
    te_set = ImageNet32(root, "val", tf_test)

    tr_loader = DataLoader(tr_set, batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True,
                           persistent_workers=True if num_workers > 0 else False,
                           prefetch_factor=4 if num_workers > 0 else None)
    te_loader = DataLoader(te_set, batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True,
                           persistent_workers=True if num_workers > 0 else False,
                           prefetch_factor=4 if num_workers > 0 else None)
    return tr_loader, te_loader
