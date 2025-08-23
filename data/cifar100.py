# data/cifar100.py

import torch
import os
import logging
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from multiprocessing import get_context


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
        # main.py 호환을 위한 클래스 정보 노출 (길이만 사용)
        self.num_classes = 100
        self.classes = list(range(100))
        
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
                    batch_images = batch_data['data']  # 기대: (N, 32, 32, 3)
                    batch_labels = batch_data['labels']  # 기대: (N,) or (N,1)
                    
                    self.data.append(batch_images)
                    # 라벨은 배치 단위로 모아두었다가 최종적으로 concatenate
                    self.targets.append(batch_labels)
            
            # 모든 배치 데이터를 하나로 합치고 라벨은 1D int64로 강제
            self.data = np.concatenate(self.data, axis=0)  # (50000, 32, 32, 3)
            self.targets = (
                np.concatenate(self.targets, axis=0)
                .reshape(-1)
                .astype(np.int64)
                .tolist()
            )
            
        else:
            # 테스트 데이터: 단일 파일 로드
            file_path = os.path.join(self.root, 'cifar100_val_npz/val_data.npz')
            with np.load(file_path) as data:
                self.data = data['data']  # (10000, 32, 32, 3)
                self.targets = data['labels']
            # 라벨 1D int64 강제
            self.targets = (
                np.array(self.targets)
                .reshape(-1)
                .astype(np.int64)
                .tolist()
            )
        
    def __getitem__(self, index):
        img = self.data[index]
        target = int(self.targets[index])  # 보장: 0..99의 int
        
        # numpy array를 PIL Image로 변환 (안전 가드 포함)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if img.ndim == 3 and img.shape[-1] == 3:
            pil = Image.fromarray(img)
        elif img.ndim == 3 and img.shape[0] == 3:
            pil = Image.fromarray(np.transpose(img, (1, 2, 0)))
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        
        if self.transform is not None:
            pil = self.transform(pil)
            
        return pil, target
    
    def __len__(self):
        return len(self.data)


def get_cifar100_loaders(
    root="./data",
    batch_size=128,
    num_workers=2,
    augment=True,
    use_spawn_dl: bool = False,
    backend: str = "npz",
    log_first_batch_stats: bool = False,
):
    """
    CIFAR-100 size = (32x32) - NPZ 파일 사용 (ImageNet-32와 동일한 형식)
    Returns:
        train_loader, test_loader
    """
    # 표준 CIFAR-100 증강(안정 수렴 우선): RandomCrop(32, pad=4) + Flip
    if augment:
        transform_train = T.Compose([
            T.RandomCrop(32, padding=4, padding_mode='reflect'),
            T.RandomHorizontalFlip(p=0.5),
            CustomToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    else:
        transform_train = T.Compose([
            CustomToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    
    transform_test = T.Compose([
        CustomToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    root = root or os.getenv("DATA_ROOT", "./data")
    backend = str(backend or "npz").lower()
    if backend == "torchvision":
        train_dataset = CIFAR100Dataset(root=root, train=True, download=True, transform=transform_train)
        test_dataset = CIFAR100Dataset(root=root, train=False, download=True, transform=transform_test)
    else:
        train_dataset = CIFAR100NPZ(root=root, train=True, transform=transform_train)
        test_dataset = CIFAR100NPZ(root=root, train=False, transform=transform_test)

    pin = bool(num_workers > 0)
    mp_ctx = get_context("spawn") if (num_workers > 0 and bool(use_spawn_dl)) else None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=pin,
        prefetch_factor=(2 if pin else None),
        multiprocessing_context=mp_ctx,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=pin,
        prefetch_factor=(2 if pin else None),
        multiprocessing_context=mp_ctx,
    )

    # Optional: one-time first-batch statistics for quick sanity
    if log_first_batch_stats:
        try:
            xb, yb = next(iter(train_loader))
            logging.info(
                "x %s %s min=%.3f max=%.3f mean=%.3f std=%.3f | y %s [%d,%d]",
                tuple(xb.shape), xb.dtype,
                float(xb.min()), float(xb.max()), float(xb.mean()), float(xb.std()),
                yb.dtype, int(yb.min()), int(yb.max()),
            )
        except Exception as _e:
            logging.debug("first-batch stats logging skipped: %s", _e)
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
