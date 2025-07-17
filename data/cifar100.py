# data/cifar100.py

import warnings
from torchvision.datasets import CIFAR100
import os, torch, torchvision.transforms as T
from typing import Mapping, Any, Optional

# ──────────────────────────────────────────────────────────────
# ①  Teacher-cache 를 포함한 전용 Dataset
#    (cache_path 가 None 이면 정상 CIFAR100 과 동일하게 동작)
# ──────────────────────────────────────────────────────────────
class CIFAR100Cached(CIFAR100):
    def __init__(self, *args,
                 cache_path: str | None = None,
                 cache_items: list[str] | None = None,
                 **kw):
        super().__init__(*args, **kw)
        self.cache = None
        if cache_path and os.path.isfile(cache_path):
            self.cache = torch.load(cache_path, map_location="cpu")
            if cache_items:                       # 불필요한 key 제거
                keep = set(cache_items)
                self.cache = {k: v for k, v in self.cache.items() if k in keep}

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if self.transform:
            img = self.transform(img)
        if self.cache is None:
            return img, target                    # ↩︎ 기존과 동일 (2‑tuple)
        sample_cache = {k: v[index] for k, v in self.cache.items()}
        return img, target, sample_cache          # ↩︎ (3‑tuple) 로 반환


# ──────────────────────────────────────────────────────────────
# ②  기존 get_cifar100_loaders 의 시그니처에 cache 인수 2개 추가
#    (호출부 diff 는 main.py 에 있음)
# ──────────────────────────────────────────────────────────────

def get_cifar100_loaders(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
    augment: bool = True,
    randaug_N: int = 0,
    randaug_M: int = 0,
    cfg: Optional[Mapping[str, Any]] = None,
    randaug_default_N: int = 2,
    randaug_default_M: int = 9,
    persistent_train: bool = False,
    persistent_test: Optional[bool] = None,
    cache_path: str | None = None,
    cache_items: list[str] | None = None,
):
    """
    CIFAR-100 size = (32x32)
    Returns:
        train_loader, test_loader
    """
    if augment:
        aug_ops = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
        if cfg is not None:
            randaug_default_N = cfg.get("randaug_default_N", randaug_default_N)
            randaug_default_M = cfg.get("randaug_default_M", randaug_default_M)
        if randaug_N > 0 and randaug_M > 0:
            aug_ops.append(T.RandAugment(num_ops=randaug_N, magnitude=randaug_M))
        else:
            aug_ops.append(T.RandAugment(num_ops=randaug_default_N, magnitude=randaug_default_M))
        aug_ops.extend([
            T.ToTensor(),
            T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
        transform_train = T.Compose(aug_ops)
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

    if cache_path:
        train_dataset = CIFAR100Cached(
            root=root, train=True, download=True, transform=transform_train,
            cache_path=cache_path, cache_items=cache_items)
    else:
        train_dataset = CIFAR100(
            root=root, train=True, download=True, transform=transform_train
        )
    test_dataset = CIFAR100(
        root=root,
        train=False,
        download=True,
        transform=transform_test
    )

    if persistent_test is None:
        persistent_test = persistent_train

    if persistent_train and num_workers == 0:
        warnings.warn("persistent_workers=True 이지만 num_workers=0 → 비활성화")
        persistent_train = False
    if persistent_test and num_workers == 0:
        warnings.warn("persistent_workers=True 이지만 num_workers=0 → 비활성화")
        persistent_test = False

    mp_ctx_train = (
        torch.multiprocessing.get_context("spawn")
        if persistent_train and num_workers > 0
        else None
    )
    mp_ctx_test = (
        torch.multiprocessing.get_context("spawn")
        if persistent_test and num_workers > 0
        else None
    )

    dl_kwargs_train = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_train and num_workers > 0,
    )
    dl_kwargs_test = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_test and num_workers > 0,
    )

    if mp_ctx_train is not None:
        if "multiprocessing_context" in torch.utils.data.DataLoader.__init__.__code__.co_varnames:
            dl_kwargs_train["multiprocessing_context"] = mp_ctx_train
    if mp_ctx_test is not None:
        if "multiprocessing_context" in torch.utils.data.DataLoader.__init__.__code__.co_varnames:
            dl_kwargs_test["multiprocessing_context"] = mp_ctx_test

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        **dl_kwargs_train,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        **dl_kwargs_test,
    )
    return train_loader, test_loader
