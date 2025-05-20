import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as T

class ClassRemapSubset(data.Dataset):
    """
    1) 내부적으로 torch.utils.data.Subset처럼 일부 샘플만 택함
    2) 클래스 인덱스를 0~(num_subset-1) 범위로 재매핑
    """
    def __init__(self, base_dataset, subset_indices, old_to_new):
        """
        base_dataset: ImageFolder 등
        subset_indices: 샘플 인덱스 리스트
        old_to_new: dict (원본 class idx -> 새 class idx)
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.subset_indices = subset_indices
        self.old_to_new = old_to_new

    def __len__(self):
        return len(self.subset_indices)

    def __getitem__(self, i):
        real_i = self.subset_indices[i]
        path, old_class = self.base_dataset.samples[real_i]
        # (img, _) = base_dataset[real_i] 로도 가능
        img = self.base_dataset.loader(path)
        if self.base_dataset.transform is not None:
            img = self.base_dataset.transform(img)

        new_class = self.old_to_new[old_class]  # 재매핑
        return img, new_class

def make_tinyimagenet100_subset(base_dataset, num_classes=100):
    """
    base_dataset: ImageFolder(train or val)
    num_classes: 100개 클래스를 선택
                 (원본 200 클래스 중 사전순 앞 100개)
    Returns:
      subset_ds: ClassRemapSubset
    """
    # 1) 전체 클래스 이름(사전순)
    all_class_names = sorted(base_dataset.class_to_idx.keys())  # e.g. length=200
    chosen_classes = all_class_names[:num_classes]              # 앞 100개

    # 2) chosen_classes에 대응하는 old_class_idx 수집
    chosen_class_idx = set(base_dataset.class_to_idx[c] for c in chosen_classes)

    # 3) subset_indices: (sample 중에서) old_class_idx in chosen_class_idx
    subset_indices = []
    for i, (path, old_label) in enumerate(base_dataset.samples):
        if old_label in chosen_class_idx:
            subset_indices.append(i)

    # 4) old_to_new 재매핑 딕셔너리 만들기
    #    chosen_classes는 [class_name0, ..., class_name99] (사전순)
    #    new_label = 0..99 로 할당
    #    old_label = base_dataset.class_to_idx[class_name]
    old_to_new = {}
    for new_label, cname in enumerate(chosen_classes):
        old_label = base_dataset.class_to_idx[cname]
        old_to_new[old_label] = new_label

    # 5) ClassRemapSubset 생성
    subset_ds = ClassRemapSubset(base_dataset, subset_indices, old_to_new)
    return subset_ds

def get_tinyimagenet100_loaders(root="./data/tinyimagenet", batch_size=128, num_workers=2):
    """
    TinyImageNet 원본 (200 classes)에서
    사전순 상위 100개 클래스를 골라
    train/test => 100-class Subset DataLoader 생성
    """
    transform_train = T.Compose([
        T.Resize((64,64)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4802,0.4481,0.3975),
                    (0.2302,0.2265,0.2262))
    ])
    transform_test = T.Compose([
        T.Resize((64,64)),
        T.ToTensor(),
        T.Normalize((0.4802,0.4481,0.3975),
                    (0.2302,0.2265,0.2262))
    ])

    train_dir = os.path.join(root, "train")
    test_dir  = os.path.join(root, "val")

    full_train_dataset = torchvision.datasets.ImageFolder(
        train_dir,
        transform=transform_train
    )
    full_test_dataset = torchvision.datasets.ImageFolder(
        test_dir,
        transform=transform_test
    )

    # 100개 클래스만 서브셋
    train_100 = make_tinyimagenet100_subset(full_train_dataset, num_classes=100)
    test_100  = make_tinyimagenet100_subset(full_test_dataset, num_classes=100)

    train_loader = data.DataLoader(
        train_100,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = data.DataLoader(
        test_100,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, test_loader
