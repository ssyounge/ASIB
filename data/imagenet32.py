# data/imagenet32.py

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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
            "train": [f"train_data_batch_{i}" for i in range(1, 11)],
            "val"  : ["val_data"],
            "test" : ["test_data"],
        }[split]

        self.data, self.labels = [], []
        for fn in fn_glob:
            fp = os.path.join(root, fn)
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
        self.classes = list(range(self.labels.max() + 1))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx]).float().div_(255.0)
        if self.transform is not None:
            img = self.transform(img)          # e.g. RandomFlip + ToTensor etc.
        target = int(self.labels[idx])         # already 0-based
        return img, target


def get_imagenet32_loaders(root, batch_size=128, num_workers=2):
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_test = transforms.ToTensor()

    tr_set = ImageNet32(root, "train", tf_train)
    te_set = ImageNet32(root, "val", tf_test)

    tr_loader = DataLoader(tr_set, batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True)
    te_loader = DataLoader(te_set, batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    return tr_loader, te_loader
