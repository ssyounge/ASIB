import os, pickle, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageNet32(Dataset):
    """
    Downsampled ImageNet-32x32 (Chrabaszcz et al.)
    – train_data_batch_{1..10}, valid_data, test_data  (Pickle)
    """
    def __init__(self, root, split="train", transform=None):
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
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img = torch.tensor(self.data[idx], dtype=torch.uint8)
        img = img.float() / 255.0
        if self.transform:
            img = self.transform(img)
        target = self.labels[idx] - 1
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
