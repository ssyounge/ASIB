import pickle
import numpy as np
import torch
from data.imagenet32 import ImageNet32


def test_imagenet32_getitem(tmp_path):
    data = np.random.randint(0, 256, size=(1, 3, 32, 32), dtype=np.uint8)
    labels = [1]
    entry = {"data": data, "labels": labels}
    with open(tmp_path / "val_data", "wb") as f:
        pickle.dump(entry, f)
    ds = ImageNet32(str(tmp_path), split="val", transform=None)
    img, target = ds[0]
    assert img.shape == (3, 32, 32)
    assert img.dtype == torch.float32
    assert target == 0
