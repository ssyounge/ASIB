import torch
from torchvision.transforms.functional import to_tensor

__all__ = ["SafeToTensor"]

class SafeToTensor:
    """Convert PIL image or ndarray to tensor; pass through tensors."""

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            if img.dtype == torch.uint8:
                img = img.float().div_(255.0)
            return img
        return to_tensor(img)
