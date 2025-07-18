# utils/transform_utils.py

import torch
import torchvision.transforms.functional as F
from torchvision.transforms.functional import to_tensor

__all__ = ["SafeToTensor", "EnsurePIL"]

class SafeToTensor:
    """Convert PIL image or ndarray to tensor; pass through tensors."""

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            if img.dtype == torch.uint8:
                img = img.float().div_(255.0)
            return img
        return to_tensor(img)


class EnsurePIL:
    """Return PIL images for tensor inputs."""

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            if img.dtype == torch.float32:
                img = (img * 255).clamp(0, 255).to(torch.uint8)
            return F.to_pil_image(img)
        return img
