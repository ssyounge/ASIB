# utils/freeze.py

import torch.nn as nn

__all__ = ["freeze_all"]


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
