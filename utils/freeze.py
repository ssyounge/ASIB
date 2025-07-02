# utils/freeze.py

from typing import Any
import torch.nn as nn

__all__ = ["freeze_all"]


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze(module: Any) -> None:
    if module is not None:
        for p in getattr(module, "parameters", lambda: [])():
            p.requires_grad = True



