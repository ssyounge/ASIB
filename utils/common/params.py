# utils/params.py

import torch.nn as nn

__all__ = ["count_trainable", "count_trainable_parameters"]

def count_trainable(module: nn.Module) -> int:
    """Return the number of trainable parameters in ``module``."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def count_trainable_parameters(module: nn.Module) -> int:
    """Return the number of trainable parameters in ``module``."""
    return count_trainable(module)
