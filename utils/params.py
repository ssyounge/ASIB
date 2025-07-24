import torch.nn as nn

__all__ = ["count_trainable"]

def count_trainable(module: nn.Module) -> int:
    """Return the number of trainable parameters in ``module``."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
