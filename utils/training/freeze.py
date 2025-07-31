# utils/freeze.py

import re
import torch.nn as nn

__all__ = ["freeze_all", "unfreeze_by_regex", "apply_bn_ln_policy"]


def freeze_all(model: nn.Module) -> None:
    """Set ``requires_grad=False`` for all parameters."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_by_regex(model: nn.Module, patterns) -> None:
    """Enable grads for parameters whose names match any regex pattern."""
    if isinstance(patterns, str):
        patterns = [patterns]
    compiled = [re.compile(p) for p in patterns]
    for name, param in model.named_parameters():
        if any(c.search(name) for c in compiled):
            param.requires_grad = True


def apply_bn_ln_policy(
    model: nn.Module,
    train_bn: bool = True,
    train_ln: bool = True,
) -> None:
    """Toggle BatchNorm and LayerNorm parameter training."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            for p in module.parameters():
                p.requires_grad = train_bn
        if isinstance(module, nn.LayerNorm):
            for p in module.parameters():
                p.requires_grad = train_ln

