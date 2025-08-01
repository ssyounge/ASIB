# utils/freeze.py

import re
import torch.nn as nn

__all__ = ["freeze_all", "unfreeze_by_regex", "apply_bn_ln_policy", "apply_partial_freeze"]


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


def apply_partial_freeze(model, level: int, freeze_bn: bool = False):
    """Apply a simple partial freeze scheme to a model.

    Parameters
    ----------
    model : nn.Module
        Target network whose ``requires_grad`` flags will be updated.
    level : int
        level<0 → no‑freeze, level=0 → head만 학습,
        ``1`` unfreezes the last block and ``2`` the last two blocks.
    freeze_bn : bool, optional
        When ``True`` the BatchNorm parameters remain frozen.
    """
    if level < 0:
        # no-op: everything trainable
        for p in model.parameters():
            p.requires_grad = True
        return

    freeze_all(model)

    patterns = []
    if level == 0:
        patterns = [
            r"(?:^|\.)fc\.",
            r"(?:^|\.)classifier\.",
            r"(?:^|\.)head\.",
        ]
    elif level == 1:
        patterns = [
            r"\.layer4\.",
            r"features\.7\.",
            r"features\.8\.",
            r"\.layers\.3\.",
            r"(?:^|\.)fc\.",
            r"(?:^|\.)classifier\.",
            r"(?:^|\.)head\.",
        ]
    else:  # level >= 2
        patterns = [
            r"\.layer3\.",
            r"\.layer4\.",
            r"features\.6\.",
            r"features\.7\.",
            r"features\.8\.",
            r"\.layers\.2\.",
            r"\.layers\.3\.",
            r"(?:^|\.)fc\.",
            r"(?:^|\.)classifier\.",
            r"(?:^|\.)head\.",
        ]

    unfreeze_by_regex(model, patterns)

    apply_bn_ln_policy(model, train_bn=not freeze_bn)

