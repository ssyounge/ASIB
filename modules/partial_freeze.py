"""Lightweight partial freeze utilities for teacher fine-tuning."""
import torch.nn as nn
from typing import Any

__all__ = [
    "freeze_all",
    "partial_freeze_teacher_resnet",
    "partial_freeze_teacher_efficientnet",
    "partial_freeze_teacher_swin",
]


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze(module: Any) -> None:
    if module is not None:
        for p in getattr(module, "parameters", lambda: [])():
            p.requires_grad = True


def partial_freeze_teacher_resnet(
    model: nn.Module,
    freeze_bn: bool = True,
    use_adapter: bool = False,
    bn_head_only: bool = False,
    freeze_level: int = 1,
) -> None:
    """Freeze the ResNet backbone except for the classifier."""
    if freeze_level <= 0:
        return
    freeze_all(model)
    backbone = getattr(model, "backbone", model)
    _unfreeze(getattr(backbone, "fc", None))


def partial_freeze_teacher_efficientnet(
    model: nn.Module,
    freeze_bn: bool = True,
    use_adapter: bool = False,
    bn_head_only: bool = False,
    freeze_level: int = 1,
) -> None:
    """Freeze EfficientNet backbone except for the classifier."""
    if freeze_level <= 0:
        return
    freeze_all(model)
    backbone = getattr(model, "backbone", model)
    _unfreeze(getattr(backbone, "classifier", None))


def partial_freeze_teacher_swin(
    model: nn.Module,
    freeze_ln: bool = True,
    use_adapter: bool = False,
    freeze_level: int = 1,
) -> None:
    """Freeze Swin backbone except for the head."""
    if freeze_level <= 0:
        return
    freeze_all(model)
    backbone = getattr(model, "backbone", model)
    _unfreeze(getattr(backbone, "head", None))
