# utils/freeze.py

from typing import Any
import torch.nn as nn

__all__ = [
    "freeze_all",
    "partial_freeze_teacher_auto",
    "partial_freeze_student_auto",
]


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze(module: Any) -> None:
    if module is not None:
        for p in getattr(module, "parameters", lambda: [])():
            p.requires_grad = True


def partial_freeze_teacher_auto(
    model: nn.Module,
    teacher_type: str,
    freeze_bn: bool = True,
    freeze_ln: bool = True,
    use_adapter: bool = False,
    bn_head_only: bool = False,
    freeze_level: int = 1,
) -> None:
    """Very simple partial freeze implementation.

    The original repository provides fine-grained control over which layers
    remain trainable. This stub freezes the whole backbone when ``freeze_level``
    is greater than ``0`` and leaves the classifier head trainable.
    """
    if freeze_level <= 0:
        return

    freeze_all(model)
    backbone = getattr(model, "backbone", model)
    if teacher_type.startswith("resnet"):
        _unfreeze(getattr(backbone, "fc", None))
    elif teacher_type.startswith("efficientnet"):
        _unfreeze(getattr(backbone, "classifier", None))


def partial_freeze_student_auto(
    model: nn.Module,
    student_name: str,
    freeze_bn: bool = True,
    freeze_ln: bool = True,
    use_adapter: bool = False,
    freeze_level: int = 1,
) -> None:
    """Simplified partial freeze for students."""
    if freeze_level <= 0:
        return

    freeze_all(model)
    backbone = getattr(model, "backbone", model)
    if student_name.startswith("convnext"):
        classifier = getattr(backbone, "classifier", None)
        if isinstance(classifier, (nn.Sequential, list)):
            _unfreeze(classifier[-1])
        else:
            _unfreeze(classifier)

