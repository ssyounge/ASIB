# modules/partial_freeze.py

import torch
import torch.nn as nn

from utils.freeze import apply_bn_ln_policy, freeze_all, unfreeze_by_regex


def freeze_bn_params(module: nn.Module):
    """Freeze the affine parameters of any BatchNorm modules.

    Call via ``model.apply(freeze_bn_params)`` to disable training of the
    gamma/beta parameters.
    """
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        for p in module.parameters():
            p.requires_grad = False


def freeze_ln_params(module: nn.Module):
    """Freeze the affine parameters of LayerNorm modules.

    Used by architectures such as Swin/ViT to keep the gamma/beta
    parameters fixed during training.
    """
    if isinstance(module, nn.LayerNorm):
        for p in module.parameters():
            p.requires_grad = False


def partial_freeze_teacher_resnet(
    model: nn.Module,
    freeze_bn: bool = True,
    use_adapter: bool = False,
    bn_head_only: bool = False,
    freeze_level: int = 1,
):
    """Partially freeze a ResNet101 teacher using a numeric level.

    ``freeze_all`` is called first and layers are selectively unfrozen based on
    ``freeze_level``:

    - ``0`` → only ``fc``
    - ``1`` → ``fc`` + ``mbm`` (default)
    - ``2`` → ``layer4`` + ``fc`` + ``mbm``
    """
    freeze_all(model)

    if bn_head_only:
        unfreeze_by_regex(model, r"^backbone\.fc\.")
        apply_bn_ln_policy(model, train_bn=True)
        return

    patterns = []
    if freeze_level == 0:
        patterns.append(r"^backbone\.fc\.")
    elif freeze_level == 2:
        patterns.extend(
            [
                r"^backbone\.layer4\.",
                r"^backbone\.fc\.",
                r"^mbm\.",
            ]
        )
    else:
        patterns.extend([r"^backbone\.fc\.", r"^mbm\."])

    if use_adapter:
        patterns.append(r"\.adapter_")

    unfreeze_by_regex(model, patterns)

    apply_bn_ln_policy(model, train_bn=not freeze_bn)


def partial_freeze_teacher_efficientnet(
    model: nn.Module,
    freeze_bn: bool = True,
    use_adapter: bool = False,
    bn_head_only: bool = False,
    freeze_level: int = 1,
):
    """Partially freeze an EfficientNet-B2 teacher using a numeric level.

    ``freeze_all`` is called first and layers are unfrozen based on
    ``freeze_level``:

    - ``0`` or ``3`` → only ``classifier``
    - ``1`` → ``classifier`` + ``mbm`` (default)
    - ``2`` → ``features`` + ``classifier`` + ``mbm``
    """
    freeze_all(model)

    if bn_head_only:
        unfreeze_by_regex(model, r"^backbone\.classifier\.")
        apply_bn_ln_policy(model, train_bn=True)
        return

    patterns = []
    if freeze_level in (0, 3):
        patterns.append(r"^backbone\.classifier\.")
    elif freeze_level == 2:
        patterns.extend(
            [
                r"^backbone\.features\.",
                r"^backbone\.classifier\.",
                r"^mbm\.",
            ]
        )
    else:
        patterns.extend([r"^backbone\.classifier\.", r"^mbm\."])

    if use_adapter:
        patterns.append(r"\.adapter_")

    unfreeze_by_regex(model, patterns)

    apply_bn_ln_policy(model, train_bn=not freeze_bn)


def partial_freeze_teacher_swin(
    model: nn.Module,
    freeze_ln: bool = True,
    use_adapter: bool = False,
    freeze_level: int = 1,
):
    """Partially freeze a Swin Tiny teacher using a numeric level.

    ``freeze_all`` is called first. ``freeze_level`` controls what to unfreeze:

    - ``0`` → only ``head``
    - ``1`` → ``head`` + ``mbm`` (default)
    """
    freeze_all(model)

    patterns = []
    if freeze_level == 0:
        patterns.append(r"^backbone\.head\.")
    else:
        patterns.extend([r"^backbone\.head\.", r"^mbm\."])

    if use_adapter:
        patterns.append(r"\.adapter_")

    unfreeze_by_regex(model, patterns)

    apply_bn_ln_policy(model, train_ln=not freeze_ln)


def partial_freeze_student_resnet(
    model: nn.Module,
    freeze_bn: bool = True,
    use_adapter: bool = False,
    freeze_level: int = 1,
):
    """Partially freeze a ResNet101 student using a numeric level."""
    freeze_all(model)

    unfreeze_patterns = []
    if freeze_level == 0:  # only the classifier head
        unfreeze_patterns.append(r"\.fc\.")
    elif freeze_level == 2:  # layer3 + layer4 + classifier
        unfreeze_patterns.extend([r"\.layer3\.", r"\.layer4\.", r"\.fc\."])
    else:  # default & level 1 => layer4 + classifier
        unfreeze_patterns.extend([r"\.layer4\.", r"\.fc\."])

    if use_adapter:
        unfreeze_patterns.append(r"adapter_")

    unfreeze_by_regex(model, unfreeze_patterns)

    if not freeze_bn:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                for p in m.parameters():
                    p.requires_grad = True


def partial_freeze_student_efficientnet(
    model: nn.Module,
    freeze_bn: bool = True,
    use_adapter: bool = False,
    freeze_level: int = 0,
):
    """Partially freeze an EfficientNet-B2 student using a numeric level."""
    freeze_all(model)

    unfreeze_patterns = []
    if freeze_level == 1:  # last few blocks + classifier
        unfreeze_patterns.extend(
            [r"features\.6\.", r"features\.7\.", r"features\.8\.", r"classifier\."]
        )
    else:  # default & level 0 => classifier only
        unfreeze_patterns.append(r"classifier\.")

    if use_adapter:
        unfreeze_patterns.append(r"adapter_")

    unfreeze_by_regex(model, unfreeze_patterns)

    if not freeze_bn:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                for p in m.parameters():
                    p.requires_grad = True


def partial_freeze_student_swin(
    model: nn.Module,
    freeze_ln: bool = True,
    use_adapter: bool = False,
    freeze_level: int = 1,
):
    """Partially freeze a Swin Tiny student using a numeric level."""
    freeze_all(model)

    unfreeze_patterns = []
    if freeze_level == 0:  # only head
        unfreeze_patterns.extend([r"\.head\.", r"\.fc\."])
    else:  # default & level 1 => last block + head
        unfreeze_patterns.extend([r"\.layers\.3\.", r"\.head\.", r"\.fc\."])

    if use_adapter:
        unfreeze_patterns.append(r"adapter_")

    unfreeze_by_regex(model, unfreeze_patterns)

    if not freeze_ln:
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True


def freeze_teacher_params(
    model: nn.Module,
    teacher_name: str = "resnet101",
    freeze_bn: bool = True,
    freeze_ln: bool = True,
    use_adapter: bool = False,
    bn_head_only: bool = False,
    freeze_level: int = 1,
) -> None:
    """Wrapper that partially freezes a teacher model by type."""
    if teacher_name == "resnet101":
        partial_freeze_teacher_resnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=use_adapter,
            bn_head_only=bn_head_only,
            freeze_level=freeze_level,
        )
    elif teacher_name == "efficientnet_b2":
        partial_freeze_teacher_efficientnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=use_adapter,
            bn_head_only=bn_head_only,
            freeze_level=freeze_level,
        )
    elif teacher_name == "swin_tiny":
        partial_freeze_teacher_swin(
            model,
            freeze_ln=freeze_ln,
            use_adapter=use_adapter,
            freeze_level=freeze_level,
        )
    else:
        freeze_all(model)


def freeze_student_with_adapter(
    model: nn.Module,
    student_name: str = "resnet_adapter",
    freeze_bn: bool = True,
    freeze_ln: bool = True,
    freeze_level: int = 1,
) -> None:
    """Wrapper that freezes a student and optionally unfreezes its adapters."""
    if student_name == "resnet_adapter":
        partial_freeze_student_resnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=True,
            freeze_level=freeze_level,
        )
    elif student_name == "efficientnet_adapter":
        partial_freeze_student_efficientnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=True,
            freeze_level=freeze_level,
        )
    elif student_name == "swin_adapter":
        partial_freeze_student_swin(
            model,
            freeze_ln=freeze_ln,
            use_adapter=True,
            freeze_level=freeze_level,
        )
    else:
        partial_freeze_student_resnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=True,
            freeze_level=freeze_level,
        )
