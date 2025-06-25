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
    freeze_scope: str = None,
):
    """Partially freeze a ResNet101 teacher.

    1) ``freeze_all`` is called first.
    2) Layers are unfrozen based on ``freeze_scope``.
    3) When ``freeze_bn`` is ``False`` the BatchNorm layers are also unfrozen.

    Example ``freeze_scope`` values:
      - ``"fc_only"``: only the fully-connected layer is unfrozen.
      - ``"layer4_fc"``: unfreeze ``layer4`` and the fully-connected layer
        (and ``mbm`` if present).
      - ``None`` (default): unfreeze the fully-connected layer and ``mbm`` only.
    """
    freeze_all(model)

    if bn_head_only:
        unfreeze_by_regex(model, r"^backbone\.fc\.")
        apply_bn_ln_policy(model, train_bn=True)
        return

    patterns = []
    if freeze_scope == "fc_only":
        patterns.append(r"^backbone\.fc\.")
    elif freeze_scope == "layer4_fc":
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
    freeze_scope: str = None,
):
    """Partially freeze an EfficientNet-B2 teacher.

    Example ``freeze_scope`` values:
       ``"classifier_only"``, ``"features_classifier"``, etc.
    """
    freeze_all(model)

    if bn_head_only:
        unfreeze_by_regex(model, r"^backbone\.classifier\.")
        apply_bn_ln_policy(model, train_bn=True)
        return

    patterns = []
    if freeze_scope == "classifier_only":
        patterns.append(r"^backbone\.classifier\.")
    elif freeze_scope == "features_classifier":
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
    freeze_scope: str = None,
):
    """Partially freeze a Swin Tiny teacher.

    - Example ``freeze_scope``: ``"head_only"``.
    - Default behaviour unfreezes the classification head and ``mbm``.
    - When ``freeze_ln=True`` the LayerNorms remain frozen.
    """
    freeze_all(model)

    patterns = []
    if freeze_scope == "head_only":
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
    freeze_scope: str = None,
):
    """Partially freeze a ResNet101 student.

    - Layers to unfreeze depend on ``freeze_scope`` (e.g. ``layer4``,
      ``fc`` or the adapters).
    - Example values: ``"layer4_fc"``, ``"fc_only"``, etc.
    """
    freeze_all(model)

    if freeze_scope == "fc_only":
        for name, param in model.named_parameters():
            if "fc." in name:
                param.requires_grad = True
    elif freeze_scope == "layer4_fc":
        for name, param in model.named_parameters():
            if "layer4." in name or "fc." in name:
                param.requires_grad = True
    else:
        # default => layer4 + fc (original logic)
        for name, param in model.named_parameters():
            if "layer4." in name or "fc." in name:
                param.requires_grad = True

    # adapters
    if use_adapter:
        for name, param in model.named_parameters():
            if "adapter_" in name:
                param.requires_grad = True

    if not freeze_bn:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                for p in m.parameters():
                    p.requires_grad = True


def partial_freeze_student_efficientnet(
    model: nn.Module,
    freeze_bn: bool = True,
    use_adapter: bool = False,
    freeze_scope: str = None,
):
    """Partially freeze an EfficientNet-B2 student.

    - Example ``freeze_scope`` values:
       ``"classifier_only"``, ``"features_classifier"``, etc.
    """
    freeze_all(model)

    if freeze_scope == "classifier_only":
        for name, param in model.named_parameters():
            if "classifier." in name:
                param.requires_grad = True

    elif freeze_scope == "features_classifier":
        for name, param in model.named_parameters():
            if "features." in name or "classifier." in name:
                param.requires_grad = True
    else:
        # default => classifier (original)
        for name, param in model.named_parameters():
            if "classifier." in name:
                param.requires_grad = True

    # adapters
    if use_adapter:
        for name, param in model.named_parameters():
            if "adapter_" in name:
                param.requires_grad = True

    if not freeze_bn:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                for p in m.parameters():
                    p.requires_grad = True


def partial_freeze_student_swin(
    model: nn.Module,
    freeze_ln: bool = True,
    use_adapter: bool = False,
    freeze_scope: str = None,
):
    """
    Student (Swin Tiny)
    """
    freeze_all(model)

    if freeze_scope == "head_only":
        for name, param in model.named_parameters():
            if "head." in name or "fc." in name:
                param.requires_grad = True
    else:
        # default => "layers.3." + "head."
        for name, param in model.named_parameters():
            if "layers.3." in name or "head." in name or "fc." in name:
                param.requires_grad = True

    if use_adapter:
        for name, param in model.named_parameters():
            if "adapter_" in name:
                param.requires_grad = True

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
    freeze_scope: str = None,
) -> None:
    """Wrapper that partially freezes a teacher model by type."""
    if teacher_name == "resnet101":
        partial_freeze_teacher_resnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=use_adapter,
            bn_head_only=bn_head_only,
            freeze_scope=freeze_scope,
        )
    elif teacher_name == "efficientnet_b2":
        partial_freeze_teacher_efficientnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=use_adapter,
            bn_head_only=bn_head_only,
            freeze_scope=freeze_scope,
        )
    elif teacher_name == "swin_tiny":
        partial_freeze_teacher_swin(
            model,
            freeze_ln=freeze_ln,
            use_adapter=use_adapter,
            freeze_scope=freeze_scope,
        )
    else:
        freeze_all(model)


def freeze_student_with_adapter(
    model: nn.Module,
    student_name: str = "resnet_adapter",
    freeze_bn: bool = True,
    freeze_ln: bool = True,
    freeze_scope: str = None,
) -> None:
    """Wrapper that freezes a student and optionally unfreezes its adapters."""
    if student_name == "resnet_adapter":
        partial_freeze_student_resnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=True,
            freeze_scope=freeze_scope,
        )
    elif student_name == "efficientnet_adapter":
        partial_freeze_student_efficientnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=True,
            freeze_scope=freeze_scope,
        )
    elif student_name == "swin_adapter":
        partial_freeze_student_swin(
            model,
            freeze_ln=freeze_ln,
            use_adapter=True,
            freeze_scope=freeze_scope,
        )
    else:
        partial_freeze_student_resnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=True,
            freeze_scope=freeze_scope,
        )
