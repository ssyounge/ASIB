# modules/partial_freeze.py

import torch
import torch.nn as nn

from utils.freeze import apply_bn_ln_policy, freeze_all, unfreeze_by_regex


def freeze_bn_params(module: nn.Module):
    """
    BatchNorm 계열 모듈(gamma/beta)도 학습하지 않도록 동결한다.
    model.apply(freeze_bn_params) 형태로 호출 가능.
    """
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        for p in module.parameters():
            p.requires_grad = False


def freeze_ln_params(module: nn.Module):
    """
    LayerNorm 계열 모듈에서 gamma/beta도 동결한다.
    (Swin/ViT 등에서 사용)
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
    """
    Teacher (ResNet101) partial freeze:
      1) freeze_all 먼저
      2) freeze_scope에 따라 특정 레이어만 unfreeze
      3) freeze_bn=False 면 BN도 unfreeze

    예시 freeze_scope:
      - "fc_only": fc 만 unfreeze
      - "layer4_fc": layer4 + fc (그리고 mbm.?) unfreeze
      - None (default): fc + mbm 만 unfreeze
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
    """
    Teacher (EfficientNet-B2) partial freeze
      - freeze_scope 예시:
         "classifier_only", "features_classifier", etc.
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
    """
    Teacher (Swin Tiny) partial freeze
      - freeze_scope 예: "head_only"
      - default: head. + mbm.
      - freeze_ln=True => LN 동결
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
    """
    Student (ResNet101)
      - freeze_scope에 따라 layer4. / fc. / adapter_ 등 분기
      - 예: "layer4_fc", "fc_only", etc.
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
        # default => layer4. + fc. (기존 로직)
        for name, param in model.named_parameters():
            if "layer4." in name or "fc." in name:
                param.requires_grad = True

    # adapter
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
    """
    Student (EfficientNet-B2) partial freeze
      - freeze_scope 예시:
         "classifier_only", "features_classifier", etc.
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
        # default => classifier. (기존)
        for name, param in model.named_parameters():
            if "classifier." in name:
                param.requires_grad = True

    # adapter
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
