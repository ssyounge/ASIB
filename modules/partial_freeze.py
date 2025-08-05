# modules/partial_freeze.py

import torch.nn as nn
import logging
import re

# 순환 import 방지를 위해 필요한 함수들을 직접 정의
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


def get_freeze_schedule(model_name: str, freeze_level: int = 1):
    """
    Get freeze schedule for a given model and freeze level.
    
    Parameters:
    -----------
    model_name : str
        Name of the model (e.g., 'resnet', 'efficientnet', 'convnext')
    freeze_level : int
        Freeze level (0: head only, 1: last block, 2: last two blocks)
        
    Returns:
    --------
    dict
        Freeze schedule configuration
    """
    if freeze_level < 0:
        return {'freeze_all': False, 'patterns': []}
    
    if freeze_level == 0:
        patterns = [
            r"(?:^|\.)fc\.",
            r"(?:^|\.)classifier\.",
            r"(?:^|\.)head\.",
        ]
    elif freeze_level == 1:
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
    
    return {
        'freeze_all': True,
        'patterns': patterns,
        'freeze_bn': True,
        'freeze_ln': True
    }


def apply_freeze_schedule(model: nn.Module, schedule: dict):
    """
    Apply freeze schedule to a model.
    
    Parameters:
    -----------
    model : nn.Module
        Model to apply freeze schedule to
    schedule : dict
        Freeze schedule from get_freeze_schedule
    """
    if schedule.get('freeze_all', False):
        freeze_all(model)
        
        if schedule.get('patterns'):
            unfreeze_by_regex(model, schedule['patterns'])
    
    apply_bn_ln_policy(
        model,
        train_bn=not schedule.get('freeze_bn', True),
        train_ln=not schedule.get('freeze_ln', True)
    )


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
    train_distill_adapter_only: bool = False,
):
    """Partially freeze a ResNet101 teacher using a numeric level.

    ``freeze_all`` is called first and layers are selectively unfrozen based on
    ``freeze_level``:

    - ``0`` → only ``fc``
    - ``1`` → ``fc`` + ``mbm`` (default)
    - ``2`` → ``layer4`` + ``fc`` + ``mbm``
    """
    freeze_all(model)

    if train_distill_adapter_only:
        # distillation_adapter 만 학습
        logging.debug("[Freeze] unfreeze distillation_adapter (teacher-side)")
        unfreeze_by_regex(model, r"\.distillation_adapter\.")
        apply_bn_ln_policy(model, train_bn=not freeze_bn)
        return

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
        patterns.append(r"\.distillation_adapter\.")
        logging.debug("[partial_freeze_teacher_*] unfreeze distillation_adapter")

    unfreeze_by_regex(model, patterns)

    apply_bn_ln_policy(model, train_bn=not freeze_bn)


def partial_freeze_teacher_efficientnet(
    model: nn.Module,
    freeze_bn: bool = True,
    use_adapter: bool = False,
    bn_head_only: bool = False,
    freeze_level: int = 1,
    train_distill_adapter_only: bool = False,
): 
    """Partially freeze an EfficientNet teacher using a numeric level.

    ``freeze_all`` is called first and layers are unfrozen based on
    ``freeze_level``:

    - ``0`` or ``3`` → only ``classifier``
    - ``1`` → ``classifier`` + ``mbm`` (default)
    - ``2`` → ``features`` + ``classifier`` + ``mbm``
    """
    freeze_all(model)

    if train_distill_adapter_only:
        # distillation_adapter 만 학습
        logging.debug("[Freeze] unfreeze distillation_adapter (teacher-side)")
        unfreeze_by_regex(model, r"\.distillation_adapter\.")
        apply_bn_ln_policy(model, train_bn=not freeze_bn)
        return

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
        patterns.append(r"\.distillation_adapter\.")

    unfreeze_by_regex(model, patterns)

    apply_bn_ln_policy(model, train_bn=not freeze_bn)


# ──────────────────────────────────────────────────────────────
# ConvNeXt-Large teacher
# ──────────────────────────────────────────────────────────────
def partial_freeze_teacher_convnext(
    model: nn.Module,
    *,
    freeze_bn: bool = True,
    freeze_level: int = 1,
    use_adapter: bool = False,
    train_distill_adapter_only: bool = False,
):
    """Partially freeze a ConvNeXt-Large teacher (rule compatible with ResNet)."""

    if freeze_level < 0:
        return

    freeze_all(model)

    if train_distill_adapter_only:
        unfreeze_by_regex(model, r"\.distillation_adapter\.")
        apply_bn_ln_policy(model, train_bn=not freeze_bn)
        return

    unfreeze = []
    if freeze_level == 0:
        unfreeze.append(r"(?:^|\.)head\.")
    elif freeze_level == 2:
        unfreeze.extend([
            r"\.stages\.3\.",
            r"(?:^|\.)head\.",
        ])
    else:  # level == 1
        unfreeze.append(r"(?:^|\.)head\.")

    if freeze_level >= 1:
        unfreeze.append(r"^mbm\.")

    if use_adapter:
        unfreeze.append(r"\.distillation_adapter\.")

    unfreeze_by_regex(model, unfreeze)
    apply_bn_ln_policy(model, train_bn=not freeze_bn)


def partial_freeze_teacher_swin(
    model: nn.Module,
    freeze_ln: bool = True,
    use_adapter: bool = False,
    freeze_level: int = 1,
    train_distill_adapter_only: bool = False,
):
    """Partially freeze a Swin Tiny teacher using a numeric level.

    ``freeze_all`` is called first. ``freeze_level`` controls what to unfreeze:

    - ``0`` → only ``head``
    - ``1`` → ``head`` + ``mbm`` (default)
    """
    freeze_all(model)

    if train_distill_adapter_only:
        # distillation_adapter 만 학습
        logging.debug("[Freeze] unfreeze distillation_adapter (teacher-side)")
        unfreeze_by_regex(model, r"\.distillation_adapter\.")
        apply_bn_ln_policy(model, train_ln=not freeze_ln)
        return

    patterns = []
    if freeze_level == 0:
        patterns.append(r"^backbone\.head\.")
    else:
        patterns.extend([r"^backbone\.head\.", r"^mbm\."])

    if use_adapter:
        patterns.append(r"\.distillation_adapter\.")

    unfreeze_by_regex(model, patterns)

    apply_bn_ln_policy(model, train_ln=not freeze_ln)


def partial_freeze_student_resnet(
    model: nn.Module,
    freeze_bn: bool = True,
    use_adapter: bool = False,
    freeze_level: int = 1,
):
    """Partially freeze a ResNet101 student.

    ``freeze_all`` is called first and individual blocks are then
    re-enabled for training based on ``freeze_level``:

    - ``0`` → only the classifier ``fc``
    - ``1`` → ``layer4`` + ``fc`` (default)
    - ``2`` → ``layer3`` + ``layer4`` + ``fc``

    Setting ``use_adapter`` will additionally unfreeze any modules whose
    name contains ``adapter_``. Passing ``freeze_bn=False`` allows the
    BatchNorm affine parameters to update.
    """
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
def partial_freeze_student_swin(
    model: nn.Module,
    freeze_ln: bool = True,
    use_adapter: bool = False,
    freeze_level: int = 1,
):
    """Partially freeze a Swin Tiny student.

    ``freeze_all`` freezes every parameter first. The ``freeze_level``
    determines which parts of the backbone are unfrozen:

    - ``0`` → only the ``head``/``fc`` layers
    - ``1`` → ``layers.3`` + ``head``/``fc`` (default)

    When ``use_adapter`` is ``True`` any ``adapter_*`` modules are also
    unfrozen. Setting ``freeze_ln=False`` allows LayerNorm parameters to be
    updated.
    """
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
    teacher_name: str = "resnet152",
    freeze_bn: bool = True,
    freeze_ln: bool = True,
    use_adapter: bool = False,
    bn_head_only: bool = False,
    freeze_level: int = 1,
    train_distill_adapter_only: bool = False,
) -> None:
    """Wrapper that partially freezes a teacher model by type.

    The caller may pass either ``"convnext_l"`` or ``"convnext_l_teacher"``
    — we strip a trailing ``"_teacher"`` so that both variants map to the
    same control branch below.
    """

    # ----------------------------------------------------------
    # (★) key normalisation →  convnext_l_teacher → convnext_l
    # ----------------------------------------------------------
    if teacher_name.endswith("_teacher"):
        teacher_name = teacher_name[:-8]       # drop "_teacher"
    if teacher_name == "resnet152":
        partial_freeze_teacher_resnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=use_adapter,
            bn_head_only=bn_head_only,
            freeze_level=freeze_level,
            train_distill_adapter_only=train_distill_adapter_only,
        )
    elif teacher_name in ("efficientnet_l2", "effnet_l2"):
        partial_freeze_teacher_efficientnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=use_adapter,
            bn_head_only=bn_head_only,
            freeze_level=freeze_level,
            train_distill_adapter_only=train_distill_adapter_only,
        )
    elif teacher_name == "swin_tiny":
        partial_freeze_teacher_swin(
            model,
            freeze_ln=freeze_ln,
            use_adapter=use_adapter,
            freeze_level=freeze_level,
            train_distill_adapter_only=train_distill_adapter_only,
        )
    elif teacher_name in ("convnext_l", "convnext_large", "convnext_l_teacher"):
        partial_freeze_teacher_convnext(
            model,
            freeze_bn=freeze_bn,
            freeze_level=freeze_level,
            use_adapter=use_adapter,
            train_distill_adapter_only=train_distill_adapter_only,
        )
    else:
        freeze_all(model)


def freeze_student_with_adapter(
    model: nn.Module,
    student_name: str = "resnet",
    freeze_bn: bool = True,
    freeze_ln: bool = True,
    freeze_level: int = 1,
) -> None:
    """Wrapper that freezes a student and optionally unfreezes its adapters."""
    if student_name == "resnet":
        partial_freeze_student_resnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=True,
            freeze_level=freeze_level,
        )
    elif student_name == "swin":
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
