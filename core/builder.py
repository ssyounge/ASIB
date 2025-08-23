# core/builder.py

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union

from models.common.base_wrapper import MODEL_REGISTRY
from models.common import registry as _reg
from models import build_ib_mbm_from_teachers as build_from_teachers
from modules.partial_freeze import (
    partial_freeze_teacher_resnet,
    partial_freeze_teacher_efficientnet,
    partial_freeze_student_resnet,
)


def build_model(name: str, **kwargs: Any) -> nn.Module:
    """Build model from registry."""
    # 요청 key 가 아직 없으면 그때 가서 import-scan
    if name not in MODEL_REGISTRY:
        _reg.ensure_scanned()
    
    # timm의 INFO 메시지 억제
    import logging
    original_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)
    
    try:
        model = MODEL_REGISTRY[name](**kwargs)
        # 로깅 레벨 복원
        logging.getLogger().setLevel(original_level)
        return model
    except KeyError as exc:
        # 로깅 레벨 복원
        logging.getLogger().setLevel(original_level)
        known = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"[build_model] Unknown model key '{name}'. "
            f"Available: {known}"
        ) from exc


def create_student_by_name(
    student_name: str,
    pretrained: bool = True,
    small_input: bool = False,
    num_classes: int = 100,
    cfg: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Create student from :data:`MODEL_REGISTRY`."""

    # Friendly aliases: allow base names to map to scratch registry keys
    NAME_ALIASES: Dict[str, str] = {
        "resnet50": "resnet50_scratch",
        "mobilenet_v2": "mobilenet_v2_scratch",
        "efficientnet_b0": "efficientnet_b0_scratch",
        "shufflenet_v2": "shufflenet_v2_scratch",
    }
    student_name = NAME_ALIASES.get(student_name, student_name)

    try:
        return build_model(
            student_name,
            pretrained=pretrained,
            num_classes=num_classes,
            small_input=small_input,
            cfg=cfg,
        )
    except ValueError as exc:
        raise ValueError(
            f"[create_student_by_name] '{student_name}' not in registry"
        ) from exc


def create_teacher_by_name(
    teacher_name: str,
    num_classes: int = 100,
    pretrained: bool = True,
    small_input: bool = False,
    cfg: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Create teacher from :data:`MODEL_REGISTRY`."""

    try:
        return build_model(
            teacher_name,
            num_classes=num_classes,
            pretrained=pretrained,
            small_input=small_input,
            cfg=cfg,
        )
    except ValueError as exc:
        raise ValueError(
            f"[create_teacher_by_name] '{teacher_name}' not in registry"
        ) from exc


def partial_freeze_teacher_auto(
    model: nn.Module,
    teacher_name: str,
    freeze_bn: bool = True,
    freeze_ln: bool = True,
    use_adapter: bool = False,
    bn_head_only: bool = False,
    freeze_level: int = 1,
    train_distill_adapter_only: bool = False,
) -> None:
    """Automatically apply partial freeze based on teacher name."""
    if teacher_name.endswith("_teacher"):
        teacher_name = teacher_name[:-8]   # "efficientnet_l2_teacher" → "efficientnet_l2"
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
    elif teacher_name in ("convnext_l", "convnext_large", "convnext_l_teacher"):
        from modules.partial_freeze import partial_freeze_teacher_convnext
        partial_freeze_teacher_convnext(
            model,
            freeze_bn=freeze_bn,
            freeze_level=freeze_level,
            use_adapter=use_adapter,
            train_distill_adapter_only=train_distill_adapter_only,
        )
    else:
        raise ValueError(
            f"[partial_freeze_teacher_auto] Unknown teacher_name={teacher_name}"
        )


def partial_freeze_student_auto(
    model: nn.Module,
    student_name: str = "resnet",
    freeze_bn: bool = True,
    freeze_ln: bool = True,
    use_adapter: bool = False,
    freeze_level: int = 1,
) -> None:
    """Automatically apply partial freeze based on student name."""
    if student_name.startswith("resnet"):
        partial_freeze_student_resnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=use_adapter,
            freeze_level=freeze_level,
        )
    else:
        raise ValueError(
            f"[partial_freeze_student_auto] Unknown student_name={student_name}"
        ) 