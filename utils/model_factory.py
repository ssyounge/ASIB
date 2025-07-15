# utils/model_factory.py

import torch
from typing import Optional

from models.teachers.teacher_resnet import create_resnet152
from models.teachers.teacher_efficientnet import create_efficientnet_b2
# Snapshot ensemble support
from models.ensemble.snapshot_teacher import SnapshotTeacher
from models.students.student_convnext import (
    create_convnext_tiny,
    create_convnext_small,
)

__all__ = [
    "create_teacher_by_name",
    "create_student_by_name",
]


def create_teacher_by_name(
    teacher_type: str,
    num_classes: int = 100,
    pretrained: bool = True,
    small_input: bool = False,
    dropout_p: float = 0.3,
    cfg: Optional[dict] = None,
):
    """Factory for teacher models in this minimal repo."""
    if cfg is not None:
        dropout_p = cfg.get("teacher_dropout_p", dropout_p)
    if teacher_type == "resnet152":
        return create_resnet152(
            num_classes=num_classes,
            pretrained=pretrained,
            small_input=small_input,
            cfg=cfg,
        )
    if teacher_type == "efficientnet_b2":
        return create_efficientnet_b2(
            num_classes=num_classes,
            pretrained=pretrained,
            small_input=small_input,
            dropout_p=dropout_p,
            cfg=cfg,
        )

    # Snapshot ensemble teacher
    if teacher_type.lower() == "snapshot":
        # cfg 에서 가져온 콤마 구분 문자열 → ['file1.pth', 'file2.pth', …]
        ckpt_raw = (cfg or {}).get("teacher1_ckpt", "")
        ckpts = [p.strip() for p in ckpt_raw.split(",") if p.strip()]
        assert ckpts, (
            "[SnapshotTeacher] teacher1_ckpt 가 비었습니다. "
            f"(현재 값='{ckpt_raw}')"
        )
        base = (cfg or {}).get("snapshot_backbone", "resnet152")
        return SnapshotTeacher(ckpt_paths=ckpts, backbone_name=base, n_cls=num_classes)
    raise ValueError(
        f"Unknown teacher_type: {teacher_type} (expected 'resnet152' or 'efficientnet_b2')"
    )


def create_student_by_name(
    student_type: str,
    num_classes: int = 100,
    pretrained: bool = True,
    small_input: bool = False,
    cfg: Optional[dict] = None,
):
    """Factory for student models in this minimal repo."""
    if student_type == "convnext_tiny":
        return create_convnext_tiny(
            num_classes=num_classes,
            pretrained=pretrained,
            small_input=small_input,
            cfg=cfg,
        )
    if student_type == "convnext_small":
        return create_convnext_small(
            num_classes=num_classes,
            pretrained=pretrained,
            small_input=small_input,
            cfg=cfg,
        )
    raise ValueError(
        f"Unknown student_type: {student_type} (expected 'convnext_tiny' or 'convnext_small')"
    )

