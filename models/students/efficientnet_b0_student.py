# models/students/efficientnet_b0_student.py
"""EfficientNet-B0 학생 • 마지막 layer 이후 ChannelAdapter 추가"""

import torch
import torch.nn as nn
from typing import Optional
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from models.common.base_wrapper import BaseKDModel, register
from models.common.adapter import ChannelAdapter2D

@register("efficientnet_b0_student")
@register("efficientnet_b0_scratch_student")
class EfficientNetB0Student(BaseKDModel):
    """EfficientNet-B0 backbone + 1×1 Adapter + 학설용 classifier(fc)."""

    def __init__(
        self,
        *,
        num_classes: int = 100,
        pretrained: bool = True,
        small_input: bool = False,
        cfg: Optional[dict] = None,
    ):
        backbone = efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        super().__init__(backbone, num_classes, role="student", cfg=cfg or {})

        # ── ChannelAdapter : 마지막 layer 이후 ───────────────
        ch = backbone.classifier[1].in_features  # 1280
        self.adapter = ChannelAdapter2D(ch)

    # --------------------------------------------------------------
    def extract_feats(self, x):
        b = self.backbone
        x = b.features(x)
        x = self.adapter(x)      # <-- adapter 적용
        f4d = x
        # Global average pooling + flatten for classifier
        x_pooled = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)), 1)
        f2d = x_pooled  # Use BaseKDModel's classifier, not backbone's classifier
        return f4d, f2d


def create_efficientnet_b0_student(
    num_classes: int = 100,
    pretrained: bool = True,
    small_input: bool = False,
    cfg: Optional[dict] = None,
) -> EfficientNetB0Student:
    """Build :class:`EfficientNetB0Student`."""
    return EfficientNetB0Student(
        num_classes=num_classes,
        pretrained=pretrained,
        small_input=small_input,
        cfg=cfg,
    )


def create_efficientnet_b0_scratch_student(
    num_classes: int = 100,
    pretrained: bool = False,
    small_input: bool = False,
    cfg: Optional[dict] = None,
) -> EfficientNetB0Student:
    """Build :class:`EfficientNetB0Student` (scratch version)."""
    return EfficientNetB0Student(
        num_classes=num_classes,
        pretrained=pretrained,
        small_input=small_input,
        cfg=cfg,
    ) 