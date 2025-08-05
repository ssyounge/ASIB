# models/students/mobilenet_v2_student.py
"""MobileNet-V2 학생 • 마지막 layer 이후 ChannelAdapter 추가"""

import torch
import torch.nn as nn
from typing import Optional
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from models.common.base_wrapper import BaseKDModel, register
from models.common.adapter import ChannelAdapter2D

@register("mobilenet_v2_student")
@register("mobilenet_v2_scratch_student")
class MobileNetV2Student(BaseKDModel):
    """MobileNet-V2 backbone + 1×1 Adapter + 학설용 classifier(fc)."""

    def __init__(
        self,
        *,
        num_classes: int = 100,
        pretrained: bool = True,
        small_input: bool = False,
        cfg: Optional[dict] = None,
    ):
        backbone = mobilenet_v2(
            weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
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


def create_mobilenet_v2_student(
    num_classes: int = 100,
    pretrained: bool = True,
    small_input: bool = False,
    cfg: Optional[dict] = None,
) -> MobileNetV2Student:
    """Build :class:`MobileNetV2Student`."""
    return MobileNetV2Student(
        num_classes=num_classes,
        pretrained=pretrained,
        small_input=small_input,
        cfg=cfg,
    )


def create_mobilenet_v2_scratch_student(
    num_classes: int = 100,
    pretrained: bool = False,
    small_input: bool = False,
    cfg: Optional[dict] = None,
) -> MobileNetV2Student:
    """Build :class:`MobileNetV2Student` (scratch version)."""
    return MobileNetV2Student(
        num_classes=num_classes,
        pretrained=pretrained,
        small_input=small_input,
        cfg=cfg,
    ) 