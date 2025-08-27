# models/students/shufflenet_v2_student.py
"""ShuffleNet-V2 학생 • 마지막 layer 이후 ChannelAdapter 추가"""

import torch
import torch.nn as nn
from typing import Optional
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights

from models.common.base_wrapper import BaseKDModel, register
from models.common.adapter import ChannelAdapter2D

@register("shufflenet_v2_student")
@register("shufflenet_v2_scratch_student")
class ShuffleNetV2Student(BaseKDModel):
    """ShuffleNet-V2 backbone + 1×1 Adapter + 학설용 classifier(fc)."""

    def __init__(
        self,
        *,
        num_classes: int = 100,
        pretrained: bool = True,
        small_input: bool = False,
        cfg: Optional[dict] = None,
    ):
        backbone = shufflenet_v2_x1_0(
            weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # CIFAR/small-input: conv1 stride=1, maxpool 제거로 과다운샘플 방지
        if small_input:
            try:
                if isinstance(backbone.conv1[0], nn.Conv2d):
                    backbone.conv1[0].stride = (1, 1)
                    backbone.conv1[0].padding = (1, 1)
                backbone.maxpool = nn.Identity()
            except Exception:
                pass
        
        super().__init__(backbone, num_classes, role="student", cfg=cfg or {})

        # ── ChannelAdapter : 마지막 layer 이후 ───────────────
        ch = backbone.conv5[0].out_channels  # 1024 (Conv2d layer)
        self.adapter = ChannelAdapter2D(ch)

    # --------------------------------------------------------------
    def extract_feats(self, x):
        b = self.backbone
        x = b.conv1(x)
        x = b.maxpool(x)
        x = b.stage2(x)
        x = b.stage3(x)
        x = b.stage4(x)
        x = b.conv5(x)                    # conv5 먼저 적용
        x = self.adapter(x)               # <-- adapter 적용
        f4d = x
        # Global average pooling + flatten for classifier
        x_pooled = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)), 1)
        f2d = x_pooled  # Use BaseKDModel's classifier, not backbone's fc
        return f4d, f2d


def create_shufflenet_v2_student(
    num_classes: int = 100,
    pretrained: bool = True,
    small_input: bool = False,
    cfg: Optional[dict] = None,
) -> ShuffleNetV2Student:
    """Build :class:`ShuffleNetV2Student`."""
    return ShuffleNetV2Student(
        num_classes=num_classes,
        pretrained=pretrained,
        small_input=small_input,
        cfg=cfg,
    )


def create_shufflenet_v2_scratch_student(
    num_classes: int = 100,
    pretrained: bool = False,
    small_input: bool = False,
    cfg: Optional[dict] = None,
) -> ShuffleNetV2Student:
    """Build :class:`ShuffleNetV2Student` (scratch version)."""
    return ShuffleNetV2Student(
        num_classes=num_classes,
        pretrained=pretrained,
        small_input=small_input,
        cfg=cfg,
    ) 