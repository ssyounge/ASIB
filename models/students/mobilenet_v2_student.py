# models/students/mobilenet_v2_student.py
"""MobileNet-V2 학생 • 분류 경로 1280ch 유지, distill은 별도 어댑터(1D) 사용"""

import torch
import torch.nn as nn
from typing import Optional
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from models.common.base_wrapper import BaseKDModel, register

@register("mobilenet_v2_student")
@register("mobilenet_v2_scratch")
@register("mobilenet_v2_scratch_student")
class MobileNetV2Student(BaseKDModel):
    """MobileNet-V2 backbone; classifier 입력은 원본 1280차원을 유지.

    distillation feature는 BaseKDModel의 distillation_adapter로 생성되며,
    분류 경로에는 2D 어댑터를 삽입하지 않는다.
    """

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
        # CIFAR/small-input: 첫 conv stride를 1로 낮춰 과도한 다운샘플 방지
        if small_input:
            try:
                conv0 = backbone.features[0][0]
                if isinstance(conv0, nn.Conv2d):
                    conv0.stride = (1, 1)
                    conv0.padding = (1, 1)
            except Exception:
                pass

        super().__init__(backbone, num_classes, role="student", cfg=cfg or {})

    # --------------------------------------------------------------
    def extract_feats(self, x):
        b = self.backbone
        x = b.features(x)
        f4d = x
        # Global average pooling + flatten for classifier (1280-dim 유지)
        x_pooled = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)), 1)
        f2d = x_pooled
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