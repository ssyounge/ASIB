"""
리팩터링 후: ResNet student는 BaseKDModel 서브클래스로 간결하게 정의
"""
# models/students/student_resnet_adapter.py

import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights

from models.common.base_wrapper import BaseKDModel, register
from models.common.adapter import ChannelAdapter2D

@register("resnet101_adapter_student")
class ResNetAdapterStudent(BaseKDModel):
    def __init__(self, *, pretrained: bool = True, num_classes: int = 100,
                 small_input: bool = False, cfg: dict | None = None):
        backbone = resnet101(
            weights=ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        )
        if small_input:
            backbone.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            backbone.maxpool = nn.Identity()

        super().__init__(backbone, num_classes, role="student", cfg=cfg or {})

        ch = backbone.layer3[-1].conv3.out_channels  # 1024
        self.adapter = ChannelAdapter2D(ch)

    # ------------------------------------------------------------------
    def extract_feats(self, x):
        b = self.backbone
        x = b.relu(b.bn1(b.conv1(x)))
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = self.adapter(b.layer3(x))
        f4d = b.layer4(x)
        f2d = torch.flatten(b.avgpool(f4d), 1)
        return f4d, f2d


def create_resnet101_with_extended_adapter(pretrained: bool = True,
                                           num_classes: int = 100,
                                           small_input: bool = False,
                                           cfg: dict | None = None):
    """Backward compatible helper."""
    return ResNetAdapterStudent(pretrained=pretrained,
                                num_classes=num_classes,
                                small_input=small_input,
                                cfg=cfg)
