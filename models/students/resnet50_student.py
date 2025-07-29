# models/students/resnet50_student.py

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

@register("resnet50_scratch")
from models.common.base_wrapper import BaseKDModel, register
from models.common.adapter import ChannelAdapter2D


@register("resnet50_student")
class ResNet50Student(BaseKDModel):
    """ResNet-50 student with ChannelAdapter after layer3."""

    def __init__(
        self,
        *,
        num_classes: int = 100,
        pretrained: bool = True,
        small_input: bool = False,
        cfg: dict | None = None,
    ) -> None:
        backbone = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        if small_input:
            backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            backbone.maxpool = nn.Identity()

        super().__init__(backbone, num_classes, role="student", cfg=cfg or {})

        ch = backbone.layer3[-1].conv3.out_channels
        self.adapter = ChannelAdapter2D(ch)

    # ------------------------------------------------------------------
    def extract_feats(self, x):
        b = self.backbone
        x = b.relu(b.bn1(b.conv1(x)))
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = self.adapter(b.layer3(x))
        feat_4d = b.layer4(x)
        feat_2d = torch.flatten(b.avgpool(feat_4d), 1)
        return feat_4d, feat_2d

