# models/students/resnet152_student.py
"""ResNet‑152 학생 • layer3 이후 ChannelAdapter 추가"""

import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights

from models.common.base_wrapper import BaseKDModel, register
from models.common.adapter import ChannelAdapter2D

@register("resnet152_pretrain_student")
@register("resnet152_scratch_student")

class ResNet152Student(BaseKDModel):
    """ResNet‑152 backbone + 1×1 Adapter + 학설용 classifier(fc)."""

    def __init__(
        self,
        *,
        num_classes: int = 100,
        pretrained: bool = True,
        small_input: bool = False,
        cfg: dict | None = None,
    ):
        backbone = resnet152(
            weights=ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
        )
        if small_input:
            # CIFAR‑size 입력 대응
            backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            backbone.maxpool = nn.Identity()

        super().__init__(backbone, num_classes, role="student", cfg=cfg or {})

        # ── ChannelAdapter : layer3 → layer4 사이 ───────────────
        ch = backbone.layer3[-1].conv3.out_channels        # 1024
        self.adapter = ChannelAdapter2D(ch)

    # --------------------------------------------------------------
    def extract_feats(self, x):
        b = self.backbone
        x = b.relu(b.bn1(b.conv1(x)))
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = self.adapter(b.layer3(x))      # <-- adapter 적용
        f4d = b.layer4(x)
        f2d = torch.flatten(b.avgpool(f4d), 1)
        return f4d, f2d
