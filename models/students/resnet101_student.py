"""ResNet‑101 student with an adapter after layer3."""
# models/students/resnet101_student.py

import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights

from models.common.base_wrapper import BaseKDModel, register
from models.common.adapter import ChannelAdapter2D

@register("resnet101_pretrain_student")
@register("resnet101_scratch_student")
class ResNetStudent(BaseKDModel):
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

# legacy registry key for backward compatibility
from models.common.base_wrapper import MODEL_REGISTRY
MODEL_REGISTRY["resnet101_adapter_student"] = ResNetStudent


# ── 역호환 alias (기존 import 유지) ────────────────────────────
def create_resnet101_with_extended_adapter(*a, **kw):
    import warnings
    warnings.warn(
        "renamed → models.students.resnet101_student.ResNetStudent",
        DeprecationWarning, stacklevel=2,
    )
    return ResNetStudent(*a, **kw)
