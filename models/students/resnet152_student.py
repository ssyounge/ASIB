"""
ResNet-152 student with Channel-Adapter after layer3 (registry v2).
"""

import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights

from models.common.base_wrapper import BaseKDModel, register
from models.common.adapter import ChannelAdapter2D


@register("resnet152_student")
class ResNet152Student(BaseKDModel):
    def __init__(
        self,
        *,
        pretrained: bool = True,
        num_classes: int = 100,
        small_input: bool = False,
        cfg: dict | None = None,
    ):
        backbone = resnet152(
            weights=ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
        )
        if small_input:
            backbone.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            backbone.maxpool = nn.Identity()

        super().__init__(backbone, num_classes, role="student", cfg=cfg or {})

        ch = backbone.layer3[-1].conv3.out_channels  # 1024
        self.adapter = ChannelAdapter2D(ch)

    # ------------------------------------------------------------
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

MODEL_REGISTRY["resnet152_adapter_student"] = ResNet152Student


def create_resnet152_with_extended_adapter(*a, **kw):
    """Backward compatibility wrapper."""
    import warnings

    warnings.warn(
        "renamed → models.students.resnet152_student.ResNet152Student",
        DeprecationWarning,
        stacklevel=2,
    )
    return ResNet152Student(*a, **kw)
