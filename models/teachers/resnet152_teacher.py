# models/teachers/resnet152_teacher.py

import torch
import torch.nn as nn
from models.common.base_wrapper import BaseKDModel, register, MODEL_REGISTRY

import torchvision.models as tv


@register("resnet152_teacher")  # single entry
class ResNet152Teacher(BaseKDModel):
    """ResNet-152 Teacher with optional distillation adapter."""

    def __init__(
        self,
        *,
        pretrained: bool = True,
        num_classes: int = 100,
        small_input: bool = False,
        cfg: dict | None = None,
    ):
        backbone = tv.resnet152(
            weights=tv.ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
        )
        if small_input:
            backbone.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            backbone.maxpool = nn.Identity()
        super().__init__(backbone, num_classes, role="teacher", cfg=cfg or {})

    def extract_feats(self, x):
        b = self.backbone
        x = b.relu(b.bn1(b.conv1(x)))
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        f4d = b.layer4(x)
        f2d = torch.flatten(b.avgpool(f4d), 1)
        return f4d, f2d


def create_resnet152(
    num_classes: int = 100,
    pretrained: bool = True,
    small_input: bool = False,
    cfg: dict | None = None,
):
    return ResNet152Teacher(
        pretrained=pretrained, num_classes=num_classes, small_input=small_input, cfg=cfg
    )

# backward compatibility
MODEL_REGISTRY["resnet_teacher"] = ResNet152Teacher
