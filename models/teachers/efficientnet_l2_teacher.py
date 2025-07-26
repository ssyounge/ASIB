# models/teachers/efficientnet_l2_teacher.py

import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import timm
from models.common.base_wrapper import BaseKDModel, register


@register("efficientnet_l2_teacher")
class EfficientNetL2Teacher(BaseKDModel):
    """EfficientNet-L2 (Noisy Student) Teacher."""

    def __init__(self, backbone: nn.Module, *, num_classes: int,
                 cfg: Optional[dict] = None):
        super().__init__(backbone, num_classes, role="teacher", cfg=cfg or {})

    def extract_feats(self, x):
        f4d = self.backbone.forward_features(x)
        f2d = F.adaptive_avg_pool2d(f4d, 1).flatten(1)
        return f4d, f2d


def create_efficientnet_l2(
    num_classes: int = 100,
    pretrained: bool = True,
    small_input: bool = False,
    dropout_p: float = 0.4,
    use_checkpointing: bool = False,
    cfg: Optional[dict] = None,
):
    """Create an EfficientNet-L2 (Noisy Student) teacher via timm."""
    model_name = "tf_efficientnet_l2_ns"
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout_p,
    )

    if use_checkpointing:
        from timm.models.layers import checkpoint_seq
        backbone.blocks = checkpoint_seq(backbone.blocks)

    if small_input:
        stem = backbone.conv_stem
        new_stem = nn.Conv2d(
            stem.in_channels,
            stem.out_channels,
            kernel_size=stem.kernel_size,
            stride=1,
            padding=stem.padding,
            bias=stem.bias is not None,
        )
        new_stem.weight.data.copy_(stem.weight.data)
        if stem.bias is not None:
            new_stem.bias.data.copy_(stem.bias.data)
        backbone.conv_stem = new_stem

    return EfficientNetL2Teacher(backbone, num_classes=num_classes, cfg=cfg)
