# models/teachers/convnext_l_teacher.py

import torch.nn as nn
from typing import Optional

import timm

from models.common.base_wrapper import BaseKDModel


class ConvNeXtLTeacher(BaseKDModel):
    """ConvNeXt-Large teacher (22K pre-train)."""

    def __init__(
        self,
        *,
        num_classes: int = 100,
        pretrained: bool = True,
        small_input: bool = False,
        cfg: Optional[dict] = None,
    ) -> None:
        backbone = timm.create_model(
            "convnext_large_in22k",
            pretrained=pretrained,
            num_classes=num_classes,
        )
        if small_input:
            # CIFAR-style 32x32 inputs remove initial downsample
            backbone.stem[0].stride = (1, 1)
            backbone.stem[1] = nn.Identity()

        super().__init__(backbone, num_classes, role="teacher", cfg=cfg or {})

    # ------------------------------------------------------------------
    def extract_feats(self, x):
        feat_4d = self.backbone.forward_features(x)
        feat_2d = feat_4d.mean([-2, -1])
        return feat_4d, feat_2d

# (등록은 registry_map.yaml에서 수행)


def create_convnext_l(
    num_classes: int = 100,
    pretrained: bool = True,
    small_input: bool = False,
    cfg: Optional[dict] = None,
) -> ConvNeXtLTeacher:
    """Build :class:`ConvNeXtLTeacher`."""
    return ConvNeXtLTeacher(
        num_classes=num_classes,
        pretrained=pretrained,
        small_input=small_input,
        cfg=cfg,
    )


