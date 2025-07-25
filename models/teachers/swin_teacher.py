# models/teachers/swin_teacher.py

import torch.nn as nn
from typing import Optional
from models.common.base_wrapper import BaseKDModel, register
from torchvision.models import swin_t, Swin_T_Weights

@register("swin_teacher")
class SwinTeacher(BaseKDModel):
    """Wrap a torchvision Swin Tiny model for distillation.

    The official ``forward`` for a Swin Transformer is::

        features -> norm -> permute -> avgpool -> flatten -> head

    This wrapper follows that sequence and also exposes the feature map
    just before pooling as ``feat_4d`` ([N, C, H, W]) for compatibility with
    the other teachers.
    """
    def __init__(self, backbone: nn.Module, *, num_classes: int,
                 cfg: Optional[dict] = None):
        super().__init__(backbone, num_classes, role="teacher", cfg=cfg or {})

    
    def extract_feats(self, x):
        # Official Swin forward: features -> norm -> permute -> avgpool -> flatten
        out = self.backbone.features(x)
        out = self.backbone.norm(out)
        out = self.backbone.permute(out)
        out = self.backbone.avgpool(out)
        feat_2d = self.backbone.flatten(out)

        # Recompute 4D feature before pooling for compatibility
        feat_4d_for_compat = self.backbone.features(x)
        feat_4d_for_compat = self.backbone.norm(feat_4d_for_compat)
        feat_4d_for_compat = self.backbone.permute(feat_4d_for_compat)

        return feat_4d_for_compat, feat_2d

def create_swin_t(
    num_classes=100,
    pretrained=True,
    cfg: Optional[dict] = None,
):
    """Load Swin Tiny and wrap it with ``SwinTeacher``."""
    if pretrained:
        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    else:
        model = swin_t(weights=None)

    in_ch = model.head.in_features
    model.head = nn.Linear(in_ch, num_classes)

    teacher_model = SwinTeacher(model, num_classes=num_classes, cfg=cfg)
    return teacher_model
