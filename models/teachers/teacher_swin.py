# models/teachers/teacher_swin.py

import torch
import torch.nn as nn
from typing import Optional
from .adapters import DistillationAdapter
from torchvision.models import swin_t, Swin_T_Weights

class TeacherSwinWrapper(nn.Module):
    """Wrap a torchvision Swin Tiny model for distillation.

    The official ``forward`` for a Swin Transformer is::

        features -> norm -> permute -> avgpool -> flatten -> head

    This wrapper follows that sequence and also exposes the feature map
    just before pooling as ``feat_4d`` ([N, C, H, W]) for compatibility with
    the other teachers.
    """
    def __init__(self, backbone: nn.Module, cfg: Optional[dict] = None):
        super().__init__()
        self.backbone = backbone
        self.criterion_ce = nn.CrossEntropyLoss()

        # Swin Tiny의 head.in_features => 768 (기본)
        #   (모델마다 다를 수 있음)
        self.feat_dim = self.backbone.head.in_features
        self.feat_channels = self.feat_dim

        # distillation adapter
        cfg = cfg or {}
        hidden_dim = cfg.get("distill_hidden_dim")
        out_dim = cfg.get("distill_out_dim")
        self.distillation_adapter = DistillationAdapter(
            self.feat_dim, hidden_dim=hidden_dim, out_dim=out_dim
        )
        self.distill_dim = self.distillation_adapter.out_dim

    
    def forward(self, x, y=None):
        """Forward pass following the official torchvision Swin order."""

        # Official Swin forward: features -> norm -> permute -> avgpool -> flatten
        out = self.backbone.features(x)
        out = self.backbone.norm(out)
        out = self.backbone.permute(out)
        out = self.backbone.avgpool(out)
        feat_2d = self.backbone.flatten(out)

        distill_feat = self.distillation_adapter(feat_2d)
        logit = self.backbone.head(feat_2d)

        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        # Recompute 4D feature before pooling for compatibility
        feat_4d_for_compat = self.backbone.features(x)
        feat_4d_for_compat = self.backbone.norm(feat_4d_for_compat)
        feat_4d_for_compat = self.backbone.permute(feat_4d_for_compat)

        return {
            "feat_4d": feat_4d_for_compat,
            "feat_2d": feat_2d,
            "distill_feat": distill_feat,
            "logit": logit,
            "ce_loss": ce_loss,
            "feat_4d_layer1": feat_4d_for_compat,
            "feat_4d_layer2": feat_4d_for_compat,
            "feat_4d_layer3": feat_4d_for_compat,
        }
        
    def get_feat_dim(self):
        """
        Swin Tiny => usually 768
        """
        return self.feat_dim

    def get_feat_channels(self):
        """Channel dimension of the 4D feature."""
        return self.feat_channels

def create_swin_t(
    num_classes=100,
    pretrained=True,
    cfg: Optional[dict] = None,
):
    """
    Swin Tiny 로드 후, head 교체 => TeacherSwinWrapper
    => (feature_dict, logit, ce_loss)
    """
    if pretrained:
        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    else:
        model = swin_t(weights=None)

    in_ch = model.head.in_features
    model.head = nn.Linear(in_ch, num_classes)

    teacher_model = TeacherSwinWrapper(model, cfg=cfg)
    return teacher_model
