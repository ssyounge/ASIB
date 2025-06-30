# models/teachers/teacher_swin.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .adapters import DistillationAdapter
from torchvision.models import swin_t, Swin_T_Weights

class TeacherSwinWrapper(nn.Module):
    """
    Teacher 모델(Swin Tiny) forward:
     => dict 반환 {"feat_4d", "feat_2d", "logit", "ce_loss"}
    feature_dict 예시:
      {
        "feat_4d": [N, C, H, W],  # backbone.forward_features(x) or unsqueezed
        "feat_2d": [N, C],       # global pooled or direct features
      }
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
        self.distillation_adapter = DistillationAdapter(
            self.feat_dim, cfg=cfg
        )
        self.distill_dim = self.distillation_adapter.out_dim

    
    def forward(self, x, y=None):
        # 1) backbone forward => feature tensor
        # keep gradients so Swin parameters remain trainable during teacher adaptation
        if hasattr(self.backbone, "forward_features"):
            x_features = self.backbone.forward_features(x)
        elif hasattr(self.backbone, "features"):
            x_features = self.backbone.features(x)
        else:
            raise AttributeError(
                "Backbone model must implement forward_features or features"
            )

        # 2) handle feature shape
        if x_features.dim() == 2:
            # already pooled -> treat as 2D feature
            f2d = x_features
            feat_4d = f2d.unsqueeze(-1).unsqueeze(-1)
        elif x_features.dim() == 3:
            # Swin Tiny may return [N, seq_len, C] or [N, C, seq_len]
            x_features = (
                self.backbone.norm(x_features)
                if hasattr(self.backbone, "norm")
                else x_features
            )
            if x_features.shape[1] == self.feat_dim:
                # [N, C, seq_len] => average over sequence dimension
                f2d = x_features.mean(dim=2)
            else:
                # [N, seq_len, C] => average over sequence dimension
                f2d = x_features.mean(dim=1)
            feat_4d = f2d.unsqueeze(-1).unsqueeze(-1)
        else:
            # standard 4D [N, C, H, W]
            feat_4d = x_features
            f2d = F.adaptive_avg_pool2d(x_features, (1, 1)).flatten(1)

        # distillation adapter feature
        distill_feat = self.distillation_adapter(f2d)

        # 3) head => logit
        logit = self.backbone.head(f2d)

        # (optional) CE loss
        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        # Dict
        return {
            "feat_4d": feat_4d,  # [N, C, H, W]
            "feat_2d": f2d,  # [N, C]
            "distill_feat": distill_feat,
            "logit": logit,
            "ce_loss": ce_loss,
            # expose identical keys for compatibility
            "feat_4d_layer1": feat_4d,
            "feat_4d_layer2": feat_4d,
            "feat_4d_layer3": feat_4d,
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
