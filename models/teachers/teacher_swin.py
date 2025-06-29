# models/teachers/teacher_swin.py

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.criterion_ce = nn.CrossEntropyLoss()

        # Swin Tiny의 head.in_features => 768 (기본)
        #   (모델마다 다를 수 있음)
        self.feat_dim = self.backbone.head.in_features
        self.feat_channels = self.feat_dim

        # distillation adapter
        self.distillation_adapter = DistillationAdapter(self.feat_dim)
        self.distill_dim = self.distillation_adapter.out_dim

    
    def forward(self, x, y=None):
        # 1) Swin forward => 4D feature
        # use gradients so that Swin parameters remain trainable during
        # teacher adaptation
        if hasattr(self.backbone, "forward_features"):
            f4d = self.backbone.forward_features(x)
        elif hasattr(self.backbone, "features"):
            f4d = self.backbone.features(x)
        else:
            raise AttributeError(
                "Backbone model must implement forward_features or features"
            )

        # 2) handle feature shape
        if f4d.dim() == 2:
            f2d = f4d
            feat_4d = f2d.unsqueeze(-1).unsqueeze(-1)
        else:
            feat_4d = f4d
            f2d = F.adaptive_avg_pool2d(f4d, (1, 1)).flatten(1)

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
        }
        
    def get_feat_dim(self):
        """
        Swin Tiny => usually 768
        """
        return self.feat_dim

    def get_feat_channels(self):
        """Channel dimension of the 4D feature."""
        return self.feat_channels

def create_swin_t(num_classes=100, pretrained=True):
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

    teacher_model = TeacherSwinWrapper(model)
    return teacher_model
