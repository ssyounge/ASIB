# models/teachers/teacher_swin.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_t, Swin_T_Weights

class TeacherSwinWrapper(nn.Module):
    """
    Teacher 모델(Swin Tiny) forward:
     => (feature_dict, logit, ce_loss) 반환
    feature_dict 예시:
      {
        "feat_4d": [N, C, H, W],  # backbone.forward_features(x)
        "feat_2d": [N, C],       # global pooled
      }
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.criterion_ce = nn.CrossEntropyLoss()

        # Swin Tiny의 head.in_features => 768 (기본)
        #   (모델마다 다를 수 있음)
        self.feat_dim = self.backbone.head.in_features

    
    def forward(self, x, y=None):
        # 1) Swin forward_features => [N, C, H, W]
        # use gradients so that Swin parameters remain trainable during
        # teacher adaptation
        f4d = self.backbone.forward_features(x)  # [N, C, H, W]

        # 2) global pool => 2D
        f2d = F.adaptive_avg_pool2d(f4d, (1,1)).flatten(1)  # [N, C]

        # 3) head => logit
        logit = self.backbone.head(f2d)

        # (optional) CE loss
        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        # Dict
        feature_dict = {
            "feat_4d": f4d,  # [N, C, H, W]
            "feat_2d": f2d,  # [N, C]
        }
        return feature_dict, logit, ce_loss
        
    def get_feat_dim(self):
        """
        Swin Tiny => usually 768
        """
        return self.feat_dim

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
