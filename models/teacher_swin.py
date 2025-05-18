"""
teacher_swin.py

- Swin Transformer (Tiny, Small, etc.)를 Teacher로 사용
- CIFAR-100에 맞춰 마지막 head를 교체하고,
- TeacherSwinWrapper에서 (feat, logit, ce_loss) 반환
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_t, Swin_T_Weights


class TeacherSwinWrapper(nn.Module):
    """
    Swin Transformer Teacher Wrapper
    - forward_features(x)로 중간 feat 추출
    - self.backbone.head(feat)로 최종 logit
    - ce_loss(선택)
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone  # pretrained swin
        self.criterion_ce = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # Swin: "forward_features(x)" -> [N,C,H,W], head(feat) -> logit
        with torch.no_grad():
            feat_map = self.backbone.forward_features(x)   # [N, C, H, W]
            # Global avgpool => [N, C]
            feat = F.adaptive_avg_pool2d(feat_map, (1,1)).flatten(1)

        # 최종 분류
        logit = self.backbone.head(feat)

        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        return feat, logit, ce_loss


def create_swin_t_for_cifar100(pretrained=True):
    """
    Swin Tiny를 불러와서 head.out_features=100으로 교체
    """
    if pretrained:
        # Swin_T_Weights.DEFAULT (PyTorch 2.0~) 
        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    else:
        model = swin_t(weights=None)

    # head.in_features를 확인 후, CIFAR-100으로 교체
    in_ch = model.head.in_features
    model.head = nn.Linear(in_ch, 100)

    return model
