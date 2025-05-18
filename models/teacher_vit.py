"""
teacher_vit.py

- Vision Transformer(ViT-B/16 등) Teacher
- CIFAR-100에 맞춰 마지막 head 교체
- TeacherViTWrapper: (feat, logit, ce_loss) tuple 반환
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

class TeacherViTWrapper(nn.Module):
    """
    Teacher 모델(ViT) 래퍼
    - model.forward_features(x) -> (N,768) or (N,seq_len,768) ...
    - model.head(feat) -> logit
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.criterion_ce = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # ViT forward_features => 토큰 시퀀스 반환할 수 있음
        with torch.no_grad():
            feat_out = self.backbone.forward_features(x)
            # torchvision의 ViT: forward_features(x)-> (N,768,14,14) 혹은 (N,seq,dim)
            if feat_out.dim() == 4:
                # (N, hidden_dim, H, W)
                feat = F.adaptive_avg_pool2d(feat_out, (1,1)).flatten(1)
            elif feat_out.dim() == 3:
                # (N, seq_len, hidden_dim), 보통 cls token = feat_out[:,0]
                feat = feat_out[:,0]
            else:
                # 혹시 모르는 case
                feat = feat_out

        logit = self.backbone.head(feat)
        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)
        return feat, logit, ce_loss


def create_vit_b16_for_cifar100(pretrained=True):
    """
    Vision Transformer (ViT-B/16) 모델 불러온 뒤, CIFAR-100 용으로 head 교체
    """
    if pretrained:
        # PyTorch>=2.0 예시: ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    else:
        model = vit_b_16(weights=None)

    # head.in_features => 100
    in_ch = model.head.in_features
    model.head = nn.Linear(in_ch, 100)
    return model
