"""
teacher_vit.py

- Vision Transformer(ViT-B/16 등)를 Teacher로 사용
- CIFAR-100에 맞게 마지막 head 교체
- TeacherViTWrapper에서 (feat, logit, ce_loss) 반환
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights


class TeacherViTWrapper(nn.Module):
    """
    ViT Teacher Wrapper
    - forward_features(x)로 중간 feat
    - model.head(feat) -> logit
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.criterion_ce = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        with torch.no_grad():
            feat = self.backbone.forward_features(x)  # [N, hidden_dim=768, ...]
            # torchvision의 ViT는 forward_features가 (N,768), (or [N,768,14,14]형태)
            # if 2D, avgpool or flatten
            if feat.dim() == 3:  # e.g. (N, seq_len, hidden_dim)
                # cls token만 쓰거나 mean pool 등 필요
                # 여기서는 cls token만 가져온다고 가정
                feat = feat[:, 0]  # shape: [N, hidden_dim]
            elif feat.dim() == 4:
                # (N,768,H,W)
                feat = F.adaptive_avg_pool2d(feat, (1,1)).flatten(1)

        logit = self.backbone.head(feat)
        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)
        return feat, logit, ce_loss


def create_vit
