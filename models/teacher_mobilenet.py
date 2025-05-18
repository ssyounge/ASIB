"""
teacher_mobilenet.py

- MobileNetV2 or V3를 Teacher로 사용
- CIFAR-100에 맞춰 마지막 classifier를 교체
- TeacherMobileNetWrapper: (feat, logit, ce_loss) 반환
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class TeacherMobileNetWrapper(nn.Module):
    """
    Teacher 모델(MobileNet V2) 래퍼
    - features(x) -> feat
    - classifier(feat) -> logit
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone  # mobilenet_v2
        self.criterion_ce = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        with torch.no_grad():
            fx = self.backbone.features(x)        # shape: [N,1280,H',W']
            fx = F.adaptive_avg_pool2d(fx, (1,1)) 
            feat = fx.flatten(1)                  # shape: [N,1280]

        logit = self.backbone.classifier(feat)    # shape: [N,100]
        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        return feat, logit, ce_loss


def create_mobilenet_v2_for_cifar100(pretrained=True):
    """
    MobileNetV2 -> 마지막 classifier => CIFAR-100
    """
    if pretrained:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    else:
        model = mobilenet_v2(weights=None)

    # classifier[-1] 은 보통 nn.Linear(1280->1000) 구조
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, 100)

    return model
