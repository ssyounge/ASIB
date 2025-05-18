
"""
teacher_efficientnet.py

- EfficientNet-B2를 CIFAR-100에 맞게 가져오는 함수
- TeacherEfficientNetWrapper: (feat, logit, ce_loss) 반환
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import efficientnet_b2

class TeacherEfficientNetWrapper(nn.Module):
    """
    Teacher 모델(EfficientNet-B2) forward 래퍼
    - 이때, self.backbone(x) => logit
    - 중간 feat: 'features' 블록 출력 -> adaptive_avg_pool2d -> flatten
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.criterion_ce = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # 최종 로짓
        logit = self.backbone(x)

        # 중간 feat
        # (EfficientNet은 self.backbone.features(x)를 통해 추출)
        with torch.no_grad():
            fx = self.backbone.features(x)      # shape: [N, 1408, H', W']
            fx = F.adaptive_avg_pool2d(fx, (1,1))
            feat = fx.flatten(1)               # shape: (N, 1408)

        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        return feat, logit, ce_loss


def create_efficientnet_b2_for_cifar100(pretrained=True):
    """
    Torchvision의 efficientnet_b2(weights=...) 로드
    마지막 classifier를 CIFAR-100으로 교체
    """
    if pretrained:
        from torchvision.models import EfficientNet_B2_Weights
        w = EfficientNet_B2_Weights.IMAGENET1K_V1
        model = efficientnet_b2(weights=w)
    else:
        model = efficientnet_b2(weights=None)

    # classifier의 마지막 Linear out_features = 100
    in_feats = model.classifier[1].in_features  # 1408
    model.classifier[1] = nn.Linear(in_feats, 100)

    return model
