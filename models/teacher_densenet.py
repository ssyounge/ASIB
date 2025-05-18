"""
teacher_densenet.py

- DenseNet (e.g., densenet121) Teacher
- CIFAR-100에 맞춰 마지막 classifier 교체
- TeacherDenseNetWrapper: (feat, logit, ce_loss) 반환
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights

class TeacherDenseNetWrapper(nn.Module):
    """
    Teacher 모델(DenseNet-121) 래퍼
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.criterion_ce = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # DenseNet: forward -> features -> classifier
        with torch.no_grad():
            fx = self.backbone.features(x)       # [N, 1024, H', W']
            fx = F.relu(fx, inplace=True)
            fx = F.adaptive_avg_pool2d(fx, (1,1))
            feat = fx.flatten(1)                # shape: [N, 1024]

        logit = self.backbone.classifier(feat)   # shape: [N, 100]
        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        return feat, logit, ce_loss


def create_densenet121_for_cifar100(pretrained=True):
    """
    DenseNet-121 -> classifier => CIFAR-100
    """
    if pretrained:
        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    else:
        model = densenet121(weights=None)

    in_feats = model.classifier.in_features  # 1024
    model.classifier = nn.Linear(in_feats, 100)
    return model
