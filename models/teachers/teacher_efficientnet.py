# teacher_efficientnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class TeacherEfficientNetWrapper(nn.Module):
    """
    Teacher 모델(EfficientNet-B2) forward 래퍼
    => (feat, logit, ce_loss) 반환
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.criterion_ce = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # 최종 로짓
        logit = self.backbone(x)

        # 중간 feat
        with torch.no_grad():
            fx = self.backbone.features(x)  # shape: [N, 1408, H', W']
            fx = F.adaptive_avg_pool2d(fx, (1,1))
            feat = fx.flatten(1)           # shape: (N, 1408)

        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        return feat, logit, ce_loss


def create_efficientnet_b2(num_classes=100, pretrained=True):
    """
    Loads EfficientNet-B2, optionally pretrained on ImageNet1K.
    => Replaces final classifier with (in_feats -> num_classes).
    => Suitable for CIFAR-100 or ImageNet100 (both 100 classes).
    """
    if pretrained:
        model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
    else:
        model = efficientnet_b2(weights=None)

    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)

    return model
