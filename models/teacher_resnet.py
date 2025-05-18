"""
models/teacher_resnet.py

- ResNet101를 CIFAR-100에 맞게 불러오는 함수
- TeacherResNetWrapper: forward 시 중간 feat, logit, CE loss 반환
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet101, ResNet101_Weights

class TeacherResNetWrapper(nn.Module):
    """
    Teacher 모델(ResNet101)의 forward를 감싸서
    (feat, logit, ce_loss) 튜플을 쉽게 얻을 수 있도록 하는 래퍼
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.criterion_ce = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # ResNet101 standard forward, but we want to extract "feat" before fc
        # 1) Stem
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # 2) Layers
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # 3) Global pooling
        x = self.backbone.avgpool(x)
        feat = torch.flatten(x, 1)  # shape: (N, 2048)

        # 4) FC => logit
        logit = self.backbone.fc(feat)

        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        return feat, logit, ce_loss


def create_resnet101_for_cifar100(pretrained=True):
    """
    ResNet101를 불러와서 마지막 FC를 CIFAR-100용으로 교체
    pretrained=True면 ImageNet weights 사용
    """
    if pretrained:
        model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    else:
        model = resnet101(weights=None)

    # 마지막 FC out_features=100 으로 수정
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)
    return model
