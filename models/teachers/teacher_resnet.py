# models/teachers/teacher_resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torchvision.models import (
    resnet101,
    ResNet101_Weights,
    resnet152,
    ResNet152_Weights,
)

class TeacherResNetWrapper(nn.Module):
    """
    Teacher 모델(ResNet101) forward:
     => dict 반환 {"feat_4d", "feat_2d", "logit", "ce_loss"}
    feature_dict 예시:
      {
        "feat_4d": [N, 2048, H, W],   # layer4까지의 4D 출력
        "feat_2d": [N, 2048],        # global pooled
      }
    """
    def __init__(self, backbone: nn.Module, cfg: Optional[dict] = None):
        super().__init__()
        self.backbone = backbone
        self.criterion_ce = nn.CrossEntropyLoss()

        # 추가: ResNet101의 글로벌 피처 차원 (기본 2048)
        self.feat_dim = 2048
        self.feat_channels = 2048

        # distillation adapter removed
    
    def forward(self, x, y=None):
        """
        Returns:
            feat_dict : {"feat_2d": tensor, ...}
            logits    : Tensor of shape (N, C)
            ce_loss   : scalar or None
        """
        # 1) stem
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # 2) layers
        x = self.backbone.layer1(x)
        feat_layer1 = x
        x = self.backbone.layer2(x)
        feat_layer2 = x
        x = self.backbone.layer3(x)
        feat_layer3 = x
        f4d = self.backbone.layer4(x)  # [N, 2048, H, W]

        # 3) global pool => 2D
        gp = self.backbone.avgpool(f4d)  # [N, 2048, 1, 1]
        feat_2d = torch.flatten(gp, 1)   # [N, 2048]

        # 4) fc => logit
        logit = self.backbone.fc(feat_2d)

        # (optional) CE loss
        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        feat_dict = {
            "feat_4d": f4d,
            "feat_2d": feat_2d,
            "feat_4d_layer1": feat_layer1,
            "feat_4d_layer2": feat_layer2,
            "feat_4d_layer3": feat_layer3,
        }
        return feat_dict, logit, ce_loss

    def get_feat_dim(self):
        """
        Returns the dimension of the 2D feature (feat_2d).
        ResNet101 => 2048
        """
        return self.feat_dim

    def get_feat_channels(self):
        """Channel dimension of the 4D feature."""
        return self.feat_channels

def create_resnet101(
    num_classes=100,
    pretrained=True,
    small_input=False,
    cfg: Optional[dict] = None,
):
    """
    ResNet101 로드 후 stem을 optional로 CIFAR-friendly 형태로 바꾸고,
    마지막 FC 교체 => TeacherResNetWrapper
    """
    if pretrained:
        model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    else:
        model = resnet101(weights=None)

    if small_input:
        # 32x32 input 등에 맞게 3x3 conv + stride1, maxpool 제거
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)

    teacher_model = TeacherResNetWrapper(model, cfg=cfg)
    return teacher_model


def create_resnet152(
    num_classes=100,
    pretrained=True,
    small_input=False,
    cfg: Optional[dict] = None,
):
    """Create a ResNet152 teacher wrapped with ``TeacherResNetWrapper``."""
    if pretrained:
        model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
    else:
        model = resnet152(weights=None)

    if small_input:
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)

    teacher_model = TeacherResNetWrapper(model, cfg=cfg)
    return teacher_model
