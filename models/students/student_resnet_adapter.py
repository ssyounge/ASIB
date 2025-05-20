"""
models/student_resnet_adapter.py

- Student로 사용할 ResNet101 + Adapter 구조를 정의
- create_resnet101_with_extended_adapter 함수로 모델 생성
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet101, ResNet101_Weights

class ExtendedAdapterResNet101(nn.Module):
    """
    ResNet101에 Adapter 모듈(layer3 뒤에 삽입 등)을 추가한 예시
    """
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model

        # 기존 ResNet101 속 레이어 alias
        self.conv1   = self.backbone.conv1
        self.bn1     = self.backbone.bn1
        self.relu    = self.backbone.relu
        self.maxpool = self.backbone.maxpool

        self.layer1  = self.backbone.layer1
        self.layer2  = self.backbone.layer2
        self.layer3  = self.backbone.layer3
        self.layer4  = self.backbone.layer4

        self.avgpool = self.backbone.avgpool
        self.fc      = self.backbone.fc

        # Adapter blocks (간단한 예시)
        # => layer3 뒤에 더해준다든지
        self.adapter_conv1 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.adapter_gn1   = nn.GroupNorm(32, 512)
        self.adapter_conv2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.adapter_gn2   = nn.GroupNorm(32, 512)
        self.adapter_conv3 = nn.Conv2d(512, 1024, kernel_size=1, bias=False)
        self.adapter_gn3   = nn.GroupNorm(32, 1024)
        self.adapter_relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        # 1) Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 2) layer1, layer2, layer3
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # 3) Adapter
        xa = self.adapter_conv1(x)
        xa = self.adapter_gn1(xa)
        xa = self.adapter_relu(xa)
        xa = self.adapter_conv2(xa)
        xa = self.adapter_gn2(xa)
        xa = self.adapter_relu(xa)
        xa = self.adapter_conv3(xa)
        xa = self.adapter_gn3(xa)

        # skip connection
        x = x + xa
        x = self.adapter_relu(x)

        # 4) layer4
        x = self.layer4(x)

        # 5) Pool + FC
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        out  = self.fc(feat)
        return out


def create_resnet101_with_extended_adapter(pretrained=True):
    """
    ResNet101를 불러와서 ExtendedAdapterResNet101에 삽입
    """
    if pretrained:
        base = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    else:
        base = resnet101(weights=None)

    # FC out_features = 100 으로 수정
    num_ftrs = base.fc.in_features
    base.fc  = nn.Linear(num_ftrs, 100)

    # ExtendedAdapterResNet101로 래핑
    model = ExtendedAdapterResNet101(base)
    return model
