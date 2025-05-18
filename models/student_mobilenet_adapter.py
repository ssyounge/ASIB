"""
student_mobilenet_adapter.py

- MobileNetV2를 Student로 쓰면서, Adapter 모듈 삽입
- ExtendedAdapterMobileNetV2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class ExtendedAdapterMobileNetV2(nn.Module):
    """
    Adapter를 features 출력(1280ch) 주변에 삽입하는 예시
    """
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        # base_model.features(x) => [N,1280,H,W]
        # 여기서 adapter로 conv -> skip add 등 시도 가능

        self.adapter_conv1 = nn.Conv2d(1280, 512, kernel_size=1, bias=False)
        self.adapter_gn1   = nn.GroupNorm(32, 512)
        self.adapter_conv2 = nn.Conv2d(512, 1280, kernel_size=1, bias=False)
        self.adapter_gn2   = nn.GroupNorm(32, 1280)
        self.adapter_relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        fx = self.backbone.features(x)
        xa = self.adapter_conv1(fx)
        xa = self.adapter_gn1(xa)
        xa = self.adapter_relu(xa)
        xa = self.adapter_conv2(xa)
        xa = self.adapter_gn2(xa)
        fx = fx + xa
        fx = self.adapter_relu(fx)

        # then global pool
        fx = F.adaptive_avg_pool2d(fx, (1,1))
        fx = fx.flatten(1)
        out = self.backbone.classifier(fx)
        return out


def create_mobilenet_v2_with_adapter(pretrained=True):
    """
    mobilenet_v2 => last layer => 100
    => ExtendedAdapterMobileNetV2로 감싸기
    """
    if pretrained:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    else:
        model = mobilenet_v2(weights=None)

    # classifier => 100
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, 100)

    student_model = ExtendedAdapterMobileNetV2(model)
    return student_model
