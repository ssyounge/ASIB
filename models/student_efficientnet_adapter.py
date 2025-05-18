"""
student_efficientnet_adapter.py

- EfficientNet을 Student로 사용하되, Adapter 모듈 삽입
- ExtendedAdapterEffNetB2: 
   - self.backbone.features(x) -> 중간 레이어 뒤에 adapter
   - self.backbone.classifier => 최종 (100-d)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class ExtendedAdapterEffNetB2(nn.Module):
    """
    EfficientNet-B2 + Adapter
    - features[...somewhere...] 뒤에 adapter를 삽입하는 예시
    """
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model

        # base_model.features => Stage별로 나뉘어 있음
        # 간단 구현: features 전부 끝까지 호출 후, adapterConv -> classifier
        # (원한다면 중간 stage 사이에 adapter를 삽입하는 게 더 바람직)
        
        # 예시로 adapter 레이어(Conv+GN+Conv...)를 만들고 features 출력에 추가
        self.adapter_conv1 = nn.Conv2d(1408, 512, kernel_size=1, bias=False)
        self.adapter_gn1   = nn.GroupNorm(32, 512)
        self.adapter_conv2 = nn.Conv2d(512, 1408, kernel_size=1, bias=False)
        self.adapter_gn2   = nn.GroupNorm(32, 1408)
        self.adapter_relu  = nn.ReLU(inplace=True)

        # classifier는 base_model.classifier 사용
        # (마지막 Linear(1408->100))

    def forward(self, x):
        # 1) EfficientNet features
        fx = self.backbone.features(x)  # shape: (N,1408,H',W')

        # 2) Adapter
        xa = self.adapter_conv1(fx)
        xa = self.adapter_gn1(xa)
        xa = self.adapter_relu(xa)
        xa = self.adapter_conv2(xa)
        xa = self.adapter_gn2(xa)

        # skip connection
        fx = fx + xa
        fx = self.adapter_relu(fx)

        # 3) Pool + classifier
        # EfficientNet: self.backbone.avgpool 생략 -> torch.no_grad()?
        # 보통 efficientnet_b2는 self.backbone.forward()가 다음을 실행:
        #  - adaptiveavgpool
        #  - flatten
        #  - classifier(...)

        # 여기서는 수동으로 해보기
        out = F.adaptive_avg_pool2d(fx, (1,1))
        out = out.flatten(1)
        out = self.backbone.classifier(out)  # shape: (N,100)
        return out


def create_efficientnet_b2_with_adapter(pretrained=True):
    """
    (1) efficientnet_b2 불러오기
    (2) last classifier -> CIFAR-100
    (3) ExtendedAdapterEffNetB2(base_model) 래핑
    """
    if pretrained:
        model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
    else:
        model = efficientnet_b2(weights=None)

    # 마지막 classifier Linear(1408->100)
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, 100)

    # 어댑터 삽입
    student_model = ExtendedAdapterEffNetB2(model)
    return student_model
