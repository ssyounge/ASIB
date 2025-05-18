"""
models/student_densenet_adapter.py

- DenseNet-121 Student + Adapter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights

class ExtendedAdapterDenseNet121(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model

        # base_model.features(x) => [N,1024,H,W]
        # 여기서 adapter block
        self.adapter_conv1 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.adapter_gn1   = nn.GroupNorm(32, 512)
        self.adapter_conv2 = nn.Conv2d(512, 1024, kernel_size=1, bias=False)
        self.adapter_gn2   = nn.GroupNorm(32, 1024)
        self.adapter_relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        fx = self.backbone.features(x)  # [N,1024,H,W]
        fx = F.relu(fx, inplace=True)

        # adapter
        xa = self.adapter_conv1(fx)
        xa = self.adapter_gn1(xa)
        xa = self.adapter_relu(xa)
        xa = self.adapter_conv2(xa)
        xa = self.adapter_gn2(xa)
        fx = fx + xa
        fx = self.adapter_relu(fx)

        fx = F.adaptive_avg_pool2d(fx, (1,1))
        feat = fx.flatten(1)
        out = self.backbone.classifier(feat)  # shape: [N,100]
        return out


def create_densenet121_with_adapter(pretrained=True):
    if pretrained:
        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    else:
        model = densenet121(weights=None)

    in_feats = model.classifier.in_features
    model.classifier = nn.Linear(in_feats, 100)

    student_model = ExtendedAdapterDenseNet121(model)
    return student_model
