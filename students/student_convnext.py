import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class StudentConvNeXtWrapper(nn.Module):
    """Wrap a ConvNeXt-Tiny backbone for student distillation."""
    def __init__(self, backbone: nn.Module, cfg: Optional[dict] = None):
        super().__init__()
        self.backbone = backbone
        self.criterion_ce = nn.CrossEntropyLoss()
        # convnext tiny final dim
        self.feat_dim = backbone.classifier[2].in_features
        self.feat_channels = self.feat_dim

    def forward(self, x, y=None):
        feat_4d = self.backbone.features(x)
        pooled = self.backbone.avgpool(feat_4d)
        feat_2d = torch.flatten(pooled, 1)
        feat_2d = self.backbone.classifier[0](feat_2d)
        logit = self.backbone.classifier[2](feat_2d)

        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        return {
            "feat_4d": feat_4d,
            "feat_2d": feat_2d,
            "logit": logit,
            "ce_loss": ce_loss,
            "feat_4d_layer1": feat_4d,
            "feat_4d_layer2": feat_4d,
            "feat_4d_layer3": feat_4d,
        }

    def get_feat_dim(self):
        return self.feat_dim

    def get_feat_channels(self):
        return self.feat_channels


def create_convnext_tiny(num_classes=100, pretrained=True, small_input=False, cfg: Optional[dict] = None):
    if pretrained:
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    else:
        model = convnext_tiny(weights=None)

    if small_input:
        conv = model.features[0][0]
        new_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=2,
            padding=conv.padding,
            bias=conv.bias is not None,
        )
        new_conv.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            new_conv.bias.data.copy_(conv.bias.data)
        model.features[0][0] = new_conv

    in_feats = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_feats, num_classes)
    return StudentConvNeXtWrapper(model, cfg=cfg)
