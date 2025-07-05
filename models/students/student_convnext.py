# students/student_convnext.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torchvision.models import (
    convnext_tiny,  ConvNeXt_Tiny_Weights,
    convnext_small, ConvNeXt_Small_Weights,
)

__all__ = [
    "create_convnext_tiny",
    "create_convnext_small",
]

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
        # the ConvNeXt classifier expects a 4D tensor as input for the initial
        # normalization layer (``LayerNorm2d`` in torchvision). We therefore
        # pass the pooled feature map directly to ``classifier[0]`` and then
        # flatten the normalized output.
        normed = self.backbone.classifier[0](pooled)
        feat_2d = self.backbone.classifier[1](normed)
        logit = self.backbone.classifier[2](feat_2d)

        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        feat_dict = {
            "feat_4d": feat_4d,
            "feat_2d": feat_2d,
        }
        # cache last 2D feature for optional retrieval
        self._cached_feat = feat_dict["feat_2d"]
        return feat_dict, logit, None

    def get_feat_dim(self):
        return self.feat_dim

    def get_feat_channels(self):
        return self.feat_channels

    def get_feat(self):
        """Return last cached feature map (feat_2d) saved in forward()."""
        return getattr(self, "_cached_feat", None)


# optional: keep stride 4 (original) – accuracy 일반적으로 ↑
def _patch_cifar_stem(model: nn.Module, stride2: bool = False):
    if not stride2:      # stride 4 유지 → 바로 return
        return
    conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=2,         # only if stride2=True
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    model.features[0][0].weight.data.copy_(conv.weight.data)
    if conv.bias is not None:
        model.features[0][0].bias.data.copy_(conv.bias.data)


def create_convnext_tiny(
    num_classes: int = 100,
    pretrained: bool = True,
    small_input: bool = True,
    cfg: Optional[dict] = None,
):
    w = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    if pretrained:
        # ① 가중치만 불러오고, num_classes는 건드리지 않는다.
        model = convnext_tiny(weights=w)              # 1000‑cls head
    else:
        model = convnext_tiny(weights=None, num_classes=num_classes)

    if small_input:
        _patch_cifar_stem(model, stride2=cfg.get("patch_stride2", False))

    in_feats = model.classifier[2].in_features  # 헤드 교체
    model.classifier[2] = nn.Linear(in_feats, num_classes)
    return StudentConvNeXtWrapper(model, cfg=cfg)


# -------------------------------------------------------
# ConvNeXt‑Small (≈50 M params)
# -------------------------------------------------------
def create_convnext_small(
    num_classes: int = 100,
    pretrained: bool = True,
    small_input: bool = True,
    cfg: Optional[dict] = None,
):
    w = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
    if pretrained:
        model = convnext_small(weights=w)
    else:
        model = convnext_small(weights=None, num_classes=num_classes)

    if small_input:
        _patch_cifar_stem(model, stride2=cfg.get("patch_stride2", False))

    in_feats = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_feats, num_classes)
    return StudentConvNeXtWrapper(model, cfg=cfg)
