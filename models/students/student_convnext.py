# students/student_convnext.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from timm.models.convnext import convnext_tiny, convnext_small

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


def create_convnext_tiny(
    num_classes: int = 100,
    pretrained: bool = True,
    small_input: bool = True,
    cfg: Optional[dict] = None,
    **kwargs,
):
    """timm convnext_tiny wrapper with optional CIFAR-friendly stem."""
    model = convnext_tiny(
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs,
    )
    if small_input:
        # stem-patch 4x4 already works for 32x32 -> no change
        pass

    return StudentConvNeXtWrapper(model, cfg=cfg)


# -------------------------------------------------------
# NEW: ConvNeXt-Small (≈50 M parameters)
# -------------------------------------------------------
def create_convnext_small(
    num_classes: int = 100,
    pretrained: bool = True,
    small_input: bool = True,
    cfg: Optional[dict] = None,
    **kwargs,
):
    """timm convnext_small wrapper aligned with Tiny helper."""
    model = convnext_small(
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs,
    )
    if small_input:
        # stem-patch 4x4 already works for 32x32 -> no change
        pass

    return StudentConvNeXtWrapper(model, cfg=cfg)
