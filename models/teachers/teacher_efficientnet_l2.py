# models/teachers/teacher_efficientnet_l2.py

import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import timm
from .adapters import DistillationAdapter


class TeacherEfficientNetL2Wrapper(nn.Module):
    """EfficientNet-L2 (Noisy Student) Teacher wrapper."""

    def __init__(self, backbone: nn.Module, cfg: Optional[dict] = None):
        super().__init__()
        self.backbone = backbone
        self.criterion_ce = nn.CrossEntropyLoss()

        # timm exposes ``num_features`` for EfficientNet models
        self.feat_dim = backbone.num_features
        self.feat_channels = self.feat_dim

        cfg = cfg or {}
        self.distillation_adapter = DistillationAdapter(
            self.feat_dim,
            hidden_dim=cfg.get("distill_hidden_dim"),
            out_dim=cfg.get("distill_out_dim"),
            cfg=cfg,
        )
        self.distill_dim = self.distillation_adapter.out_dim

    def _forward_backbone(self, x):
        """Return 4D feature map using timm's forward_features."""
        f4d = self.backbone.forward_features(x)
        return f4d

    def forward(self, x, y=None):
        f4d = self._forward_backbone(x)
        fpool = F.adaptive_avg_pool2d(f4d, 1).flatten(1)
        logit = self.backbone.get_classifier()(fpool)

        distill_feat = self.distillation_adapter(fpool)

        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        return {
            "feat_4d": f4d,
            "feat_2d": fpool,
            "distill_feat": distill_feat,
            "logit": logit,
            "ce_loss": ce_loss,
        }

    def get_feat_dim(self):
        return self.feat_dim

    def get_feat_channels(self):
        return self.feat_channels


def create_efficientnet_l2(
    num_classes: int = 100,
    pretrained: bool = True,
    small_input: bool = False,
    dropout_p: float = 0.4,
    cfg: Optional[dict] = None,
):
    """Create an EfficientNet-L2 (Noisy Student) teacher via timm."""
    model_name = "tf_efficientnet_l2_ns"
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout_p,
    )

    if small_input:
        stem = backbone.conv_stem
        new_stem = nn.Conv2d(
            stem.in_channels,
            stem.out_channels,
            kernel_size=stem.kernel_size,
            stride=1,
            padding=stem.padding,
            bias=stem.bias is not None,
        )
        new_stem.weight.data.copy_(stem.weight.data)
        if stem.bias is not None:
            new_stem.bias.data.copy_(stem.bias.data)
        backbone.conv_stem = new_stem

    return TeacherEfficientNetL2Wrapper(backbone, cfg)
