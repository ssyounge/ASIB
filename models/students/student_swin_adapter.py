import torch
import torch.nn as nn
from typing import Optional
from torchvision.models import swin_t, Swin_T_Weights

class StudentSwinWrapper(nn.Module):
    """Wrap a Swin Tiny backbone for student distillation."""
    def __init__(self, backbone: nn.Module, cfg: Optional[dict] = None):
        super().__init__()
        self.backbone = backbone
        self.criterion_ce = nn.CrossEntropyLoss()
        self.feat_dim = backbone.head.in_features
        self.feat_channels = self.feat_dim

    def forward(self, x, y=None):
        out = self.backbone.features(x)
        out = self.backbone.norm(out)
        out = self.backbone.permute(out)
        out = self.backbone.avgpool(out)
        feat_2d = self.backbone.flatten(out)
        logit = self.backbone.head(feat_2d)

        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        feat_4d = self.backbone.features(x)
        feat_4d = self.backbone.norm(feat_4d)
        feat_4d = self.backbone.permute(feat_4d)

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


def create_swin_adapter(num_classes=100, pretrained=True, small_input=False, cfg: Optional[dict] = None):
    if pretrained:
        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    else:
        model = swin_t(weights=None)

    in_feats = model.head.in_features
    model.head = nn.Linear(in_feats, num_classes)
    return StudentSwinWrapper(model, cfg=cfg)
