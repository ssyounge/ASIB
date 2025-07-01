import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class StudentEfficientNetWrapper(nn.Module):
    """Wrap an EfficientNet backbone for student distillation."""
    def __init__(self, backbone, cfg: Optional[dict] = None):
        super().__init__()
        self.backbone = backbone
        self.criterion_ce = nn.CrossEntropyLoss()
        self.feat_dim = 1408
        self.feat_channels = 1408

    def forward(self, x, y=None):
        feat_layer1 = None
        feat_layer2 = None
        feat_layer3 = None
        out = x
        for idx, block in enumerate(self.backbone.features):
            out = block(out)
            if idx == 2:
                feat_layer1 = out
            elif idx == 4:
                feat_layer2 = out
            elif idx == 6:
                feat_layer3 = out
        feat_4d = out
        fpool = F.adaptive_avg_pool2d(feat_4d, (1,1)).flatten(1)
        logit = self.backbone.classifier(fpool)

        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        return {
            "feat_4d": feat_4d,
            "feat_2d": fpool,
            "logit": logit,
            "ce_loss": ce_loss,
            "feat_4d_layer1": feat_layer1,
            "feat_4d_layer2": feat_layer2,
            "feat_4d_layer3": feat_layer3,
        }

    def get_feat_dim(self):
        return self.feat_dim

    def get_feat_channels(self):
        return self.feat_channels


def create_efficientnet_adapter(num_classes=100, pretrained=True, small_input=False, cfg: Optional[dict] = None, dropout_p=0.3):
    if pretrained:
        model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
    else:
        model = efficientnet_b2(weights=None)

    if small_input:
        old_conv = model.features[0][0]
        new_conv = nn.Conv2d(
            old_conv.in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=1,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        new_conv.weight.data.copy_(old_conv.weight.data)
        if old_conv.bias is not None:
            new_conv.bias.data.copy_(old_conv.bias.data)
        model.features[0][0] = new_conv

    in_feats = model.classifier[1].in_features
    model.classifier[0] = nn.Dropout(p=dropout_p)
    model.classifier[1] = nn.Linear(in_feats, num_classes)
    return StudentEfficientNetWrapper(model, cfg=cfg)
