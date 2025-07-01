import torch
import torch.nn as nn
from typing import Optional
from torchvision.models import resnet50, ResNet50_Weights, resnet152, ResNet152_Weights

class StudentResNetWrapper(nn.Module):
    """Wrap a ResNet backbone for student distillation."""
    def __init__(self, backbone: nn.Module, cfg: Optional[dict] = None):
        super().__init__()
        self.backbone = backbone
        self.criterion_ce = nn.CrossEntropyLoss()
        self.feat_dim = backbone.fc.in_features
        self.feat_channels = backbone.fc.in_features

    def forward(self, x, y=None):
        out = self.backbone.conv1(x)
        out = self.backbone.bn1(out)
        out = self.backbone.relu(out)
        out = self.backbone.maxpool(out)

        out = self.backbone.layer1(out)
        feat_layer1 = out
        out = self.backbone.layer2(out)
        feat_layer2 = out
        out = self.backbone.layer3(out)
        feat_layer3 = out
        feat_4d = self.backbone.layer4(out)

        gp = self.backbone.avgpool(feat_4d)
        feat_2d = torch.flatten(gp, 1)
        logit = self.backbone.fc(feat_2d)

        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        return {
            "feat_4d": feat_4d,
            "feat_2d": feat_2d,
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


def create_resnet_adapter(num_classes=100, pretrained=True, small_input=False, cfg: Optional[dict] = None):
    if pretrained:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = resnet50(weights=None)

    if small_input:
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return StudentResNetWrapper(model, cfg=cfg)


def create_resnet152_adapter(num_classes=100, pretrained=True, small_input=False, cfg: Optional[dict] = None):
    if pretrained:
        model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
    else:
        model = resnet152(weights=None)

    if small_input:
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return StudentResNetWrapper(model, cfg=cfg)
